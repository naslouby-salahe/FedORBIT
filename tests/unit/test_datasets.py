from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fedorbit.config.models import GothamConfig
from fedorbit.datasets.device_data import (
    DatasetValidationError,
    DeviceData,
    chronological_split,
    purge_exact_duplicates,
)
from fedorbit.datasets.gotham import PacketColumns, window_features
from fedorbit.datasets.nbaiot import load_nbaiot_device, signed_log
from fedorbit.datasets.preparation import parse_device, serialise_device
from fedorbit.types import DatasetId, DeviceName, RandomSeed
from tests.support import (
    FIXTURE_DEVICES,
    synthetic_config,
    synthetic_device,
    synthetic_nbaiot_directory,
)


def test_chronological_split_preserves_order() -> None:
    rows = np.arange(30, dtype=np.float64).reshape(10, 3)
    support, test = chronological_split(rows, 0.6)
    assert len(support) == 6 and len(test) == 4
    assert np.array_equal(np.vstack([support, test]), rows)


def test_purge_removes_only_exact_duplicates_of_support_rows() -> None:
    support = np.array([[1.0, 2.0], [3.0, 4.0]])
    test = np.array([[1.0, 2.0], [1.0, 2.0000001], [5.0, 6.0]])
    kept, purged = purge_exact_duplicates(support, test)
    assert purged == 1
    assert kept.tolist() == [[1.0, 2.0000001], [5.0, 6.0]]


def test_device_data_rejects_non_finite_and_mismatched_widths() -> None:
    good = np.ones((3, 2))
    with pytest.raises(DatasetValidationError):
        DeviceData(DatasetId.NBAIOT, DeviceName("d"), good, np.array([[np.nan, 1.0]]), good, 0)
    with pytest.raises(DatasetValidationError):
        DeviceData(DatasetId.NBAIOT, DeviceName("d"), good, np.ones((3, 3)), good, 0)


def test_target_requires_attack_and_benign_test_rows() -> None:
    data = synthetic_device("a", 0, 1.0)
    assert data.is_evaluation_target
    benign_only = DeviceData(
        DatasetId.GOTHAM, DeviceName("b"), data.support_pool, data.test_benign, np.zeros((0, 6)), 0
    )
    assert not benign_only.is_evaluation_target


def test_signed_log_is_odd_and_monotone() -> None:
    values = np.array([-5.0, 0.0, 5.0])
    transformed = signed_log(values)
    assert transformed[1] == 0.0
    assert transformed[0] == -transformed[2]


def test_prepared_round_trip() -> None:
    data = synthetic_device("round_trip", 3, 2.0)
    restored = parse_device(DatasetId.NBAIOT, data.device, serialise_device(data))
    assert np.array_equal(restored.support_pool, data.support_pool)
    assert np.array_equal(restored.test_attack, data.test_attack)
    assert restored.purged_duplicate_rows == data.purged_duplicate_rows


def test_nbaiot_loader_splits_purges_and_samples_deterministically(tmp_path: Path) -> None:
    devices = FIXTURE_DEVICES[:2]
    raw = synthetic_nbaiot_directory(tmp_path, devices, seed=5)
    config = synthetic_config(devices)
    first = load_nbaiot_device(raw, devices[0], config.datasets.nbaiot, RandomSeed(11))
    second = load_nbaiot_device(raw, devices[0], config.datasets.nbaiot, RandomSeed(11))
    other = load_nbaiot_device(raw, devices[0], config.datasets.nbaiot, RandomSeed(12))
    assert len(first.support_pool) == int(900 * config.datasets.nbaiot.benign_support_fraction)
    assert len(first.test_attack) == 10 * config.datasets.nbaiot.attack_rows_per_file
    assert np.array_equal(first.test_attack, second.test_attack)
    assert not np.array_equal(first.test_attack, other.test_attack)


def test_window_features_on_a_hand_built_packet_stream() -> None:
    packets = PacketColumns(
        window=np.array([10, 10, 10, 11, 11], dtype=np.int64),
        length=np.array([100.0, 200.0, 300.0, 50.0, 50.0]),
        destination_code=np.array([0, 0, 1, -1, 2], dtype=np.int64),
        protocol=np.array([6.0, 6.0, 17.0, 1.0, 6.0]),
        syn_only=np.array([True, False, False, False, True]),
        port_code=np.array([80, 80, 53, -1, 22], dtype=np.int64),
        ttl=np.array([64.0, 64.0, 32.0, np.nan, np.nan]),
        attack=np.array([False, False, False, True, True]),
    )
    matrix, attack_fraction = window_features(packets)
    assert matrix.shape == (2, 11)
    first = matrix[0]
    assert first[0] == pytest.approx(np.log1p(3.0))
    assert first[1] == pytest.approx(200.0)
    assert first[2] == pytest.approx(100.0)
    assert first[4] == pytest.approx(2.0 / 3.0)
    assert first[5] == pytest.approx(1.0 / 3.0)
    assert first[7] == pytest.approx(1.0 / 3.0)
    assert first[8] == pytest.approx((64.0 + 64.0 + 32.0) / 3.0)
    assert first[9] == pytest.approx(np.log1p(2.0))
    assert first[10] == pytest.approx(np.log1p(2.0))
    second = matrix[1]
    assert second[8] == 0.0
    assert second[9] == pytest.approx(np.log1p(1.0))
    assert attack_fraction.tolist() == [0.0, 1.0]


def test_gotham_configuration_bounds() -> None:
    with pytest.raises(ValueError):
        GothamConfig.model_validate(
            {
                "relative_directory": "x",
                "window_seconds": 0,
                "attack_window_minimum_fraction": 0.5,
                "benign_support_fraction": 0.5,
                "benign_label": "Benign",
                "excluded_label": "Unknown",
            }
        )
