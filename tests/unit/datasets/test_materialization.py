from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.config.loading import active_config
from fedorbit.datasets.common import AdapterSchema, FieldRole
from fedorbit.datasets.materialization import (
    MaterializationError,
    PreprocessingFitAccess,
    assign_materialized_splits,
    require_safe_memory_budget,
    retained_local_classes,
)
from fedorbit.datasets.preprocessing import NormalizedFeatureVector, NormalizedRow
from fedorbit.types import (
    DatasetId,
    FineLabel,
    NormalizedGroupIdentifier,
    PreprocessingFitStage,
    Split,
    TabularColumnName,
)


class _FakeVirtualMemory:
    def __init__(self, available: int) -> None:
        self.available = available


def _write_file_of_size(path: Path, size_bytes: int) -> Path:
    path.write_bytes(b"0" * size_bytes)
    return path


def test_memory_guard_refuses_when_estimated_peak_exceeds_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big_file = _write_file_of_size(tmp_path / "big.csv", 10_000_000)
    monkeypatch.setattr(
        "fedorbit.infrastructure.runtime.psutil.virtual_memory",
        lambda: _FakeVirtualMemory(available=1_000_000),
    )
    with pytest.raises(MaterializationError):
        require_safe_memory_budget(DatasetId.TON_IOT_NETWORK, (big_file,))


def test_memory_guard_passes_when_estimated_peak_fits_the_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_file = _write_file_of_size(tmp_path / "small.csv", 1_000)
    monkeypatch.setattr(
        "fedorbit.infrastructure.runtime.psutil.virtual_memory",
        lambda: _FakeVirtualMemory(available=10_000_000_000),
    )
    require_safe_memory_budget(DatasetId.TON_IOT_WINDOWS10_HOST, (small_file,))


def test_preprocessing_fit_access_rejects_non_train_split() -> None:
    with pytest.raises(MaterializationError, match="non-TRAIN"):
        PreprocessingFitAccess(
            PreprocessingFitStage.FEATURE_QUALITY,
            (Split.META,),
        )


def test_materialization_rejects_feature_duplicate_groups_with_conflicting_labels() -> None:
    feature = TabularColumnName("behavioral")
    schema = AdapterSchema(
        dataset_id=DatasetId.TON_IOT_NETWORK,
        feature_order=(feature,),
        roles={feature: FieldRole.BEHAVIORAL_NUMERIC},
    )
    rows = tuple(
        NormalizedRow(
            features=NormalizedFeatureVector({feature: 1.0}),
            label=FineLabel(label),
            timestamp_fraction=float(index),
            group_id=NormalizedGroupIdentifier(""),
        )
        for index, label in enumerate(("normal", "attack"))
    )
    with pytest.raises(MaterializationError, match="conflicting labels"):
        assign_materialized_splits(schema, rows)


def test_local_class_manifest_retains_normal_and_only_supported_attack_classes() -> None:
    feature = TabularColumnName("behavioral")
    threshold = (
        active_config().scientific.transfer_support.local_prediction_attack_class_total_rows_minimum
    )

    def row(label: str) -> NormalizedRow:
        return NormalizedRow(
            features=NormalizedFeatureVector({feature: 1.0}),
            label=FineLabel(label),
            timestamp_fraction=0.0,
            group_id=NormalizedGroupIdentifier(""),
        )

    rows = (
        row("normal"),
        *(row("retained_attack") for _ in range(threshold)),
        *(row("excluded_attack") for _ in range(threshold - 1)),
    )
    manifest = retained_local_classes(rows)

    assert manifest.class_names == (FineLabel("normal"), FineLabel("retained_attack"))
    assert manifest.excluded_classes == ((FineLabel("excluded_attack"), threshold - 1),)


def test_local_class_eligibility_uses_only_the_train_split() -> None:
    feature = TabularColumnName("behavioral")
    schema = AdapterSchema(
        dataset_id=DatasetId.TON_IOT_NETWORK,
        feature_order=(feature,),
        roles={feature: FieldRole.BEHAVIORAL_NUMERIC},
    )
    threshold = (
        active_config().scientific.transfer_support.local_prediction_attack_class_total_rows_minimum
    )

    def row(label: str, index: int) -> NormalizedRow:
        return NormalizedRow(
            features=NormalizedFeatureVector({feature: float(index)}),
            label=FineLabel(label),
            timestamp_fraction=float(index),
            group_id=NormalizedGroupIdentifier(""),
        )

    rows = (
        *(row("normal", index) for index in range(2)),
        *(row("attack", index + 2) for index in range(threshold)),
    )
    split_result = assign_materialized_splits(schema, rows)
    manifest = retained_local_classes(split_result.buckets[Split.TRAIN])
    train_attack_count = sum(
        row.label == FineLabel("attack") for row in split_result.buckets[Split.TRAIN]
    )

    assert train_attack_count < threshold
    assert manifest.class_names == (FineLabel("normal"),)
    assert manifest.excluded_classes == ((FineLabel("attack"), train_attack_count),)
