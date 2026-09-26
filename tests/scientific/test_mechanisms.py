from __future__ import annotations

import numpy as np

from fedorbit.analysis.contrasts import index_cells
from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.device_data import DeviceData
from fedorbit.study.records import AnyRecord
from fedorbit.study.runner import run_experiment
from fedorbit.types import ConditionName, DatasetId, DeviceName, ExperimentId, FloatMatrix
from tests.support import FIXTURE_DEVICES, synthetic_config

FEATURES = 8


def bursty_benign(generator: np.random.Generator, rows: int) -> FloatMatrix:
    values = generator.normal(0.0, 1.0, size=(rows, FEATURES))
    values[:, :4] = generator.lognormal(0.0, 2.0, size=(rows, 4))
    return values


def bursty_device(name: str, seed: int) -> DeviceData:
    generator = np.random.default_rng(seed)
    attack = bursty_benign(generator, 300)
    attack[:, 4:] += 3.0
    return DeviceData(
        dataset=DatasetId.NBAIOT,
        device=DeviceName(name),
        support_pool=bursty_benign(generator, 900),
        test_benign=bursty_benign(generator, 900),
        test_attack=attack,
        purged_duplicate_rows=0,
    )


def correlated_device(name: str, seed: int) -> DeviceData:
    generator = np.random.default_rng(seed)

    def benign(rows: int) -> FloatMatrix:
        latent = generator.normal(0.0, 1.0, size=(rows, 1))
        return latent + generator.normal(0.0, 0.05, size=(rows, FEATURES))

    attack = generator.normal(0.0, 1.0, size=(300, FEATURES))
    return DeviceData(
        dataset=DatasetId.NBAIOT,
        device=DeviceName(name),
        support_pool=benign(900),
        test_benign=benign(600),
        test_attack=attack,
        purged_duplicate_rows=0,
    )


def independent_partner(name: str, seed: int) -> DeviceData:
    generator = np.random.default_rng(seed)
    return DeviceData(
        dataset=DatasetId.NBAIOT,
        device=DeviceName(name),
        support_pool=generator.normal(0.0, 1.0, size=(900, FEATURES)),
        test_benign=generator.normal(0.0, 1.0, size=(50, FEATURES)),
        test_attack=np.zeros((0, FEATURES)),
        purged_duplicate_rows=0,
    )


def mean_auroc(records: list[AnyRecord], condition: str, support_size: int) -> float:
    index = index_cells(records)
    values = [
        record.auroc
        for (_device, size, name), replicates in index.items()
        if name == ConditionName(condition) and size == support_size
        for record in replicates.values()
    ]
    return float(np.mean(values))


def config_for(devices: int) -> FedorbitConfig:
    return synthetic_config(FIXTURE_DEVICES[:devices])


def test_shared_marginals_repair_unstable_local_scale_at_small_support() -> None:
    config = config_for(3)
    devices = {
        DeviceName(f"bursty_{index}"): bursty_device(f"bursty_{index}", index) for index in range(4)
    }
    records = run_experiment(config, config.experiment(ExperimentId.COLD_START_LADDER), devices)
    local = mean_auroc(records, "local-standardised", 30)
    shared = mean_auroc(records, "shared-marginals", 30)
    assert shared > local + 0.03


def test_local_full_support_reference_is_at_least_as_good_as_small_support() -> None:
    config = config_for(3)
    devices = {
        DeviceName(f"bursty_{index}"): bursty_device(f"bursty_{index}", index) for index in range(4)
    }
    records = run_experiment(config, config.experiment(ExperimentId.COLD_START_LADDER), devices)
    small = mean_auroc(records, "local-standardised", 30)
    reference = mean_auroc(records, "local-full-support", 900)
    assert reference >= small


def test_sharing_covariance_from_mismatched_partners_harms_a_structured_device() -> None:
    config = config_for(3)
    devices = {
        DeviceName("structured"): correlated_device("structured", 1),
        DeviceName("partner_a"): independent_partner("partner_a", 2),
        DeviceName("partner_b"): independent_partner("partner_b", 3),
    }
    records = run_experiment(config, config.experiment(ExperimentId.COLD_START_LADDER), devices)
    marginals = mean_auroc(records, "shared-marginals", 60)
    covariance = mean_auroc(records, "shared-covariance-3000", 60)
    assert covariance < marginals - 0.05


def test_local_statistics_dominate_when_local_support_is_ample() -> None:
    config = config_for(3)
    devices = {
        DeviceName(f"bursty_{index}"): bursty_device(f"bursty_{index}", index) for index in range(4)
    }
    records = run_experiment(config, config.experiment(ExperimentId.COLD_START_LADDER), devices)
    reference = mean_auroc(records, "local-full-support", 900)
    marginals_small = mean_auroc(records, "shared-marginals", 30)
    assert abs(reference - marginals_small) < 0.25
