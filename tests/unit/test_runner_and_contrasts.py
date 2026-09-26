from __future__ import annotations

import numpy as np
import pytest

from fedorbit.analysis.contrasts import (
    ContrastStatus,
    analyse_contrast,
    index_cells,
    reference_aurocs,
    summarise_conditions,
)
from fedorbit.analysis.synthesis import analyse_study
from fedorbit.config.loading import load_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.study.records import (
    AnyRecord,
    CellRecord,
    IneligibleDeviceRecord,
    InfeasibleSupportRecord,
)
from fedorbit.study.runner import log_scale_gap, run_experiment, scale_diagnostics
from fedorbit.types import (
    ConditionName,
    ContrastFamily,
    DatasetId,
    DeviceName,
    ExperimentId,
    ReplicateIndex,
    SupportSize,
)
from tests.support import FIXTURE_DEVICES, synthetic_config, synthetic_device

COLD = ExperimentId.COLD_START_LADDER


def fixture_config() -> FedorbitConfig:
    return synthetic_config(FIXTURE_DEVICES)


def run_fixture() -> tuple[FedorbitConfig, list[AnyRecord]]:
    config = fixture_config()
    devices = {
        DeviceName(f"device_{index}"): synthetic_device(f"device_{index}", index, 1.0 + index)
        for index in range(4)
    }
    return config, run_experiment(config, config.experiment(COLD), devices)


def test_runner_records_every_condition_replicate_and_reference() -> None:
    config, records = run_fixture()
    experiment = config.experiment(COLD)
    cells = [record for record in records if isinstance(record, CellRecord)]
    reference_name = ConditionName("local-full-support")
    per_cell = len(experiment.conditions) - 1
    expected = 4 * len(experiment.support_sizes) * experiment.replicates * per_cell + 4
    assert len(cells) == expected
    assert {record.condition for record in cells if record.replicate == 0} >= {reference_name}
    references = [record for record in cells if record.condition == reference_name]
    assert all(record.support_size == 400 for record in references)


def test_runner_is_deterministic() -> None:
    _, first = run_fixture()
    _, second = run_fixture()
    assert first == second


def test_paired_conditions_share_windows_so_local_conditions_coincide_when_equal() -> None:
    _, records = run_fixture()
    cells = [record for record in records if isinstance(record, CellRecord)]
    full = {
        (record.device, record.support_size, record.replicate): record.auroc
        for record in cells
        if record.condition == "local-standardised"
    }
    assert len(full) == 4 * 2 * 4


def test_infeasible_support_is_recorded_not_dropped() -> None:
    config = fixture_config()
    small = {
        DeviceName("tiny"): synthetic_device("tiny", 1, 1.0, support_rows=40),
        DeviceName("other"): synthetic_device("other", 2, 1.0, support_rows=400),
    }
    records = run_experiment(config, config.experiment(COLD), small)
    infeasible = [record for record in records if isinstance(record, InfeasibleSupportRecord)]
    assert [(record.device, record.support_size) for record in infeasible] == [("tiny", 60)]
    assert infeasible[0].available_rows == 40


def test_devices_without_attack_rows_only_serve_as_partners() -> None:
    config = fixture_config()
    target = synthetic_device("target", 1, 1.0)
    partner = synthetic_device("partner", 2, 1.0)
    benign_only = type(partner)(
        dataset=DatasetId.NBAIOT,
        device=DeviceName("benign_only"),
        support_pool=partner.support_pool,
        test_benign=partner.test_benign,
        test_attack=np.zeros((0, partner.support_pool.shape[1])),
        purged_duplicate_rows=0,
    )
    records = run_experiment(
        config,
        config.experiment(COLD),
        {DeviceName("target"): target, DeviceName("benign_only"): benign_only},
    )
    assert {record.device for record in records if isinstance(record, CellRecord)} == {"target"}


def test_scale_diagnostics_detect_scale_instability() -> None:
    generator = np.random.default_rng(0)
    pool = generator.normal(0.0, 1.0, size=(1000, 3))
    stable = scale_diagnostics(pool[:400], pool, 1e-6)
    shifted = scale_diagnostics(pool[:400] * 5.0, pool, 1e-6)
    assert shifted.scale_mismatch > stable.scale_mismatch + 1.0
    assert log_scale_gap(np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1e-6) == 0.0


def cell(device: str, condition: str, replicate: int, value: float, size: int = 30) -> CellRecord:
    return CellRecord(
        experiment=COLD,
        dataset=DatasetId.NBAIOT,
        device=DeviceName(device),
        support_size=SupportSize(size),
        replicate=ReplicateIndex(replicate),
        condition=ConditionName(condition),
        auroc=value,
        scale_mismatch=0.5,
        split_half_instability=0.2,
    )


def constructed_records(deltas: dict[str, float]) -> list[AnyRecord]:
    records: list[AnyRecord] = []
    for device, delta in deltas.items():
        records.append(cell(device, "local-full-support", 0, 0.97, size=400))
        for replicate in range(3):
            base = 0.80 + 0.01 * replicate
            records.append(cell(device, "local-standardised", replicate, base))
            records.append(cell(device, "shared-marginals", replicate, base + delta))
    return records


def test_contrast_reports_paired_deltas_signs_and_recovery() -> None:
    config = load_config()
    experiment = config.experiment(COLD)
    deltas = {f"d{index}": 0.05 + 0.01 * index for index in range(8)}
    deltas["d8"] = -0.02
    index = index_cells(constructed_records(deltas))
    spec = next(
        item for item in experiment.contrasts if item.name == "marginals-over-local-standardised"
    )
    result = analyse_contrast(config, experiment, spec, SupportSize(30), index)
    assert result.status is ContrastStatus.ESTIMATED
    assert result.positive_devices == 8 and result.negative_devices == 1
    assert result.mean_delta == pytest.approx(np.mean(list(deltas.values())))
    assert result.signed_rank_p is not None and result.signed_rank_p < 0.01
    assert result.mean_recovery is not None
    assert all(delta.recovery is not None for delta in result.device_deltas)
    first = result.device_deltas[0]
    assert first.control_auroc == pytest.approx(0.81)
    assert first.treatment_auroc == pytest.approx(0.81 + deltas["d0"])


def test_saturated_stratum_selects_by_reference_headroom() -> None:
    config = load_config()
    experiment = config.experiment(COLD)
    records = constructed_records({"a": 0.01, "b": 0.02})
    records = [
        cell("a", "local-full-support", 0, 0.999, size=400)
        if isinstance(record, CellRecord)
        and record.device == "a"
        and record.condition == "local-full-support"
        else record
        for record in records
    ]
    index = index_cells(records)
    assert reference_aurocs(experiment, index) == pytest.approx({"a": 0.999, "b": 0.97})
    spec = next(
        item
        for item in experiment.contrasts
        if item.name == "marginals-over-local-standardised-unsaturated"
    )
    result = analyse_contrast(config, experiment, spec, SupportSize(30), index)
    assert [delta.device for delta in result.device_deltas] == ["b"]


def test_contrast_without_eligible_devices_is_reported_as_such() -> None:
    config = load_config()
    experiment = config.experiment(COLD)
    spec = experiment.contrasts[0]
    result = analyse_contrast(config, experiment, spec, SupportSize(30), {})
    assert result.status is ContrastStatus.NO_ELIGIBLE_DEVICES
    assert result.signed_rank_p is None and result.mean_delta is None


def test_zero_deltas_are_ties_not_evidence() -> None:
    config = load_config()
    experiment = config.experiment(COLD)
    index = index_cells(constructed_records({"a": 0.0, "b": 0.0}))
    spec = experiment.contrasts[0]
    result = analyse_contrast(config, experiment, spec, SupportSize(30), index)
    assert result.tied_devices == 2
    assert result.signed_rank_p is None and result.sign_p is None


def test_condition_summaries_exclude_the_reference_condition() -> None:
    config = load_config()
    experiment = config.experiment(COLD)
    index = index_cells(constructed_records({"a": 0.05, "b": 0.06}))
    summaries = summarise_conditions(config, experiment, index)
    assert {summary.condition for summary in summaries} == {
        "local-standardised",
        "shared-marginals",
    }
    marginals = next(summary for summary in summaries if summary.condition == "shared-marginals")
    assert marginals.devices == 2
    assert marginals.interval_lower <= marginals.mean_auroc <= marginals.interval_upper


def test_holm_is_applied_across_the_primary_family() -> None:
    config = load_config()
    records: dict[ExperimentId, list[AnyRecord]] = {
        experiment.id: [] for experiment in config.experiments
    }
    deltas = {f"d{index}": 0.03 + 0.01 * index for index in range(9)}
    generated: list[AnyRecord] = []
    for name in ("local-standardised", "local-unscaled", "local-standard-deviation-floor"):
        for device, delta in deltas.items():
            for replicate in range(3):
                generated.append(cell(device, name, replicate, 0.8))
                generated.append(cell(device, "shared-marginals", replicate, 0.8 + delta))
    records[COLD] = generated
    analysis = analyse_study(config, records)
    primary = [item for item in analysis.contrasts if item.family is ContrastFamily.PRIMARY]
    assert len(primary) == 3
    assert all(item.holm_adjusted_p is not None for item in primary)
    assert all(
        item.holm_adjusted_p is not None
        and item.signed_rank_p is not None
        and item.holm_adjusted_p >= item.signed_rank_p
        for item in primary
    )
    secondary = [item for item in analysis.contrasts if item.family is not ContrastFamily.PRIMARY]
    assert all(item.holm_adjusted_p is None for item in secondary)


def test_devices_with_too_little_test_support_are_recorded_as_ineligible() -> None:
    config = fixture_config()
    thin = synthetic_device("thin", 1, 1.0)
    thin = type(thin)(
        dataset=DatasetId.NBAIOT,
        device=DeviceName("thin"),
        support_pool=thin.support_pool,
        test_benign=thin.test_benign[:3],
        test_attack=thin.test_attack,
        purged_duplicate_rows=0,
    )
    solid = synthetic_device("solid", 2, 1.0)
    records = run_experiment(
        config,
        config.experiment(COLD),
        {DeviceName("thin"): thin, DeviceName("solid"): solid},
    )
    ineligible = [record for record in records if isinstance(record, IneligibleDeviceRecord)]
    assert [(record.device, record.benign_rows) for record in ineligible] == [("thin", 3)]
    assert ineligible[0].worst_case_standard_error > config.statistics.maximum_auroc_standard_error
    assert {record.device for record in records if isinstance(record, CellRecord)} == {"solid"}
