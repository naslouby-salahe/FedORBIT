from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.config.loading import REPOSITORY_ROOT, section_digest
from fedorbit.config.models import FedorbitConfig
from fedorbit.infrastructure.artifacts import manifest_path
from fedorbit.infrastructure.workspace import Workspace
from fedorbit.pipeline.execution import (
    EvidenceUnavailableError,
    ExecutionContext,
    analysis_provenance,
    analysis_state,
    execute_analysis,
    execute_experiment,
    experiment_state,
    load_devices,
    load_verified_records,
    prepare_dataset,
)
from fedorbit.reporting import write_report
from fedorbit.types import DatasetId, EvidenceState, ExperimentId, OverwritePolicy
from tests.support import FIXTURE_DEVICES, synthetic_config, synthetic_nbaiot_directory

PACKAGE = REPOSITORY_ROOT / "src" / "fedorbit"
COLD = ExperimentId.COLD_START_LADDER


def build(tmp_path: Path) -> tuple[FedorbitConfig, ExecutionContext]:
    raw = synthetic_nbaiot_directory(tmp_path / "raw", FIXTURE_DEVICES, seed=3)
    workspace = Workspace(root=tmp_path / "workspace", results_directory=tmp_path / "results")
    context = ExecutionContext(
        repository_root=REPOSITORY_ROOT,
        raw_directory=raw,
        workspace=workspace,
        package_directory=PACKAGE,
    )
    return synthetic_config(FIXTURE_DEVICES), context


def test_full_pipeline_from_raw_files_to_verified_report(tmp_path: Path) -> None:
    config, context = build(tmp_path)
    prepared = prepare_dataset(config, context, DatasetId.NBAIOT, OverwritePolicy.REUSE)
    assert set(prepared.values()) == {EvidenceState.VALID}
    assert execute_experiment(config, context, COLD, OverwritePolicy.REUSE) is EvidenceState.VALID
    experiment = config.experiment(COLD)
    assert experiment_state(config, context, experiment) is EvidenceState.VALID
    analysis = execute_analysis(config, context, OverwritePolicy.REUSE)
    assert analysis_state(config, context) is EvidenceState.VALID
    assert analysis.contrasts
    written = write_report(
        config,
        analysis,
        context.workspace.results_directory,
        analysis_provenance(config, context),
        "revision",
    )
    assert {path.name for path in written} >= {"contrasts.csv", "devices.csv", "conditions.csv"}
    assert all(manifest_path(path).is_file() for path in written)


def test_valid_evidence_is_reused_and_overwrite_recomputes(tmp_path: Path) -> None:
    config, context = build(tmp_path)
    prepare_dataset(config, context, DatasetId.NBAIOT, OverwritePolicy.REUSE)
    execute_experiment(config, context, COLD, OverwritePolicy.REUSE)
    records_path = context.workspace.records_path(COLD)
    first = records_path.stat().st_mtime_ns
    execute_experiment(config, context, COLD, OverwritePolicy.REUSE)
    assert records_path.stat().st_mtime_ns == first
    execute_experiment(config, context, COLD, OverwritePolicy.OVERWRITE)
    assert records_path.stat().st_mtime_ns > first


def test_scientific_configuration_change_invalidates_records(tmp_path: Path) -> None:
    config, context = build(tmp_path)
    prepare_dataset(config, context, DatasetId.NBAIOT, OverwritePolicy.REUSE)
    execute_experiment(config, context, COLD, OverwritePolicy.REUSE)
    changed = config.model_copy(
        update={
            "operating_point": config.operating_point.model_copy(
                update={"nominal_false_positive_rate": 0.05}
            )
        }
    )
    assert experiment_state(changed, context, changed.experiment(COLD)) is EvidenceState.STALE
    assert section_digest(config.operating_point) != section_digest(changed.operating_point)


def test_statistics_only_change_keeps_records_but_invalidates_analysis(tmp_path: Path) -> None:
    config, context = build(tmp_path)
    prepare_dataset(config, context, DatasetId.NBAIOT, OverwritePolicy.REUSE)
    execute_experiment(config, context, COLD, OverwritePolicy.REUSE)
    execute_analysis(config, context, OverwritePolicy.REUSE)
    changed = config.model_copy(
        update={"statistics": config.statistics.model_copy(update={"alpha": 0.01})}
    )
    assert experiment_state(changed, context, changed.experiment(COLD)) is EvidenceState.VALID
    assert analysis_state(changed, context) is EvidenceState.STALE


def test_corrupted_prepared_data_blocks_downstream_work(tmp_path: Path) -> None:
    config, context = build(tmp_path)
    prepare_dataset(config, context, DatasetId.NBAIOT, OverwritePolicy.REUSE)
    victim = context.workspace.prepared_path(DatasetId.NBAIOT, FIXTURE_DEVICES[0])
    victim.write_bytes(b"corrupted")
    with pytest.raises(EvidenceUnavailableError):
        load_devices(config, context, DatasetId.NBAIOT)
    with pytest.raises(EvidenceUnavailableError):
        execute_experiment(config, context, COLD, OverwritePolicy.REUSE)


def test_changed_prepared_data_makes_records_stale(tmp_path: Path) -> None:
    config, context = build(tmp_path)
    prepare_dataset(config, context, DatasetId.NBAIOT, OverwritePolicy.REUSE)
    execute_experiment(config, context, COLD, OverwritePolicy.REUSE)
    prepare_dataset(config, context, DatasetId.NBAIOT, OverwritePolicy.OVERWRITE)
    state = experiment_state(config, context, config.experiment(COLD))
    assert state in {EvidenceState.VALID, EvidenceState.STALE}


def test_records_are_unavailable_until_produced(tmp_path: Path) -> None:
    config, context = build(tmp_path)
    prepare_dataset(config, context, DatasetId.NBAIOT, OverwritePolicy.REUSE)
    with pytest.raises(EvidenceUnavailableError):
        load_verified_records(config, context, config.experiment(COLD))
    assert analysis_state(config, context) is EvidenceState.INCOMPLETE


def test_tampered_records_are_rejected_by_analysis(tmp_path: Path) -> None:
    config, context = build(tmp_path)
    prepare_dataset(config, context, DatasetId.NBAIOT, OverwritePolicy.REUSE)
    execute_experiment(config, context, COLD, OverwritePolicy.REUSE)
    path = context.workspace.records_path(COLD)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(EvidenceUnavailableError):
        execute_analysis(config, context, OverwritePolicy.REUSE)
