from __future__ import annotations

import json
from pathlib import Path

from fedorbit.analysis.records import MetricDirection, MetricRecord
from fedorbit.experiments import solvers as experiment_solvers
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.experiments.solvers import (
    CompletedMetricArtifact,
    WorkStructureTrendState,
    completed_scalability_metric_artifacts,
    scalability_condition_coordinates,
    scalability_predicted_work,
    synthetic_solver_instance,
    work_structure_strata,
)
from fedorbit.experiments.synthesis import completed_experiment_metric_records
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.manifests import ReusableArtifactManifest
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.optimization.correspondence import (
    BlockNodeCounts,
    active_image_assignment_count,
)
from fedorbit.types import (
    SYNTHETIC_DIRECTED_PAIR,
    ArtifactIdentifier,
    ArtifactState,
    EvaluationConditionName,
    ExperimentName,
    MetricId,
    MetricUnit,
    OverwritePolicy,
    RandomSeed,
    ScalabilityBlockPattern,
    Sha256Digest,
    SupportSize,
    TransferMethod,
)


def _record(
    condition: str,
    seed: RandomSeed,
    metric_name: MetricId,
    metric_value: float | None,
    *,
    valid: bool = True,
) -> MetricRecord:
    return MetricRecord(
        experiment=ExperimentName.SCALABILITY_AND_EFFICIENCY,
        pair=SYNTHETIC_DIRECTED_PAIR,
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        condition=EvaluationConditionName(condition),
        seed=seed,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=MetricUnit.SECONDS,
        direction=MetricDirection.DESCRIPTIVE,
        evaluation_class_set_sha256=Sha256Digest("a" * 64),
        input_artifact_ids=(ArtifactIdentifier("fixture"),),
        dependency_fingerprint_sha256=Sha256Digest("b" * 64),
        valid=valid,
        invalid_reason=None,
    )


def _artifact(
    condition: str,
    seed: RandomSeed,
    metric_name: MetricId,
    metric_value: float | None,
    support: int | None = 1,
) -> CompletedMetricArtifact:
    return CompletedMetricArtifact(
        ArtifactIdentifier(f"{condition}-{seed}-{metric_name.value}"),
        _record(condition, seed, metric_name, metric_value),
        None if support is None else SupportSize(support),
    )


def _cell(
    condition: str,
    seed: int,
    support: int,
    time_seconds: float,
    work: float,
) -> tuple[CompletedMetricArtifact, CompletedMetricArtifact]:
    return (
        _artifact(condition, seed, MetricId.PREDICTED_WORK_COORDINATE, work, support),
        _artifact(condition, seed, MetricId.WALL_TIME, time_seconds, support),
    )


def test_scalability_predicted_work_uses_the_registered_active_image_assignment_count() -> None:
    kernel_instance = synthetic_solver_instance(
        6, ScalabilityBlockPattern.MAXIMALLY_SKEWED, 2, 1103
    )
    assert kernel_instance.blocks.padded_size_tuple == (5, 1)
    expected = float(
        active_image_assignment_count(
            kernel_instance.blocks, BlockNodeCounts(kernel_instance.blocks, (2, 0))
        )
        * (5**3 + 1**3)
    )
    assert expected == 20 * 126
    assert scalability_predicted_work(kernel_instance.blocks, kernel_instance.action) == expected
    assert scalability_predicted_work(kernel_instance.blocks, kernel_instance.action) != float(
        6 * 126
    )


def test_scalability_condition_coordinates_rejects_unregistered_conditions() -> None:
    coordinates = scalability_condition_coordinates(EvaluationConditionName("k8-balanced"))
    assert coordinates is not None
    assert coordinates.k == 8
    assert coordinates.block_pattern is ScalabilityBlockPattern.BALANCED
    assert scalability_condition_coordinates(EvaluationConditionName("k8-dense-balanced")) is None
    assert scalability_condition_coordinates(EvaluationConditionName("real-a-to-b")) is None
    assert scalability_condition_coordinates(EvaluationConditionName("k8-unknown_pattern")) is None


def test_work_structure_strata_require_the_registered_minimum_distinct_k() -> None:
    artifacts: list[CompletedMetricArtifact] = []
    for index, k in enumerate((6, 8, 10, 12, 16)):
        artifacts.extend(_cell(f"k{k}-balanced", 1103, 1, 1.0 + index, float(k)))
    for index, k in enumerate((6, 8, 10)):
        artifacts.extend(_cell(f"k{k}-balanced", 1103, 2, 2.0 + index, float(k)))
    artifacts.extend(_cell("k20-balanced", 1103, 3, 9.0, 20.0))
    artifacts.extend(_cell("real-windows-to-network", 1103, 1, 5.0, 30.0))
    strata = work_structure_strata(tuple(artifacts))
    by_key = {(stratum.block_pattern, stratum.support): stratum for stratum in strata}
    assert set(by_key) == {
        (ScalabilityBlockPattern.BALANCED, SupportSize(1)),
        (ScalabilityBlockPattern.BALANCED, SupportSize(2)),
        (ScalabilityBlockPattern.BALANCED, SupportSize(3)),
    }
    assert by_key[(ScalabilityBlockPattern.BALANCED, SupportSize(3))].correlation is None
    eligible = by_key[(ScalabilityBlockPattern.BALANCED, SupportSize(1))]
    assert len(eligible.points) == 5
    assert eligible.correlation is not None
    assert [point.k for point in eligible.points] == [6, 8, 10, 12, 16]
    assert [point.predicted_work for point in eligible.points] == [6.0, 8.0, 10.0, 12.0, 16.0]
    assert [point.median_runtime_seconds for point in eligible.points] == [1.0, 2.0, 3.0, 4.0, 5.0]
    insufficient = by_key[(ScalabilityBlockPattern.BALANCED, SupportSize(2))]
    assert len(insufficient.points) == 3
    assert insufficient.correlation is None


def test_work_structure_strata_exclude_timeouts_and_aggregate_seed_medians() -> None:
    artifacts: list[CompletedMetricArtifact] = []
    for k in (6, 8, 10, 12, 16):
        artifacts.extend(_cell(f"k{k}-balanced", 1103, 1, float(k), float(k)))
        artifacts.extend(_cell(f"k{k}-balanced", 2207, 1, float(k) * 3.0, float(k)))
    artifacts.append(_artifact("k20-balanced", 1103, MetricId.TIMEOUT_INDICATOR, 1.0, 1))
    artifacts.extend(_cell("k20-balanced", 1103, 1, 1000.0, 20.0))
    strata = work_structure_strata(tuple(artifacts))
    assert len(strata) == 1
    stratum = strata[0]
    assert [point.k for point in stratum.points] == [6, 8, 10, 12, 16]
    assert stratum.points[0].median_runtime_seconds == 12.0
    assert stratum.points[2].median_runtime_seconds == 20.0


def _request(experiment: ExperimentName) -> ExperimentExecutionRequest:
    catalogue = build_catalogue()
    return ExperimentExecutionRequest(
        experiment=experiment,
        definition=catalogue.definition(experiment),
        overwrite_policy=OverwritePolicy.REPLACE,
    )


def test_insufficient_trend_evidence_is_persisted_without_an_eligible_stratum(
    tmp_path: Path,
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = _request(ExperimentName.SCALABILITY_AND_EFFICIENCY)
    assert completed_scalability_metric_artifacts(store) == ()
    experiment_solvers.persist_work_structure_trend(store, layout, request)
    records = tuple(
        record
        for record in completed_experiment_metric_records(
            store, ExperimentName.SCALABILITY_AND_EFFICIENCY
        )
        if record.metric_name is MetricId.WORK_STRUCTURE_SPEARMAN
    )
    assert len(records) == 1
    assert not records[0].valid
    assert records[0].metric_value is None
    assert records[0].invalid_reason == WorkStructureTrendState.INSUFFICIENT_TREND_EVIDENCE.value
    assert records[0].input_artifact_ids
    manifests = store.all_manifests()
    assert any("work-structure-trend" in Path(path).parts[-1] for path in _payload_paths(manifests))
    for manifest in manifests:
        assert store.resolve(manifest.artifact_id).state == ArtifactState.COMPLETED
    summary = next(
        Path(path)
        for path in _payload_paths(manifests)
        if "work-structure-trend" in Path(path).parts[-1]
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["strata"] == []
    assert "active-image assignment count" in payload["predicted_work_coordinate"]


def _payload_paths(
    manifests: tuple[ReusableArtifactManifest, ...],
) -> tuple[str, ...]:
    return tuple(path for manifest in manifests for path in manifest.payload_paths)
