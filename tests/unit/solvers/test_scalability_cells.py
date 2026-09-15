from __future__ import annotations

import json
from pathlib import Path

import pytest

from fedorbit.analysis.records import MetricDirection
from fedorbit.config.loading import active_config
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.experiments.solvers import (
    REPEATED_TIMING_METHODS,
    persist_synthetic_benchmark_metric,
    persist_unavailable_registered_methods,
    synthetic_solver_instance,
)
from fedorbit.experiments.synthesis import completed_experiment_metric_records
from fedorbit.experiments.synthetic import ScalabilityGenerationError
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactState,
    EvaluationConditionName,
    ExperimentName,
    InvalidReason,
    MetricId,
    MetricUnit,
    OverwritePolicy,
    ScalabilityBlockPattern,
    SupportSize,
    TransferMethod,
)


def _request(experiment: ExperimentName) -> ExperimentExecutionRequest:
    catalogue = build_catalogue()
    return ExperimentExecutionRequest(
        experiment=experiment,
        definition=catalogue.definition(experiment),
        overwrite_policy=OverwritePolicy.REPLACE,
    )


def test_the_registered_generator_reports_infeasible_support_for_small_balanced_blocks() -> None:
    with pytest.raises(ScalabilityGenerationError) as error:
        synthetic_solver_instance(4, ScalabilityBlockPattern.BALANCED, 3, 1103)
    assert str(error.value) == "support size exceeds the largest coarse block"
    instance = synthetic_solver_instance(4, ScalabilityBlockPattern.BALANCED, 2, 1103)
    assert instance.blocks.padded_size_tuple == (2, 2)


def test_unavailable_benchmark_cells_are_persisted_for_every_registered_method(
    tmp_path: Path,
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = _request(ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK)
    methods = active_config().experiments.exact_sparse_solver_benchmark.methods
    with pytest.raises(ScalabilityGenerationError) as error:
        synthetic_solver_instance(4, ScalabilityBlockPattern.BALANCED, 3, 1103)
    persist_unavailable_registered_methods(
        store,
        layout,
        request.experiment,
        EvaluationConditionName("k4-balanced"),
        SupportSize(3),
        1103,
        methods,
        InvalidReason(str(error.value)),
        request.overwrite_policy,
    )
    records = completed_experiment_metric_records(
        store, ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK
    )
    assert len(records) == len(methods)
    assert {record.method for record in records} == set(methods)
    for record in records:
        assert record.metric_name is MetricId.CORRESPONDENCE_CERTIFICATE_VALIDITY
        assert not record.valid
        assert record.metric_value is None
        assert record.invalid_reason == "support size exceeds the largest coarse block"
        assert record.valid is False
    for manifest in store.all_manifests():
        assert store.resolve(manifest.artifact_id).state == ArtifactState.COMPLETED


def test_dense_scalability_cells_are_recorded_without_a_sparse_support_coordinate(
    tmp_path: Path,
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = _request(ExperimentName.SCALABILITY_AND_EFFICIENCY)
    node_count = active_config().experiments.scalability_and_efficiency.k_values[0]
    dense_instance = synthetic_solver_instance(
        node_count,
        ScalabilityBlockPattern.BALANCED,
        active_config().experiments.scalability_and_efficiency.exact_qap_supports[0],
        1103,
        node_count,
    )
    assert dense_instance.problem.actionable_nodes() == tuple(range(node_count))
    manifest = persist_synthetic_benchmark_metric(
        store,
        layout,
        request.experiment,
        EvaluationConditionName(f"k{node_count}-balanced-dense"),
        None,
        TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
        1103,
        MetricId.DENSE_BOUND_GAP,
        0.0,
        MetricUnit.SCORE,
        MetricDirection.DESCRIPTIVE,
        (ArtifactIdentifier("synthetic-generator"),),
        request.overwrite_policy,
    )
    assert manifest is not None
    assert "support" not in json.loads(manifest.semantic_producer_coordinates)
    assert "support-uncapped" in Path(manifest.payload_paths[0]).name
    assert store.resolve(manifest.artifact_id).state == ArtifactState.COMPLETED


def test_scalability_derived_cells_match_the_registered_real_timing_methods() -> None:
    config = active_config()
    scalability = config.experiments.scalability_and_efficiency
    seeds = config.scientific.randomness.confirmatory_seeds
    pairs = config.scientific.datasets.primary_directed_pairs
    catalogue = build_catalogue()
    exact_cells = (
        len(scalability.k_values)
        * len(scalability.block_patterns)
        * len(scalability.exact_qap_supports)
        * len(seeds)
        * 2
    )
    dense_cells = len(scalability.k_values) * len(scalability.block_patterns) * len(seeds)
    real_timing_cells = len(pairs) * len(seeds) * len(scalability.real_timing_methods)
    assert exact_cells == 960
    assert dense_cells == 160
    assert scalability.real_timing_methods == (
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.GENERIC_EXACT_QAP,
        TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
    )
    assert real_timing_cells == 180
    assert (
        catalogue.definition(ExperimentName.SCALABILITY_AND_EFFICIENCY).derived_planned_cells
        == (exact_cells + dense_cells + real_timing_cells)
        == 1300
    )
    assert frozenset({TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER}) == REPEATED_TIMING_METHODS
    assert set(scalability.real_timing_methods) >= REPEATED_TIMING_METHODS


def test_real_timing_dense_cells_are_recorded_without_a_sparse_support_coordinate(
    tmp_path: Path,
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = _request(ExperimentName.SCALABILITY_AND_EFFICIENCY)
    condition = EvaluationConditionName("real-ton_iot_windows10_host-to-ton_iot_network")
    principal_support = active_config().scientific.action.principal_sparse_support
    supported = persist_synthetic_benchmark_metric(
        store,
        layout,
        request.experiment,
        condition,
        SupportSize(principal_support),
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        1103,
        MetricId.WALL_TIME,
        1.0,
        MetricUnit.SECONDS,
        MetricDirection.LOWER_IS_BETTER,
        (ArtifactIdentifier("synthetic-generator"),),
        request.overwrite_policy,
    )
    dense = persist_synthetic_benchmark_metric(
        store,
        layout,
        request.experiment,
        condition,
        None,
        TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
        1103,
        MetricId.WALL_TIME,
        1.0,
        MetricUnit.SECONDS,
        MetricDirection.LOWER_IS_BETTER,
        (ArtifactIdentifier("synthetic-generator"),),
        request.overwrite_policy,
    )
    assert supported is not None and dense is not None
    supported_coordinates = json.loads(supported.semantic_producer_coordinates)
    dense_coordinates = json.loads(dense.semantic_producer_coordinates)
    assert supported_coordinates["support"] == principal_support
    assert "support" not in dense_coordinates
    assert supported_coordinates["method"] == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value
    assert dense_coordinates["method"] == TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK.value
    assert supported.artifact_id != dense.artifact_id
    records = completed_experiment_metric_records(store, ExperimentName.SCALABILITY_AND_EFFICIENCY)
    assert {record.method for record in records} == {
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
    }
    for manifest in store.all_manifests():
        assert store.resolve(manifest.artifact_id).state == ArtifactState.COMPLETED
