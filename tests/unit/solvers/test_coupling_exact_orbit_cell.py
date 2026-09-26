from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from fedorbit.experiments import solvers
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.experiments.solvers import (
    execute_synthetic_coupling_mechanism_validation,
    persist_exact_orbit_coupling_cell,
    synthetic_coupling_problem,
)
from fedorbit.experiments.synthetic import (
    CouplingCompatibility,
    CouplingGenerationError,
    CouplingInstanceRequest,
    generate_coupling_instance,
)
from fedorbit.infrastructure.artifacts import ArtifactStore, ExecutionError
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.optimization.correspondence import enumerate_block_permutations
from fedorbit.types import (
    PRINCIPAL_EVALUATION_CONDITION,
    ArtifactIdentifier,
    ArtifactState,
    EvaluationConditionName,
    ExperimentLocalMethod,
    ExperimentName,
    MetricId,
    OverwritePolicy,
)


def test_exact_orbit_coupling_cell_is_persisted_for_a_registered_synthetic_cell(
    tmp_path: Path,
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    catalogue = build_catalogue()
    request = ExperimentExecutionRequest(
        experiment=ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION,
        definition=catalogue.definition(ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION),
        overwrite_policy=OverwritePolicy.REPLACE,
    )
    instance = generate_coupling_instance(
        CouplingInstanceRequest(
            compatibility=CouplingCompatibility.JOINTLY_REALIZABLE,
            response_heterogeneity=1.0,
            directed_asymmetry=0.0,
            response_sparsity=1.0,
            block_pattern=(2, 2),
            support_size=2,
            seed=1103,
            instance_index=0,
        )
    )
    problem, orbit, alpha, hull = synthetic_coupling_problem(instance, 2)
    assert len(orbit) == len(tuple(enumerate_block_permutations(problem.blocks)))
    condition = EvaluationConditionName("jointly_realizable-1.0-0.0-1.0-2x2")
    persist_exact_orbit_coupling_cell(
        store, layout, request, condition, 2, 1103, problem, orbit, alpha, hull
    )
    manifests = store.all_manifests()
    assert len(manifests) == 1
    manifest = manifests[0]
    assert store.resolve(manifest.artifact_id).state == ArtifactState.COMPLETED
    payload_path = Path(manifest.payload_paths[0])
    assert payload_path.name.startswith("exact-orbit.")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["method"] == ExperimentLocalMethod.EXACT_ORBIT.value
    assert payload["orbit_size"] == len(orbit)
    assert payload["condition"] == condition
    assert payload["support"] == 2
    assert payload["seed"] == 1103
    metrics = payload["metrics"]
    for metric_name in (
        MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP,
        MetricId.ROBUST_COUPLING_VALUE_GAP,
        MetricId.COUPLING_UPPER_BOUND_DIAGNOSTIC,
        MetricId.COUPLING_ACTION_SET_SUPPORT,
    ):
        assert metric_name.value in metrics
    assert metrics[MetricId.COUPLING_ACTION_SET_SUPPORT.value] == float(problem.principal_support)


def test_coupling_generation_failure_fails_closed_with_cell_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    catalogue = build_catalogue()
    experiment = ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION
    request = ExperimentExecutionRequest(
        experiment=experiment,
        definition=catalogue.definition(experiment),
        overwrite_policy=OverwritePolicy.REPLACE,
    )

    def fail_generation(_request: CouplingInstanceRequest) -> object:
        raise CouplingGenerationError("fixture rejection")

    monkeypatch.setattr(solvers, "generate_coupling_instance", fail_generation)
    with pytest.raises(
        ExecutionError,
        match=r"registered synthetic coupling cell could not be generated .*support=.*seed=",
    ):
        execute_synthetic_coupling_mechanism_validation(store, layout, request)


def test_exact_orbit_cell_reuses_the_same_artifact_for_identical_inputs(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    catalogue = build_catalogue()
    request = ExperimentExecutionRequest(
        experiment=ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION,
        definition=catalogue.definition(ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION),
        overwrite_policy=OverwritePolicy.REUSE,
    )
    instance = generate_coupling_instance(
        CouplingInstanceRequest(
            compatibility=CouplingCompatibility.INCOMPATIBLE,
            response_heterogeneity=2.0,
            directed_asymmetry=1.0,
            response_sparsity=0.5,
            block_pattern=(2, 3),
            support_size=2,
            seed=2207,
            instance_index=0,
        )
    )
    problem, orbit, alpha, hull = synthetic_coupling_problem(instance, 2)
    problem = replace(problem, principal_support=2)
    condition = EvaluationConditionName("incompatible-2.0-1.0-0.5-2x3")
    persist_exact_orbit_coupling_cell(
        store, layout, request, condition, 2, 2207, problem, orbit, alpha, hull
    )
    persist_exact_orbit_coupling_cell(
        store, layout, request, condition, 2, 2207, problem, orbit, alpha, hull
    )
    assert len(store.all_manifests()) == 1


def test_exact_orbit_cell_records_real_packet_upstream_identities(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    catalogue = build_catalogue()
    request = ExperimentExecutionRequest(
        experiment=ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION,
        definition=catalogue.definition(ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION),
        overwrite_policy=OverwritePolicy.REPLACE,
    )
    instance = generate_coupling_instance(
        CouplingInstanceRequest(
            compatibility=CouplingCompatibility.JOINTLY_REALIZABLE,
            response_heterogeneity=1.0,
            directed_asymmetry=0.0,
            response_sparsity=1.0,
            block_pattern=(2, 2),
            support_size=2,
            seed=1103,
            instance_index=0,
        )
    )
    problem, orbit, alpha, hull = synthetic_coupling_problem(instance, 2)
    persist_exact_orbit_coupling_cell(
        store,
        layout,
        request,
        PRINCIPAL_EVALUATION_CONDITION.name,
        2,
        1103,
        problem,
        orbit,
        alpha,
        hull,
        input_artifact_ids=(ArtifactIdentifier("source-packet"),),
        declare_no_upstream_inputs=False,
    )
    manifest = store.all_manifests()[0]
    assert manifest.has_no_upstream_inputs is False
    assert manifest.upstream_artifact_ids == (ArtifactIdentifier("source-packet"),)
    payload = json.loads(Path(manifest.payload_paths[0]).read_text(encoding="utf-8"))
    assert payload["method"] == ExperimentLocalMethod.EXACT_ORBIT.value
    assert payload["input_artifact_ids"] == ["source-packet"]
