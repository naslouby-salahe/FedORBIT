from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.experiments.solvers import (
    persist_exact_orbit_coupling_cell,
    synthetic_coupling_problem,
)
from fedorbit.experiments.synthetic import (
    CouplingCompatibility,
    CouplingInstanceRequest,
    generate_coupling_instance,
)
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.optimization.correspondence import enumerate_block_permutations
from fedorbit.types import (
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
