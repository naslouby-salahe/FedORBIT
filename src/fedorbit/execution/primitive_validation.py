from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import cast

import numpy as np

from fedorbit.artifacts.manifests import (
    CompletionManifest,
    ReusableArtifactManifest,
    artifact_id,
    completion_manifest_self_hash,
    file_sha256,
)
from fedorbit.artifacts.paths import WorkspaceLayout, experiment_workspace
from fedorbit.artifacts.provenance import (
    configuration_subset_digest,
    implementation_fingerprint,
    runtime_fingerprint,
    stage_dependency_fingerprint,
)
from fedorbit.artifacts.storage import ArtifactStore, atomic_write_json
from fedorbit.config.context import active_config
from fedorbit.domain.enums import ArtifactStage, ArtifactState, ExperimentName, TerminalState
from fedorbit.domain.records import ExperimentSeed, SemanticCell
from fedorbit.domain.serialization import StableJsonPayload, stable_json
from fedorbit.experiments.cells import experiment_relevance
from fedorbit.runtime.environment import environment_snapshot
from fedorbit.runtime.reproducibility import current_code_revision
from fedorbit.runtime.seeds import RandomSeed
from fedorbit.solvers.assignment import solve_minimum_cost_assignment
from fedorbit.synthetic.exactness import (
    ExactSeparatorInstanceRequest,
    generate_exact_separator_instance,
)


class PrimitiveValidationError(ValueError):
    pass


_EXPERIMENT = ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION
_STAGE = ArtifactStage.EVALUATION
_CONFIGURATION_SECTIONS = frozenset({"action", "generators"})
_PRODUCER_MODULE = "fedorbit.execution.primitive_validation"


def execute_primitive_validation(
    store: ArtifactStore, layout: WorkspaceLayout
) -> ReusableArtifactManifest:
    configuration = active_config()
    seed = ExperimentSeed(0)
    cell = SemanticCell(experiment=_EXPERIMENT, seed=seed)
    relevance = experiment_relevance(_EXPERIMENT)
    coordinates = cell.identity_json(relevance)
    fingerprint = stage_dependency_fingerprint(
        _STAGE,
        cell,
        relevance,
        (),
        _CONFIGURATION_SECTIONS,
        _PRODUCER_MODULE,
    )
    payload_path = _payload_path(layout, fingerprint)
    payload = _validation_payload(
        configuration.generators.exact_separator_theorem.block_patterns[0]
    )
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = configuration_subset_digest(_CONFIGURATION_SECTIONS)
    code_sha256 = implementation_fingerprint(_PRODUCER_MODULE)
    runtime_sha256 = runtime_fingerprint(_STAGE).sha256
    completion = _completion(
        coordinates,
        fingerprint,
        payload_path,
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
    )
    manifest = ReusableArtifactManifest.model_validate(
        {
            "artifact_id": artifact_id("other", payload, fingerprint),
            "artifact_type": "other",
            "semantic_producer_coordinates": coordinates,
            "producer_stage": _STAGE,
            "dependency_fingerprint_sha256": fingerprint,
            "upstream_artifact_ids": (),
            "applicable_configuration_sha256": configuration_sha256,
            "relevant_code_sha256": code_sha256,
            "material_runtime_sha256": runtime_sha256,
            "payload_paths": (str(payload_path),),
            "payload_sha256": payload_sha256,
            "schema_version": "1.0",
            "created_git_commit": current_code_revision().commit,
            "created_environment_sha256": environment_snapshot().fingerprint_sha256,
            "state": ArtifactState.COMPLETED,
            "completion_required": True,
            "completion_manifest_sha256": completion.completion_manifest_sha256,
        }
    )
    store.write_completed(manifest, completion)
    return manifest


def _payload_path(layout: WorkspaceLayout, fingerprint: str) -> Path:
    return (
        experiment_workspace(layout, _EXPERIMENT)
        / "artifacts"
        / "derived"
        / f"primitive-validation.{fingerprint[:16]}.json"
    )


def _validation_payload(block_pattern: tuple[int, ...]) -> StableJsonPayload:
    source_seed = RandomSeed(active_config().scientific.randomness.pilot_seeds[0])
    instance = generate_exact_separator_instance(
        ExactSeparatorInstanceRequest(block_pattern, source_seed)
    )
    if instance.lower_response_matrix.shape != instance.upper_response_matrix.shape:
        raise PrimitiveValidationError("response bounds have different shapes")
    if not np.all(instance.lower_response_matrix <= instance.upper_response_matrix):
        raise PrimitiveValidationError("response lower bounds exceed upper bounds")
    assignment = solve_minimum_cost_assignment(
        instance.lower_response_matrix,
        active_config().solvers.exact_sparse.lap_objective_tie_tolerance,
    )
    if len(set(assignment.column_for_row)) != len(assignment.column_for_row):
        raise PrimitiveValidationError("assignment is not bijective")
    if not np.isfinite(assignment.objective_value):
        raise PrimitiveValidationError("assignment objective is not finite")
    return cast(
        StableJsonPayload,
        OrderedDict(
            block_pattern=list(block_pattern),
            seed=source_seed.value,
            response_shape=list(instance.lower_response_matrix.shape),
            lower_bound_not_above_upper_bound=True,
            assignment_is_bijective=True,
            assignment_objective=assignment.objective_value,
        ),
    )


def _completion(
    coordinates: str,
    fingerprint: str,
    payload_path: Path,
    payload_sha256: str,
    configuration_sha256: str,
    code_sha256: str,
    runtime_sha256: str,
) -> CompletionManifest:
    completion = CompletionManifest.model_validate(
        {
            "schema_version": "1.0",
            "semantic_experiment_coordinates": coordinates,
            "producer_stage": _STAGE,
            "terminal_state": TerminalState.COMPLETED,
            "dependency_fingerprint_sha256": fingerprint,
            "upstream_artifact_ids": (),
            "mandatory_artifact_paths": (str(payload_path),),
            "mandatory_artifact_sha256": payload_sha256,
            "scientific_configuration_sha256": configuration_sha256,
            "relevant_code_sha256": code_sha256,
            "material_runtime_sha256": runtime_sha256,
            "upstream_lineage": stable_json(cast(StableJsonPayload, OrderedDict())),
            "completion_validation_state": "validated",
            "completion_written_last": True,
            "completion_manifest_sha256": "",
        }
    )
    return completion.model_copy(
        update={"completion_manifest_sha256": completion_manifest_self_hash(completion)}
    )
