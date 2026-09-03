from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from fedorbit.analysis.records import (
    MetricDirection,
    MetricRecord,
    MetricRecordCollection,
    validate_metric_records,
)
from fedorbit.config.loading import active_config, repository_root
from fedorbit.datasets.common import (
    DatasetInspectionRequest,
    DatasetObservation,
    DatasetObservationPersistenceRequest,
    inspect_dataset,
    persist_dataset_observation,
)
from fedorbit.experiments.catalogue import ExperimentDefinition
from fedorbit.experiments.cells import experiment_relevance
from fedorbit.experiments.synthetic import (
    ExactSeparatorInstanceRequest,
    ScalabilityInstanceRequest,
    UnresolvedMapWorldKind,
    UnresolvedMapWorldRequest,
    generate_exact_separator_instance,
    generate_scalability_instance,
    generate_unresolved_map_world,
)
from fedorbit.infrastructure.environment import environment_snapshot
from fedorbit.infrastructure.manifests import (
    CompletionManifest,
    ReusableArtifactManifest,
    artifact_id,
    completion_manifest_self_hash,
    file_sha256,
)
from fedorbit.infrastructure.provenance import (
    configuration_subset_digest,
    implementation_fingerprint,
    runtime_fingerprint,
    stage_dependency_fingerprint,
)
from fedorbit.infrastructure.reuse import (
    CellDecision,
    ExecutionAction,
    ExecutionReuse,
    validate_completed_artifact,
    validate_reusable_artifact,
)
from fedorbit.infrastructure.runtime import (
    ExecutionLogEvent,
    ExecutionLogger,
    RandomSeed,
    current_code_revision,
    execution_logger,
)
from fedorbit.infrastructure.workspace import (
    RawDuplicateReportRequest,
    RawInventoryPersistenceRequest,
    RawInventoryRequest,
    WorkspaceLayout,
    build_layout,
    experiment_workspace,
    inspect_raw_inventory,
    persist_raw_duplicate_report,
    persist_raw_inventory,
)
from fedorbit.learning.scoring import LocalClassCount, ScoringRequest, score_model
from fedorbit.methods.assimilation import capture_pre_confirm_pair
from fedorbit.optimization.assignment import solve_minimum_cost_assignment
from fedorbit.optimization.correspondence import build_padded_block_structure
from fedorbit.optimization.dense_ccp import solve_dense_ccp
from fedorbit.optimization.exact_sparse import fixed_action_worst_correspondence
from fedorbit.optimization.objective import (
    build_robust_action_problem,
    curriculum_action_from_entries,
)
from fedorbit.response.packet import build_source_packet
from fedorbit.response.uncertainty import FinalResponseEntry, FinalResponseEstimate
from fedorbit.types import (
    ArtifactFingerprint,
    ArtifactIdentifier,
    ArtifactPath,
    ArtifactStage,
    ArtifactState,
    CoarseGroup,
    DatasetId,
    ExecutionCell,
    ExperimentName,
    ExperimentSeed,
    MetricId,
    OverwritePolicy,
    ScalabilityBlockPattern,
    SemanticCell,
    SemanticCoordinates,
    StableJsonPayload,
    TerminalState,
    TransferMethod,
    stable_json,
)


class StorageError(ValueError):
    pass


def atomic_write_json(path: Path, payload: StableJsonPayload) -> None:
    atomic_write_bytes(path, (stable_json(payload) + "\n").encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._manifests = root / "manifests"
        self._completions = root / "completions"
        self._staging = root / "staging"

    @property
    def root(self) -> Path:
        return self._root

    def manifest_path(self, artifact_id: ArtifactIdentifier) -> Path:
        return self._manifests / f"{artifact_id.value}.json"

    def manifest_dir(self) -> Path:
        return self._manifests

    def completion_path(self, artifact_id: ArtifactIdentifier) -> Path:
        return self._completions / f"{artifact_id.value}.json"

    def staging_dir(self) -> Path:
        return self._staging

    def write_reusable(self, manifest: ReusableArtifactManifest) -> None:
        atomic_write_json(
            self.manifest_path(ArtifactIdentifier(manifest.artifact_id)),
            manifest.model_dump(mode="json"),
        )

    def write_completed(
        self,
        manifest: ReusableArtifactManifest,
        completion: CompletionManifest,
    ) -> None:
        if not manifest.completion_required:
            raise StorageError("completed artifacts must require a completion record")
        try:
            validate_completed_artifact(manifest, completion)
        except ValueError as error:
            raise StorageError(str(error)) from error
        self.write_reusable(manifest)
        atomic_write_json(
            self.completion_path(ArtifactIdentifier(manifest.artifact_id)),
            completion.model_dump(mode="json"),
        )

    def read_reusable(self, artifact_id: ArtifactIdentifier) -> ReusableArtifactManifest:
        path = self.manifest_path(artifact_id)
        if not path.is_file():
            raise StorageError(f"no artifact manifest for {artifact_id.value}")
        return ReusableArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def read_completion(self, artifact_id: ArtifactIdentifier) -> CompletionManifest:
        path = self.completion_path(artifact_id)
        if not path.is_file():
            raise StorageError(f"no completion manifest for {artifact_id.value}")
        return CompletionManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def resolve(self, artifact_id: ArtifactIdentifier) -> ReusableArtifactManifest:
        manifest = self.read_reusable(artifact_id)
        validate_reusable_artifact(manifest)
        if manifest.completion_required:
            try:
                validate_completed_artifact(manifest, self.read_completion(artifact_id))
            except ValueError as error:
                raise StorageError(str(error)) from error
        return manifest

    def find_by_fingerprint(
        self, fingerprint_sha256: ArtifactFingerprint
    ) -> ReusableArtifactManifest | None:
        if not self._manifests.is_dir():
            return None
        for path in sorted(self._manifests.glob("*.json")):
            manifest = ReusableArtifactManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if manifest.dependency_fingerprint_sha256 != fingerprint_sha256.value:
                continue
            try:
                self.resolve(ArtifactIdentifier(manifest.artifact_id))
            except ValueError:
                return None
            return manifest
        return None

    def remove_manifest(self, artifact_id: ArtifactIdentifier) -> None:
        self.manifest_path(artifact_id).unlink(missing_ok=True)
        self.completion_path(artifact_id).unlink(missing_ok=True)

    def all_manifests(self) -> tuple[ReusableArtifactManifest, ...]:
        if not self._manifests.is_dir():
            return ()
        return tuple(
            ReusableArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self._manifests.glob("*.json"))
        )


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    valid_artifact_ids: tuple[ArtifactIdentifier, ...]
    next_resume_coordinates: SemanticCoordinates | None
    stochastic_boundary_ok: bool


class RecoveryBoundary:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def discard_interrupted_staging(self) -> None:
        staging = self._store.staging_dir()
        if staging.is_dir():
            shutil.rmtree(staging)

    def valid_artifact_ids(self) -> tuple[ArtifactIdentifier, ...]:
        valid: list[ArtifactIdentifier] = []
        for manifest in self._store.all_manifests():
            try:
                resolved = self._store.resolve(ArtifactIdentifier(manifest.artifact_id))
            except ValueError:
                continue
            if resolved.state == ArtifactState.COMPLETED:
                valid.append(ArtifactIdentifier(resolved.artifact_id))
        return tuple(sorted(valid, key=lambda identifier: identifier.value))

    def next_resume(self, ordered_cells: tuple[ExecutionCell, ...]) -> RecoveryRecord:
        valid = frozenset(self.valid_artifact_ids())
        resume = next(
            (cell.coordinates for cell in ordered_cells if cell.artifact_identifier not in valid),
            None,
        )
        return RecoveryRecord(
            valid_artifact_ids=tuple(sorted(valid, key=lambda identifier: identifier.value)),
            next_resume_coordinates=resume,
            stochastic_boundary_ok=resume is not None,
        )


class ExecutionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetPreparationRequest:
    datasets: tuple[DatasetId, ...]
    overwrite_policy: OverwritePolicy


@dataclass(frozen=True, slots=True)
class ExperimentExecutionRequest:
    experiment: ExperimentName
    definition: ExperimentDefinition
    overwrite_policy: OverwritePolicy


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    decision: CellDecision
    manifest: ReusableArtifactManifest | None


@dataclass(frozen=True, slots=True)
class DatasetPreparationResult:
    observations: tuple[DatasetObservation, ...]
    validation_artifact_paths: tuple[ArtifactPath, ...]
    duplicate_artifact_paths: tuple[ArtifactPath, ...]

    @property
    def blocked_datasets(self) -> tuple[DatasetId, ...]:
        return tuple(
            observation.dataset
            for observation in self.observations
            if not observation.valid_for_chronological_preprocessing
        )


class ExecutionExecutor:
    def __init__(self, store: ArtifactStore, logger: ExecutionLogger | None = None) -> None:
        self._store = store
        self._logger = logger if logger is not None else execution_logger()

    def execute(
        self,
        decisions: tuple[CellDecision, ...],
        producer: Callable[[CellDecision], ReusableArtifactManifest],
    ) -> tuple[ExecutionResult, ...]:
        results: list[ExecutionResult] = []
        for decision in decisions:
            if decision.action == ExecutionAction.REUSE:
                if decision.manifest is None:
                    raise ExecutionError("reuse decision has no manifest")
                manifest = self._store.resolve(ArtifactIdentifier(decision.manifest.artifact_id))
                results.append(ExecutionResult(decision, manifest))
                self._logger.record(
                    ExecutionLogEvent(
                        occurred_at=datetime.now(UTC),
                        cell_coordinates=decision.cell_coordinates,
                        artifact_id=ArtifactIdentifier(manifest.artifact_id),
                        state=ArtifactState.COMPLETED,
                    )
                )
                continue
            manifest = producer(decision)
            self._store.write_reusable(manifest)
            validated = self._store.resolve(ArtifactIdentifier(manifest.artifact_id))
            results.append(ExecutionResult(decision, validated))
            self._logger.record(
                ExecutionLogEvent(
                    occurred_at=datetime.now(UTC),
                    cell_coordinates=decision.cell_coordinates,
                    artifact_id=ArtifactIdentifier(validated.artifact_id),
                    state=ArtifactState.COMPLETED,
                )
            )
        return tuple(results)


def execution_store() -> ArtifactStore:
    return ArtifactStore(build_layout().execution_root)


def _recover(store: ArtifactStore, cells: tuple[ExecutionCell, ...]) -> None:
    recovery = RecoveryBoundary(store)
    recovery.discard_interrupted_staging()
    recovery.next_resume(cells)


def preprocess_datasets(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    raw_root = repository_root() / "data" / "raw"
    inventories = tuple(
        inspect_raw_inventory(RawInventoryRequest(dataset, raw_root))
        for dataset in request.datasets
    )
    if len(inventories) != len(request.datasets):
        raise ExecutionError("raw inventory collection did not cover every requested dataset")
    store = execution_store()
    persisted_inventory_paths = tuple(
        persist_raw_inventory(
            RawInventoryPersistenceRequest(inventory, store.root / "preprocessing")
        )
        for inventory in inventories
    )
    if len(persisted_inventory_paths) != len(inventories):
        raise ExecutionError("raw inventory persistence did not cover every requested dataset")
    observations = tuple(
        inspect_dataset(DatasetInspectionRequest(dataset, raw_root)) for dataset in request.datasets
    )
    if len(observations) != len(request.datasets):
        raise ExecutionError("dataset observation collection did not cover every requested dataset")
    validation_paths = tuple(
        persist_dataset_observation(
            DatasetObservationPersistenceRequest(observation, store.root / "preprocessing")
        )
        for observation in observations
    )
    duplicate_paths = tuple(
        persist_raw_duplicate_report(
            RawDuplicateReportRequest(dataset, raw_root, store.root / "preprocessing")
        )
        for dataset in request.datasets
    )
    if len(duplicate_paths) != len(request.datasets):
        raise ExecutionError("duplicate diagnostics did not cover every requested dataset")
    return DatasetPreparationResult(
        observations=observations,
        validation_artifact_paths=tuple(ArtifactPath(path) for path in validation_paths),
        duplicate_artifact_paths=tuple(ArtifactPath(path) for path in duplicate_paths),
    )


def run_smoke_validation(overwrite_policy: OverwritePolicy) -> None:
    del overwrite_policy
    seed = RandomSeed(active_config().scientific.randomness.pilot_seeds[0])
    exact = generate_exact_separator_instance(ExactSeparatorInstanceRequest((2, 2), seed))
    if exact.lower_response_matrix.shape != (4, 4):
        raise ExecutionError("synthetic exactness smoke instance has an invalid matrix shape")
    mechanism = generate_unresolved_map_world(
        UnresolvedMapWorldRequest(UnresolvedMapWorldKind.COMMON_ACTION, seed)
    )
    if mechanism.lower_response_matrix.shape != (4, 4):
        raise ExecutionError("synthetic mechanism smoke instance has an invalid matrix shape")
    scalability = generate_scalability_instance(
        ScalabilityInstanceRequest(4, ScalabilityBlockPattern.BALANCED, 1, seed)
    )
    if scalability.fixed_action.shape != (4,):
        raise ExecutionError("synthetic scalability smoke instance has an invalid action shape")
    estimate = FinalResponseEstimate(
        entries=(FinalResponseEntry(0, 0, 1.0, 0.0, 1.0, 1.0, True),),
        critical_value=1.0,
        useful_intervention_columns=1,
        median_band_width_ratio=0.0,
        stability_rule_passed=True,
    )
    packet = build_source_packet(
        estimate,
        anonymous_fine_node_ids=("node-0001",),
        exposed_coarse_group_id="smoke",
        per_node_train_support=(1,),
        per_node_meta_support=(1,),
        per_node_effective_replicate_count=(1,),
        source_checkpoint_sha256="0" * 64,
        response_configuration_sha256="1" * 64,
        creation_timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    packet.validate()
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=active_config().scientific.base_model_pilot.learning_rates[0]
    )
    snapshots = capture_pre_confirm_pair(model, optimizer)
    if len(snapshots.baseline.model_state.tensors) != len(snapshots.curriculum.model_state.tensors):
        raise ExecutionError("pre-confirm snapshots have inconsistent parameter counts")


def run_experiment(request: ExperimentExecutionRequest) -> None:
    store = execution_store()
    layout = build_layout()
    if request.experiment == ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION:
        execute_primitive_validation(store, layout)
        return
    if request.experiment in _SYNTHETIC_EXPERIMENTS:
        execute_synthetic_experiment(store, layout, request)
        return
    blocked = _chronology_block_reasons()
    if blocked:
        _persist_blocked_experiment(layout, request, blocked)
        return
    reuse = ExecutionReuse(store)
    cells = tuple(
        ExecutionCell(
            SemanticCoordinates(f"{request.experiment.value}:{seed}"),
            ArtifactIdentifier(f"cell-{request.experiment.value}-{seed}"),
            ArtifactFingerprint(f"cell-{request.experiment.value}-{seed}"),
        )
        for seed in request.definition.seeds
    )
    _recover(store, cells)
    decisions = reuse.decide(cells, request.overwrite_policy)
    reuse.validate_existing(decisions)
    if any(
        decision.action in (ExecutionAction.EXECUTE, ExecutionAction.OVERWRITE)
        for decision in decisions
    ):
        raise ExecutionError("registered experiment has no scientific producer")


_SYNTHETIC_EXPERIMENTS = frozenset(
    {
        ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION,
        ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION,
        ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION,
        ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK,
        ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION,
        ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP,
        ExperimentName.ROBUST_COMPROMISE_UNDER_UNIDENTIFIED_MAP,
        ExperimentName.MAP_DEPENDENT_ACTION_BOUNDARY,
        ExperimentName.EXACT_MAP_VALUE_BOUND_VALIDATION,
        ExperimentName.SPARSITY_AND_DENSE_FALLBACK,
        ExperimentName.SCALABILITY_AND_EFFICIENCY,
    }
)


def _chronology_block_reasons() -> OrderedDict[str, str]:
    raw_root = repository_root() / "data" / "raw"
    observations = tuple(
        inspect_dataset(DatasetInspectionRequest(dataset, raw_root)) for dataset in DatasetId
    )
    return OrderedDict(
        (observation.dataset.value, observation.event_time.reason)
        for observation in observations
        if not observation.valid_for_chronological_preprocessing
    )


def _persist_blocked_experiment(
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    reasons: OrderedDict[str, str],
) -> None:
    destination = experiment_workspace(layout, request.experiment) / "artifacts" / "derived"
    payload = cast(
        StableJsonPayload,
        OrderedDict(
            experiment=request.experiment.value,
            state=ArtifactState.BLOCKED.value,
            reason="chronological preprocessing prerequisite is unsatisfied",
            blocked_datasets=reasons,
        ),
    )
    atomic_write_json(destination / "blocked.json", payload)


def execute_synthetic_experiment(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> ReusableArtifactManifest:
    pattern = active_config().generators.exact_separator_theorem.block_patterns[3]
    seed = ExperimentSeed(request.definition.seeds[0])
    cell = SemanticCell(experiment=request.experiment, seed=seed)
    relevance = experiment_relevance(request.experiment)
    coordinates = cell.identity_json(relevance)
    fingerprint = stage_dependency_fingerprint(
        ArtifactStage.EVALUATION,
        cell,
        relevance,
        (),
        _CONFIGURATION_SECTIONS,
        _PRODUCER_MODULE,
    )
    payload = _synthetic_experiment_payload(
        request.experiment, pattern, RandomSeed(seed.value), fingerprint
    )
    payload_path = (
        experiment_workspace(layout, request.experiment)
        / "artifacts"
        / "derived"
        / f"synthetic-validation.{fingerprint[:16]}.json"
    )
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = configuration_subset_digest(_CONFIGURATION_SECTIONS)
    code_sha256 = implementation_fingerprint(_PRODUCER_MODULE)
    runtime_sha256 = runtime_fingerprint(ArtifactStage.EVALUATION).sha256
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
        OrderedDict(
            artifact_id=artifact_id("other", payload, fingerprint),
            artifact_type="other",
            semantic_producer_coordinates=coordinates,
            producer_stage=ArtifactStage.EVALUATION,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=(),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            material_runtime_sha256=runtime_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_sha256,
            schema_version="1.0",
            created_git_commit=current_code_revision().commit,
            created_environment_sha256=environment_snapshot().fingerprint_sha256,
            state=ArtifactState.COMPLETED,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
    )
    store.write_completed(manifest, completion)
    return manifest


def _synthetic_experiment_payload(
    experiment: ExperimentName,
    pattern: tuple[int, ...],
    seed: RandomSeed,
    fingerprint: str,
) -> StableJsonPayload:
    instance = generate_exact_separator_instance(ExactSeparatorInstanceRequest(pattern, seed))
    groups = tuple(CoarseGroup)[: len(pattern)]
    counts = OrderedDict((group, size) for group, size in zip(groups, pattern, strict=True))
    blocks = build_padded_block_structure(groups, counts, counts)
    problem = build_robust_action_problem(
        blocks,
        instance.lower_response_matrix,
        instance.upper_response_matrix,
        instance.target_importance / instance.target_importance.sum(),
        tuple(range(sum(pattern))),
    )
    action = curriculum_action_from_entries(
        problem, ((0, min(problem.total_budget, problem.coordinate_caps[0])),)
    )
    outcome = fixed_action_worst_correspondence(
        problem,
        action,
        active_config().solvers.exact_sparse.lap_objective_tie_tolerance,
        active_config().solvers.exact_sparse.action_tie_tolerance,
    )
    metric = MetricRecord(
        experiment=experiment,
        pair="synthetic",
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        condition="generated",
        seed=seed.value,
        metric_name=MetricId.ACTIVE_IMAGE_CANDIDATES,
        metric_value=float(outcome.active_image_candidates),
        metric_unit="count",
        direction=MetricDirection.DESCRIPTIVE,
        evaluation_class_set_sha256=hashlib.sha256(b"synthetic-correspondence").hexdigest(),
        input_artifact_ids=("synthetic-generator",),
        dependency_fingerprint_sha256=fingerprint,
        valid=True,
        invalid_reason=None,
    )
    validate_metric_records(MetricRecordCollection((metric,)))
    dense = None
    if experiment == ExperimentName.SPARSITY_AND_DENSE_FALLBACK:
        dense = solve_dense_ccp(problem, seed.value, experiment.value)
    return cast(
        StableJsonPayload,
        OrderedDict(
            experiment=experiment.value,
            seed=seed.value,
            block_pattern=list(pattern),
            response_shape=list(instance.lower_response_matrix.shape),
            separator_objective=outcome.separator_objective,
            active_image_candidates=outcome.active_image_candidates,
            lap_calls=outcome.lap_calls,
            worst_correspondence_images=list(outcome.worst_correspondence.images),
            metric_record=metric.model_dump(mode="json"),
            dense_ccp=(
                None
                if dense is None
                else OrderedDict(
                    master_objective=dense.master_objective,
                    projected_objective=dense.best_projected_response_objective,
                    bound_gap=dense.dense_bound_gap,
                    integrality_residual=dense.integrality_residual,
                    exact=dense.is_exact,
                )
            ),
        ),
    )


class PrimitiveValidationError(ValueError):
    pass


_EXPERIMENT = ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION
_STAGE = ArtifactStage.EVALUATION
_CONFIGURATION_SECTIONS = frozenset({"action", "generators", "metrics"})
_PRODUCER_MODULE = "fedorbit.infrastructure.execution"


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
        OrderedDict(
            artifact_id=artifact_id("other", payload, fingerprint),
            artifact_type="other",
            semantic_producer_coordinates=coordinates,
            producer_stage=_STAGE,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=(),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            material_runtime_sha256=runtime_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_sha256,
            schema_version="1.0",
            created_git_commit=current_code_revision().commit,
            created_environment_sha256=environment_snapshot().fingerprint_sha256,
            state=ArtifactState.COMPLETED,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
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
    scoring = _score_deterministic_validation_batch()
    return cast(
        StableJsonPayload,
        OrderedDict(
            block_pattern=list(block_pattern),
            seed=source_seed.value,
            response_shape=list(instance.lower_response_matrix.shape),
            lower_bound_not_above_upper_bound=True,
            assignment_is_bijective=True,
            assignment_objective=assignment.objective_value,
            scoring_row_count=len(scoring.rows),
            scoring_macro_cross_entropy=scoring.macro_cross_entropy.value,
        ),
    )


def _score_deterministic_validation_batch():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(((1.0, -1.0), (-1.0, 1.0))))
    return score_model(
        ScoringRequest(
            model=model,
            features=torch.tensor(((1.0, 0.0), (0.0, 1.0))),
            targets=torch.tensor((0, 1)),
            local_class_count=LocalClassCount(2),
        )
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
        OrderedDict(
            schema_version="1.0",
            semantic_experiment_coordinates=coordinates,
            producer_stage=_STAGE,
            terminal_state=TerminalState.COMPLETED,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=(),
            mandatory_artifact_paths=(str(payload_path),),
            mandatory_artifact_sha256=payload_sha256,
            scientific_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            material_runtime_sha256=runtime_sha256,
            upstream_lineage=stable_json(cast(StableJsonPayload, OrderedDict())),
            completion_validation_state="validated",
            completion_written_last=True,
            completion_manifest_sha256="",
        )
    )
    return completion.model_copy(
        update=OrderedDict(completion_manifest_sha256=completion_manifest_self_hash(completion))
    )
