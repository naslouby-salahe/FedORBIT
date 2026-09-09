from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from fedorbit.analysis.metrics import (
    ClassF1,
    ClassF1Set,
    ClassRecall,
    ClassRecallSet,
    EfficiencyRecord,
    balanced_accuracy,
    confusion_counts,
    f1_from_counts,
    macro_f1,
    recall_from_counts,
)
from fedorbit.analysis.records import (
    ComparisonDecision,
    MetricDirection,
    MetricRecord,
    MetricRecordCollection,
    PairedComparisonRecord,
    validate_metric_records,
)
from fedorbit.analysis.statistics import (
    NamedPValue,
    PValueSet,
    exact_sign_flip_test,
    holm_step_down,
    paired_bca_interval,
    statistical_bootstrap_seed,
)
from fedorbit.config.loading import active_config, raw_dataset_root
from fedorbit.datasets.common import (
    DatasetInspectionRequest,
    DatasetObservation,
    DatasetObservationPersistenceRequest,
    file_sha256,
    inspect_dataset,
    persist_dataset_observation,
)
from fedorbit.datasets.materialization import (
    MaterializationError,
    MaterializationResourceLimitError,
    MaterializedClient,
    SplitTensors,
    TransferConceptGroup,
    materialize_client,
    subsampled_materialized_client,
    transfer_concept_groups,
)
from fedorbit.datasets.ontology import TRANSFER_ONTOLOGY
from fedorbit.experiments.catalogue import ExperimentDefinition
from fedorbit.experiments.cells import experiment_relevance
from fedorbit.experiments.synthetic import (
    CouplingGenerationError,
    CouplingInstance,
    CouplingInstanceRequest,
    ExactSeparatorInstanceRequest,
    MechanismGenerationError,
    ScalabilityGenerationError,
    ScalabilityInstanceRequest,
    UnresolvedMapWorld,
    UnresolvedMapWorldKind,
    UnresolvedMapWorldRequest,
    eligible_coupling_support_sizes,
    generate_coupling_instance,
    generate_exact_separator_instance,
    generate_scalability_instance,
    generate_unresolved_map_world,
)
from fedorbit.infrastructure.environment import environment_snapshot
from fedorbit.infrastructure.failures import (
    InfrastructureFailureError,
    RetryPolicy,
    classify_failure,
)
from fedorbit.infrastructure.manifests import (
    CompletionManifest,
    DatasetManifest,
    FeatureQualityManifest,
    ReusableArtifactManifest,
    artifact_id,
    completion_manifest_self_hash,
)
from fedorbit.infrastructure.provenance import (
    configuration_subset_digest,
    implementation_fingerprint,
    runtime_fingerprint,
    stage_dependency_fingerprint,
)
from fedorbit.infrastructure.reuse import (
    ExecutionAction,
    ExecutionReuse,
    validate_completed_artifact,
    validate_reusable_artifact,
)
from fedorbit.infrastructure.runtime import (
    EfficiencyMeasurement,
    ExecutionLogEvent,
    ExecutionLogger,
    RandomSeed,
    SeedDerivationRequest,
    current_code_revision,
    derive_seed32,
    execution_logger,
    measure_efficiency,
    principal_determinism,
)
from fedorbit.infrastructure.storage import StorageError, atomic_write_bytes, atomic_write_json
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
from fedorbit.interface import (
    StrictResourceViolationError,
    validate_disjoint_feature_namespaces,
    validate_oracle_acl_isolation,
)
from fedorbit.learning.checkpoints import load_base_checkpoint, save_base_checkpoint
from fedorbit.learning.pilot import (
    PilotData,
    create_classifier,
    run_base_model_pilot,
    select_pilot_configuration,
)
from fedorbit.learning.scoring import LocalClassCount, ScoreArtifact, ScoringRequest, score_model
from fedorbit.learning.training import (
    BaseCheckpoint,
    ClassWeights,
    make_adamw,
    train_base_model,
)
from fedorbit.methods.assimilation import (
    AssimilationCoordinates,
    ConfirmationRequest,
    ConfirmationVerdict,
    PreTestLifecycle,
    PreTestPhase,
    apply_accepted_assimilation,
    capture_pre_confirm_pair,
    run_proposal_confirmation,
    settle_rejected_proposal,
)
from fedorbit.methods.baselines import (
    coarse_block_mean_matrix,
    coarse_block_min_matrix,
    coupling_destroyed_matrices,
    local_sir_action,
    optimize_against_fixed_matrix,
    orbit_mean_matrix,
)
from fedorbit.methods.map_availability_audit import (
    MapAvailabilityAuditSubmission,
    blank_audit_template,
    distinct_researcher_ids,
    documented_public_labels,
    submission_sha256,
    validate_submission,
)
from fedorbit.methods.target import (
    CurriculumMultipliers,
    SourceProposal,
    TargetImportanceError,
    TransferNodeRisk,
    build_target_importance,
    rank_source_proposals,
    select_source_sequentially,
)
from fedorbit.optimization.assignment import solve_minimum_cost_assignment
from fedorbit.optimization.certificates import (
    RectangularHull,
    build_rectangular_hull,
    rectangular_value_over_candidates,
    robust_coupling_gap,
    verify_correspondence_certificate,
    verify_exactness_certificate,
)
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    PaddedBlockStructure,
    ResponseMatrix,
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.optimization.dense_ccp import solve_dense_ccp
from fedorbit.optimization.diagnostics import (
    analytic_rectangular_hull_bounds,
    fixed_action_rectangularization_gap,
)
from fedorbit.optimization.exact_qap import (
    fixed_action_worst_correspondence_qap,
    point_correspondence_commitment,
    solve_robust_action_qap,
)
from fedorbit.optimization.exact_sparse import (
    fixed_action_worst_correspondence,
    solve_robust_action,
)
from fedorbit.optimization.objective import (
    CurriculumAction,
    RobustActionProblem,
    build_robust_action_problem,
    evaluate_objective,
    zero_action,
)
from fedorbit.response.packet import (
    PacketConstructionContext,
    SourcePacket,
    build_source_packet,
    construct_source_packet,
)
from fedorbit.response.pilot import (
    PilotCheckpoint,
    ResponseCandidate,
    run_pooled_source_response_pilot,
    select_response_configuration,
)
from fedorbit.response.pilot import PilotData as ResponsePilotData
from fedorbit.response.uncertainty import FinalResponseEntry, FinalResponseEstimate
from fedorbit.types import (
    AnonymousNodeDisplayId,
    ArtifactFingerprint,
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactPath,
    ArtifactSchemaVersion,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    ArtifactTypeName,
    BootstrapPurpose,
    CheckpointDirectorySegment,
    ClassCount,
    ClassIndex,
    ClientRole,
    CoarseGroup,
    Coefficient,
    ConceptCount,
    ConfigurationSection,
    ContrastCoordinates,
    ContrastName,
    CouplingCompatibility,
    DatasetId,
    DatasetPreprocessingState,
    DirectedPair,
    DirectedPairName,
    EvaluationConditionName,
    ExecutionCell,
    ExecutionStageName,
    ExperimentCondition,
    ExperimentLocalMethod,
    ExperimentName,
    ExperimentSeed,
    ExposedCoarseGroupId,
    Index,
    InfrastructureLogCoordinate,
    InvalidReason,
    MethodName,
    MetricId,
    MetricUnit,
    MultiplicityFamily,
    OverwritePolicy,
    ProducerModuleName,
    PValueName,
    ReplicateCount,
    ResourceLimitReason,
    ReuseDecision,
    Rfc3339UtcTimestamp,
    RngNamespace,
    SampleCount,
    ScalabilityBlockPattern,
    Score,
    SemanticCell,
    SemanticCoordinate,
    SemanticCoordinates,
    SemanticCoordinateText,
    SerializedPacket,
    Sha256Digest,
    SourceClientName,
    Split,
    StableJsonPayload,
    StepCount,
    StorageLayoutSegment,
    SupportCount,
    SupportSize,
    TerminalState,
    Threshold,
    Tolerance,
    TransferMethod,
    ValidationReason,
    stable_json,
)

_MODULE_NAME = ProducerModuleName("fedorbit.infrastructure.execution")
_PRINCIPAL_CONDITION = EvaluationConditionName("principal")


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._manifests = root / StorageLayoutSegment.MANIFESTS
        self._completions = root / StorageLayoutSegment.COMPLETIONS
        self._staging = root / StorageLayoutSegment.STAGING

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
            self.manifest_path(manifest.artifact_id),
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
            self.completion_path(manifest.artifact_id),
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
        for path in sorted(self._manifests.glob(StorageLayoutSegment.MANIFEST_GLOB)):
            manifest = ReusableArtifactManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if manifest.dependency_fingerprint_sha256 != fingerprint_sha256.value:
                continue
            try:
                self.resolve(manifest.artifact_id)
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
            for path in sorted(self._manifests.glob(StorageLayoutSegment.MANIFEST_GLOB))
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
                resolved = self._store.resolve(manifest.artifact_id)
            except ValueError:
                continue
            if resolved.state == ArtifactState.COMPLETED:
                valid.append(resolved.artifact_id)
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
class DatasetPreparationResult:
    observations: tuple[DatasetObservation, ...]
    validation_artifact_paths: tuple[ArtifactPath, ...]
    duplicate_artifact_paths: tuple[ArtifactPath, ...]
    resource_blocked_datasets: tuple[tuple[DatasetId, ResourceLimitReason], ...] = ()

    @property
    def blocked_datasets(self) -> tuple[DatasetId, ...]:
        return tuple(
            observation.dataset
            for observation in self.observations
            if not observation.valid_for_chronological_preprocessing
        )


def execution_store() -> ArtifactStore:
    return ArtifactStore(build_layout().execution_root)


def _recover(store: ArtifactStore, cells: tuple[ExecutionCell, ...]) -> None:
    recovery = RecoveryBoundary(store)
    recovery.discard_interrupted_staging()
    recovery.next_resume(cells)


def preprocess_datasets(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    raw_root = raw_dataset_root()
    inventories = tuple(
        inspect_raw_inventory(RawInventoryRequest(dataset, raw_root))
        for dataset in request.datasets
    )
    if len(inventories) != len(request.datasets):
        raise ExecutionError("raw inventory collection did not cover every requested dataset")
    store = execution_store()
    persisted_inventory_paths = tuple(
        persist_raw_inventory(
            RawInventoryPersistenceRequest(
                inventory, store.root / StorageLayoutSegment.PREPROCESSING
            )
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
            DatasetObservationPersistenceRequest(
                observation, store.root / StorageLayoutSegment.PREPROCESSING
            )
        )
        for observation in observations
    )
    duplicate_paths = tuple(
        persist_raw_duplicate_report(
            RawDuplicateReportRequest(
                dataset, raw_root, store.root / StorageLayoutSegment.PREPROCESSING
            )
        )
        for dataset in request.datasets
    )
    if len(duplicate_paths) != len(request.datasets):
        raise ExecutionError("duplicate diagnostics did not cover every requested dataset")
    layout = build_layout()
    resource_blocked: list[tuple[DatasetId, ResourceLimitReason]] = []
    for observation in observations:
        if not observation.valid_for_chronological_preprocessing:
            continue
        try:
            materialized = materialize_client(observation.dataset, raw_root)
        except MaterializationResourceLimitError as error:
            resource_blocked.append((observation.dataset, ResourceLimitReason(str(error))))
            continue
        except MaterializationError as error:
            raise ExecutionError(
                f"could not materialize {observation.dataset.value}: {error}"
            ) from error
        persist_materialized_client(layout, materialized, request.overwrite_policy)
    return DatasetPreparationResult(
        observations=observations,
        validation_artifact_paths=tuple(ArtifactPath(path) for path in validation_paths),
        duplicate_artifact_paths=tuple(ArtifactPath(path) for path in duplicate_paths),
        resource_blocked_datasets=tuple(resource_blocked),
    )


def run_smoke_validation(
    overwrite_policy: OverwritePolicy,
) -> None:
    del overwrite_policy
    seed = active_config().scientific.randomness.pilot_seeds[0]
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
        anonymous_fine_node_ids=(AnonymousNodeDisplayId("node-0001"),),
        exposed_coarse_group_id=ExposedCoarseGroupId("smoke"),
        per_node_train_support=(1,),
        per_node_meta_support=(1,),
        per_node_effective_replicate_count=(1,),
        source_checkpoint_sha256=Sha256Digest("0" * 64),
        response_configuration_sha256=Sha256Digest("1" * 64),
        creation_timestamp=Rfc3339UtcTimestamp(
            datetime.now(UTC).isoformat().replace("+00:00", "Z")
        ),
    )
    packet.validate()
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=active_config().scientific.base_model_pilot.learning_rates[0]
    )
    snapshots = capture_pre_confirm_pair(model, optimizer)
    if len(snapshots.baseline.model_state.tensors) != len(snapshots.curriculum.model_state.tensors):
        raise ExecutionError("pre-confirm snapshots have inconsistent parameter counts")


def _execute_producer_with_retry(producer: Callable[[], ReusableArtifactManifest | None]) -> None:
    policy = RetryPolicy(
        active_config().runtime.failure_handling.retries_after_initial_infrastructure_failure
    )
    logger = execution_logger()
    attempt = 0
    while True:
        try:
            producer()
            return
        except InfrastructureFailureError as error:
            classification = classify_failure(error)
            decision = policy.decide(attempt, classification)
            logger.record(
                ExecutionLogEvent(
                    occurred_at=datetime.now(UTC),
                    cell_coordinates=SemanticCoordinates(InfrastructureLogCoordinate.RETRY),
                    artifact_id=None,
                    state=ArtifactState.RUNNING if decision.retry else ArtifactState.FAILED,
                    reuse_decision=ReuseDecision(
                        f"attempt {attempt + 1}: {type(error).__name__}: {error} -> "
                        f"{'retry' if decision.retry else 'exhausted'}"
                    ),
                )
            )
            if not decision.retry:
                raise ExecutionError(
                    f"infrastructure failure exhausted retries: {error}"
                ) from error
            attempt += 1


def _registered_experiment_producers(
    store: ArtifactStore, layout: WorkspaceLayout, request: ExperimentExecutionRequest
) -> Mapping[ExperimentName, Callable[[], ReusableArtifactManifest | None]]:
    producers: OrderedDict[ExperimentName, Callable[[], ReusableArtifactManifest | None]] = (
        OrderedDict()
    )
    producers[ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION] = lambda: (
        execute_primitive_validation(store, layout, request.overwrite_policy)
    )
    producers[ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION] = lambda: (
        execute_exact_sparse_theorem_exhaustive_validation(store, layout, request)
    )
    producers[ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION] = lambda: (
        execute_coupling_and_map_bound_validation(store, layout, request)
    )
    producers[ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION] = lambda: (
        execute_dataset_client_and_resource_validation(store, layout, request)
    )
    producers[ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT] = lambda: execute_base_model_pilot(
        store, layout, request
    )
    producers[ExperimentName.SOURCE_RESPONSE_ESTIMATOR_PILOT] = lambda: (
        execute_source_response_estimator_pilot(store, layout, request)
    )
    producers[ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION] = lambda: (
        execute_final_source_response_band_validation(store, layout, request)
    )
    producers[ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER] = lambda: (
        execute_primary_strict_cross_telemetry_transfer(store, layout, request)
    )
    producers[ExperimentName.STATISTICAL_SYNTHESIS] = lambda: execute_statistical_synthesis(
        store, layout, request
    )
    producers[ExperimentName.MECHANISM_ABLATIONS] = lambda: execute_mechanism_ablations(
        store, layout, request
    )
    producers[ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY] = lambda: (
        execute_target_confirmation_and_portability(store, layout, request)
    )
    producers[ExperimentName.SECONDARY_CROSS_MODALITY_GENERALIZATION] = lambda: (
        execute_secondary_cross_modality_generalization(store, layout, request)
    )
    producers[ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK] = lambda: (
        execute_exact_sparse_solver_benchmark(store, layout, request)
    )
    producers[ExperimentName.SCALABILITY_AND_EFFICIENCY] = lambda: (
        execute_scalability_and_efficiency(store, layout, request)
    )
    producers[ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP] = lambda: (
        execute_common_action_under_unidentified_map(store, layout, request)
    )
    producers[ExperimentName.ROBUST_COMPROMISE_UNDER_UNIDENTIFIED_MAP] = lambda: (
        execute_robust_compromise_under_unidentified_map(store, layout, request)
    )
    producers[ExperimentName.MAP_DEPENDENT_ACTION_BOUNDARY] = lambda: (
        execute_map_dependent_action_boundary(store, layout, request)
    )
    producers[ExperimentName.EXACT_MAP_VALUE_BOUND_VALIDATION] = lambda: (
        execute_exact_map_value_bound_validation(store, layout, request)
    )
    producers[ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION] = lambda: (
        execute_synthetic_coupling_mechanism_validation(store, layout, request)
    )
    producers[ExperimentName.SPARSITY_AND_DENSE_FALLBACK] = lambda: (
        execute_sparsity_and_dense_fallback(store, layout, request)
    )
    producers[ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION] = lambda: (
        execute_real_packet_coupling_mechanism_validation(store, layout, request)
    )
    producers[ExperimentName.MULTI_SOURCE_SELECTION_VALIDATION] = lambda: (
        execute_multi_source_selection_validation(store, layout, request)
    )
    producers[ExperimentName.SEMANTIC_SUFFICIENCY_FRONTIER] = lambda: (
        execute_semantic_sufficiency_frontier(store, layout, request)
    )
    producers[ExperimentName.WEAK_SIGNAL_SUPPORT_AND_HETEROGENEITY_BOUNDARIES] = lambda: (
        execute_weak_signal_support_and_heterogeneity_boundaries(store, layout, request)
    )
    producers[ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT] = lambda: (
        execute_map_availability_applicability_audit(store, layout, request)
    )
    producers[ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION] = lambda: (
        execute_baseline_and_oracle_correctness_validation(store, layout, request)
    )
    return producers


def run_experiment(request: ExperimentExecutionRequest) -> None:
    store = execution_store()
    layout = build_layout()
    producer = _registered_experiment_producers(store, layout, request).get(request.experiment)
    if producer is not None:
        _execute_producer_with_retry(producer)
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


def execute_dataset_client_and_resource_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> ReusableArtifactManifest:
    raw_root = raw_dataset_root()
    datasets: list[StableJsonPayload] = []
    materialized: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()
    for dataset in active_config().scientific.datasets.clients:
        try:
            client = materialize_client(dataset, raw_root)
        except MaterializationError as error:
            datasets.append(
                cast(
                    StableJsonPayload,
                    OrderedDict(
                        dataset=dataset.value,
                        state=ArtifactState.INVALID.value,
                        reason=str(error),
                    ),
                )
            )
            continue
        materialized[dataset] = client
        datasets.append(
            cast(
                StableJsonPayload,
                OrderedDict(
                    dataset=dataset.value,
                    state=ArtifactState.COMPLETED.value,
                    local_class_count=client.class_manifest.class_count,
                    feature_count=len(client.feature_names),
                    transfer_candidates=len(transfer_concept_groups(dataset, client)),
                ),
            )
        )
    pairs: list[StableJsonPayload] = []
    for pair in active_config().scientific.datasets.primary_directed_pairs:
        source = materialized.get(pair.source)
        target = materialized.get(pair.target)
        if source is None or target is None:
            pairs.append(
                cast(
                    StableJsonPayload,
                    OrderedDict(
                        source=pair.source.value,
                        target=pair.target.value,
                        state=ArtifactState.INVALID.value,
                    ),
                )
            )
            continue
        source_groups = {group.concept for group in transfer_concept_groups(pair.source, source)}
        target_groups = {group.concept for group in transfer_concept_groups(pair.target, target)}
        pairs.append(
            cast(
                StableJsonPayload,
                OrderedDict(
                    source=pair.source.value,
                    target=pair.target.value,
                    state=ArtifactState.COMPLETED.value,
                    shared_transfer_concept_count=len(source_groups & target_groups),
                ),
            )
        )
    seed = ExperimentSeed(active_config().scientific.randomness.confirmatory_seeds[0])
    return _persist_synthetic_experiment_payload(
        store,
        layout,
        request,
        seed,
        lambda fingerprint: cast(
            StableJsonPayload,
            OrderedDict(
                experiment=request.experiment.value,
                dependency_fingerprint_sha256=fingerprint,
                datasets=tuple(datasets),
                primary_pairs=tuple(pairs),
            ),
        ),
        frozenset(),
        _MODULE_NAME,
        "dataset-client-resource-validation",
    )


def _chronology_block_reasons() -> OrderedDict[DatasetId, ValidationReason]:
    raw_root = raw_dataset_root()
    primary_datasets = tuple(
        dataset
        for dataset, client in active_config().scientific.datasets.clients.items()
        if client.role == ClientRole.PRIMARY
    )
    observations = tuple(
        inspect_dataset(DatasetInspectionRequest(dataset, raw_root)) for dataset in primary_datasets
    )
    return OrderedDict(
        (observation.dataset, observation.event_time.reason)
        for observation in observations
        if not observation.valid_for_chronological_preprocessing
    )


def _persist_blocked_experiment(
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    reasons: OrderedDict[DatasetId, ValidationReason],
) -> None:
    destination = (
        experiment_workspace(layout, request.experiment)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
    )
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


def _persist_synthetic_experiment_payload(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    seed: ExperimentSeed,
    payload_builder: Callable[[Sha256Digest], StableJsonPayload],
    configuration_sections: frozenset[ConfigurationSection],
    producer_module: ProducerModuleName,
    artifact_name: str,
) -> ReusableArtifactManifest:
    cell = SemanticCell(experiment=request.experiment, seed=seed)
    relevance = experiment_relevance(request.experiment)
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.EVALUATION,
            cell,
            relevance,
            (),
            configuration_sections,
            producer_module,
        )
    )
    if request.overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    payload = payload_builder(fingerprint)
    payload_path = (
        experiment_workspace(layout, request.experiment)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
        / f"{artifact_name}.{fingerprint[:16]}.json"
    )
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(configuration_subset_digest(configuration_sections))
    code_sha256 = Sha256Digest(implementation_fingerprint(producer_module))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(ArtifactStage.EVALUATION).sha256)
    completion = _completion(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.OTHER.value), payload, Sha256Digest(fingerprint)
            ),
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


_THEOREM_VALIDATION_CONFIGURATION_SECTIONS = frozenset(
    {ConfigurationSection.ACTION, ConfigurationSection.GENERATORS, ConfigurationSection.SOLVERS}
)


def execute_exact_sparse_theorem_exhaustive_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> ReusableArtifactManifest:
    seed = ExperimentSeed(request.definition.seeds[0])
    return _persist_synthetic_experiment_payload(
        store,
        layout,
        request,
        seed,
        _theorem_exhaustive_validation_payload,
        _THEOREM_VALIDATION_CONFIGURATION_SECTIONS,
        _MODULE_NAME,
        "theorem-exhaustive-validation",
    )


def _theorem_exhaustive_validation_payload(fingerprint: Sha256Digest) -> StableJsonPayload:
    generator_config = active_config().generators.exact_separator_theorem
    solver_config = active_config().solvers.exact_sparse
    seeds = active_config().scientific.randomness.confirmatory_seeds
    instances_per_seed = generator_config.generated_instances_per_block_pattern_support_seed_cell
    cell_records: list[StableJsonPayload] = []
    for pattern in generator_config.block_patterns:
        total_nodes = sum(pattern)
        groups = tuple(CoarseGroup)[: len(pattern)]
        counts = OrderedDict(zip(groups, pattern, strict=True))
        blocks = build_padded_block_structure(groups, counts, counts)
        orbit = tuple(enumerate_block_permutations(blocks))
        for support in generator_config.supports:
            if support > total_nodes:
                continue
            cell_records.append(
                _theorem_exhaustive_validation_cell(
                    pattern,
                    support,
                    seeds,
                    instances_per_seed,
                    blocks,
                    orbit,
                    solver_config.lap_objective_tie_tolerance,
                    solver_config.action_tie_tolerance,
                    solver_config.exact_validation_absolute_tolerance,
                )
            )
    generated_instances = len(seeds) * instances_per_seed
    return cast(
        StableJsonPayload,
        OrderedDict(
            experiment=ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION.value,
            dependency_fingerprint_sha256=fingerprint,
            total_cells=len(cell_records),
            generated_instances_per_cell=generated_instances,
            total_instances=len(cell_records) * generated_instances,
            cells=cell_records,
        ),
    )


def _theorem_exhaustive_validation_cell(
    pattern: tuple[ConceptCount, ...],
    support: SupportCount,
    seeds: tuple[RandomSeed, ...],
    instances_per_seed: ReplicateCount,
    blocks: PaddedBlockStructure,
    orbit: tuple[BlockCorrespondence, ...],
    lap_objective_tie_tolerance: Tolerance,
    action_tie_tolerance: Tolerance,
    exact_validation_absolute_tolerance: Tolerance,
) -> StableJsonPayload:
    total_nodes = sum(pattern)
    max_absolute_objective_error = 0.0
    wrong_minima_count = 0
    invalid_certificate_count = 0
    for seed in seeds:
        for instance_index in range(instances_per_seed):
            instance = generate_exact_separator_instance(
                ExactSeparatorInstanceRequest(pattern, seed, support, instance_index)
            )
            problem = build_robust_action_problem(
                blocks,
                instance.lower_response_matrix,
                instance.upper_response_matrix,
                instance.target_importance / instance.target_importance.sum(),
                tuple(range(total_nodes)),
            )
            action = CurriculumAction(problem=problem, coordinates=instance.active_action)
            outcome = fixed_action_worst_correspondence(
                problem, action, lap_objective_tie_tolerance, action_tie_tolerance
            )
            exhaustive_truth = min(
                evaluate_objective(action, correspondence) for correspondence in orbit
            )
            error = abs(outcome.separator_objective - exhaustive_truth)
            max_absolute_objective_error = max(max_absolute_objective_error, error)
            if not verify_exactness_certificate(
                outcome.separator_objective, exhaustive_truth, exact_validation_absolute_tolerance
            ):
                wrong_minima_count += 1
            if not verify_correspondence_certificate(
                outcome.worst_correspondence,
                outcome.separator_objective,
                action,
                exact_validation_absolute_tolerance,
            ):
                invalid_certificate_count += 1
    return cast(
        StableJsonPayload,
        OrderedDict(
            block_pattern=list(pattern),
            support=support,
            seeds=list(seeds),
            generated_instances=len(seeds) * instances_per_seed,
            max_absolute_objective_error=max_absolute_objective_error,
            wrong_minima_count=wrong_minima_count,
            invalid_certificate_count=invalid_certificate_count,
        ),
    )


_COUPLING_VALIDATION_CONFIGURATION_SECTIONS = frozenset(
    {ConfigurationSection.ACTION, ConfigurationSection.GENERATORS, ConfigurationSection.SOLVERS}
)


def execute_coupling_and_map_bound_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> ReusableArtifactManifest:
    seed = ExperimentSeed(request.definition.seeds[0])
    return _persist_synthetic_experiment_payload(
        store,
        layout,
        request,
        seed,
        _coupling_and_map_bound_validation_payload,
        _COUPLING_VALIDATION_CONFIGURATION_SECTIONS,
        _MODULE_NAME,
        "coupling-and-map-bound-validation",
    )


def _coupling_and_map_bound_validation_payload(fingerprint: Sha256Digest) -> StableJsonPayload:
    coupling_config = active_config().generators.coupling_structure
    seeds = active_config().scientific.randomness.confirmatory_seeds
    incompatible_gap_threshold = coupling_config.incompatible_fixed_action_gap_strictly_greater_than
    experiment = ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION
    logger = execution_logger()
    total_cells = sum(
        len(eligible_coupling_support_sizes(compatibility, coupling_config.supports))
        * len(coupling_config.response_heterogeneity)
        * len(coupling_config.directed_asymmetry)
        * len(coupling_config.response_sparsity)
        * len(coupling_config.block_patterns)
        for compatibility in coupling_config.compatibility
    )
    total_planned = total_cells * len(seeds)
    started_at = time.monotonic()
    total_generated = 0
    total_generation_failures = 0
    total_incompatible_gap_failures = 0
    cells_started = 0
    for compatibility in coupling_config.compatibility:
        eligible_supports = eligible_coupling_support_sizes(compatibility, coupling_config.supports)
        for support in eligible_supports:
            for heterogeneity in coupling_config.response_heterogeneity:
                for asymmetry in coupling_config.directed_asymmetry:
                    for sparsity in coupling_config.response_sparsity:
                        for block_pattern in coupling_config.block_patterns:
                            cells_started += 1
                            logger.record(
                                ExecutionLogEvent(
                                    occurred_at=datetime.now(UTC),
                                    cell_coordinates=SemanticCoordinates(
                                        f"{experiment.value}:{compatibility.value}:{support}:"
                                        f"{heterogeneity}:{asymmetry}:{sparsity}:{block_pattern}"
                                    ),
                                    artifact_id=None,
                                    state=ArtifactState.RUNNING,
                                    stage=ExecutionStageName(ArtifactStage.EVALUATION.value),
                                    experiment=experiment,
                                    elapsed_seconds=time.monotonic() - started_at,
                                    reuse_decision=ReuseDecision(
                                        f"{cells_started}/{total_cells} cells, "
                                        f"{total_generated}/{total_planned} instances, "
                                        f"{total_generation_failures} generation failures"
                                    ),
                                )
                            )
                            for seed in seeds:
                                generated, gap_failed = _coupling_validation_instance(
                                    compatibility,
                                    heterogeneity,
                                    asymmetry,
                                    sparsity,
                                    block_pattern,
                                    support,
                                    seed,
                                    incompatible_gap_threshold,
                                )
                                total_generated += 1
                                if not generated:
                                    total_generation_failures += 1
                                elif gap_failed:
                                    total_incompatible_gap_failures += 1
    logger.record(
        ExecutionLogEvent(
            occurred_at=datetime.now(UTC),
            cell_coordinates=SemanticCoordinates(f"{experiment.value}:map_bound_fixtures"),
            artifact_id=None,
            state=ArtifactState.RUNNING,
            stage=ExecutionStageName(ArtifactStage.EVALUATION.value),
            experiment=experiment,
            elapsed_seconds=time.monotonic() - started_at,
        )
    )
    map_bound_results = _map_bound_fixture_results(seeds)
    logger.record(
        ExecutionLogEvent(
            occurred_at=datetime.now(UTC),
            cell_coordinates=SemanticCoordinates(f"{experiment.value}"),
            artifact_id=None,
            state=ArtifactState.COMPLETED,
            stage=ExecutionStageName(ArtifactStage.EVALUATION.value),
            experiment=experiment,
            elapsed_seconds=time.monotonic() - started_at,
            reuse_decision=ReuseDecision(
                f"{total_generated}/{total_planned} instances, "
                f"{total_generation_failures} generation failures, "
                f"{total_incompatible_gap_failures} gap failures"
            ),
        )
    )
    return cast(
        StableJsonPayload,
        OrderedDict(
            experiment=ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION.value,
            dependency_fingerprint_sha256=fingerprint,
            total_instances=total_generated,
            total_generation_failures=total_generation_failures,
            total_incompatible_gap_failures=total_incompatible_gap_failures,
            map_bound_fixtures=map_bound_results,
        ),
    )


def _coupling_validation_instance(
    compatibility: CouplingCompatibility,
    heterogeneity: Coefficient,
    asymmetry: Coefficient,
    sparsity: Coefficient,
    block_pattern: tuple[ConceptCount, ...],
    support: SupportCount,
    seed: RandomSeed,
    incompatible_gap_threshold: Threshold,
) -> tuple[bool, bool]:
    request = CouplingInstanceRequest(
        compatibility=compatibility,
        response_heterogeneity=heterogeneity,
        directed_asymmetry=asymmetry,
        response_sparsity=sparsity,
        block_pattern=tuple(block_pattern),
        support_size=support,
        seed=seed,
        instance_index=0,
    )
    try:
        instance = generate_coupling_instance(request)
    except CouplingGenerationError:
        return (False, False)
    if compatibility != CouplingCompatibility.INCOMPATIBLE:
        return (True, False)
    groups = tuple(CoarseGroup)[: len(instance.block_pattern)]
    counts = OrderedDict(zip(groups, instance.block_pattern, strict=True))
    blocks = build_padded_block_structure(groups, counts, counts)
    orbit = tuple(enumerate_block_permutations(blocks))
    problem = build_robust_action_problem(
        blocks,
        instance.lower_response_matrix,
        instance.lower_response_matrix,
        instance.target_importance,
        tuple(range(sum(instance.block_pattern))),
    )
    alpha = CurriculumAction(problem=problem, coordinates=instance.active_action)
    hull = build_rectangular_hull(
        blocks, instance.lower_response_matrix, instance.lower_response_matrix
    )
    gap = fixed_action_rectangularization_gap(alpha, orbit, hull.lower_bounds)
    return (True, not gap > incompatible_gap_threshold)


def _map_bound_fixture_results(seeds: tuple[RandomSeed, ...]) -> StableJsonPayload:
    zero_map_value_failures = 0
    high_map_value_failures = 0
    for seed in seeds:
        try:
            generate_unresolved_map_world(
                UnresolvedMapWorldRequest(UnresolvedMapWorldKind.COMMON_ACTION, seed)
            )
        except MechanismGenerationError:
            zero_map_value_failures += 1
        try:
            generate_unresolved_map_world(
                UnresolvedMapWorldRequest(UnresolvedMapWorldKind.MAP_DEPENDENT, seed)
            )
        except MechanismGenerationError:
            high_map_value_failures += 1
    return cast(
        StableJsonPayload,
        OrderedDict(
            zero_map_value_fixtures=len(seeds),
            zero_map_value_failures=zero_map_value_failures,
            high_map_value_fixtures=len(seeds),
            high_map_value_failures=high_map_value_failures,
        ),
    )


_BASE_MODEL_PILOT_CONFIGURATION_SECTIONS = frozenset({ConfigurationSection.MODELS})


def build_dataset_manifest(materialized: MaterializedClient) -> DatasetManifest:
    provenance = materialized.provenance
    raw_files = tuple(entry.path for entry in provenance.raw_files)
    raw_sha256 = Sha256Digest(
        hashlib.sha256(
            ",".join(f"{entry.path}:{entry.sha256}" for entry in provenance.raw_files).encode(
                "utf-8"
            )
        ).hexdigest()
    )
    raw_counts = OrderedDict((entry.path, entry.row_count) for entry in provenance.raw_files)
    adapter_feature_roles = OrderedDict(
        (column, materialized.schema.role_of(column).value)
        for column in materialized.schema.feature_order
    )
    local_class_counts = OrderedDict(
        (label, sum(counts.values())) for label, counts in materialized.class_row_counts.items()
    )
    transfer_candidate_counts: Mapping[str, SampleCount] = OrderedDict(
        (str(group.concept.value), group.train_support)
        for group in transfer_concept_groups(materialized.dataset, materialized)
    )
    feature_quality = FeatureQualityManifest(
        dropped_feature_count=materialized.feature_quality.dropped_feature_count,
        candidate_count_before_filtering=(
            materialized.feature_quality.candidate_count_before_filtering
        ),
        client_invalid=materialized.feature_quality.client_invalid,
        client_invalid_reason=materialized.feature_quality.client_invalid_reason,
    )
    dependency_fingerprint_sha256 = Sha256Digest(
        hashlib.sha256(
            f"{raw_sha256}|{','.join(materialized.schema.feature_order)}".encode()
        ).hexdigest()
    )
    return DatasetManifest.model_validate(
        OrderedDict(
            dataset=materialized.dataset,
            component=provenance.component,
            raw_files=raw_files,
            raw_sha256=raw_sha256,
            raw_counts=raw_counts,
            schema=ArtifactSchemaVersion("1.0"),
            adapter_feature_order=materialized.schema.feature_order,
            adapter_feature_roles=adapter_feature_roles,
            accepted_schema_aliases=(provenance.accepted_timestamp_column,),
            adapter_adaptations=(),
            timestamp_field=provenance.accepted_timestamp_column,
            timestamp_range=provenance.timestamp_range,
            duplicate_counts=OrderedDict(total=provenance.duplicate_group_count),
            conflicting_duplicate_counts=OrderedDict(
                total=provenance.conflicting_duplicate_group_count
            ),
            local_class_counts=local_class_counts,
            transfer_candidate_counts=transfer_candidate_counts,
            feature_quality=feature_quality,
            preprocessing_state=DatasetPreprocessingState.MATERIALIZED,
            dependency_fingerprint_sha256=dependency_fingerprint_sha256,
            producer_code_sha256=Sha256Digest(
                implementation_fingerprint("fedorbit.datasets.materialization")
            ),
        )
    )


def persist_dataset_manifest(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    dataset: DatasetId,
    manifest: DatasetManifest,
) -> Path:
    destination = (
        experiment_workspace(layout, experiment)
        / "artifacts"
        / "derived"
        / f"dataset-manifest.{dataset.value}.json"
    )
    atomic_write_json(destination, manifest.model_dump(mode="json"))
    return destination


def persist_materialized_client(
    layout: WorkspaceLayout,
    materialized: MaterializedClient,
    overwrite_policy: OverwritePolicy,
) -> tuple[Path, ...]:
    dataset = materialized.dataset
    split_paths: list[Path] = []
    for split, tensors in materialized.splits.items():
        destination = layout.preprocessing / "splits" / dataset.value / split.value / "data.parquet"
        if overwrite_policy == OverwritePolicy.REUSE and destination.is_file():
            split_paths.append(destination)
            continue
        frame = pd.DataFrame(
            tensors.features.detach().cpu().numpy(),
            columns=materialized.feature_names,
        )
        frame.insert(len(frame.columns), "target", tensors.targets.detach().cpu().numpy())
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".parquet")
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        split_paths.append(destination)
    manifest = build_dataset_manifest(materialized)
    atomic_write_json(
        layout.preprocessing / "prepared" / dataset.value / "data.json",
        manifest.model_dump(mode="json"),
    )
    eligibility = tuple(
        cast(
            StableJsonPayload,
            OrderedDict(
                concept=group.concept.value,
                native_class_indices=group.native_class_indices,
                train_support=group.train_support,
                meta_support=group.meta_support,
                source_eligible=group.source_eligible,
            ),
        )
        for group in transfer_concept_groups(dataset, materialized)
    )
    atomic_write_json(
        layout.preprocessing / "features" / dataset.value / "data.json",
        cast(
            StableJsonPayload,
            OrderedDict(
                feature_names=materialized.feature_names,
                local_class_names=materialized.class_manifest.class_names,
                excluded_classes=materialized.class_manifest.excluded_classes,
                transfer_eligibility=eligibility,
            ),
        ),
    )
    return tuple(split_paths)


def _persist_client_invalid(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    dataset: DatasetId,
    reason: InvalidReason,
) -> None:
    destination = experiment_workspace(layout, experiment) / "artifacts" / "derived"
    payload = cast(
        StableJsonPayload,
        OrderedDict(
            experiment=experiment.value,
            dataset=dataset.value,
            state=ArtifactState.INVALID.value,
            reason=reason,
        ),
    )
    atomic_write_json(destination / f"{dataset.value}-invalid.json", payload)


def execute_base_model_pilot(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    experiment = request.experiment
    relevance = experiment_relevance(experiment)
    raw_root = raw_dataset_root()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    device = torch.device("cuda")
    logger = execution_logger()
    for dataset in active_config().scientific.datasets.clients:
        logger.record(
            ExecutionLogEvent(
                occurred_at=datetime.now(UTC),
                cell_coordinates=SemanticCoordinates(f"{experiment.value}:{dataset.value}"),
                artifact_id=None,
                state=ArtifactState.RUNNING,
                stage=ExecutionStageName(ArtifactStage.PREPROCESSING.value),
                experiment=experiment,
                dataset=dataset,
            )
        )

        materialize_started_at = time.monotonic()
        try:
            materialized = materialize_client(dataset, raw_root)
        except MaterializationError as error:
            _persist_client_invalid(layout, experiment, dataset, InvalidReason(str(error)))
            logger.record(
                ExecutionLogEvent(
                    occurred_at=datetime.now(UTC),
                    cell_coordinates=SemanticCoordinates(f"{experiment.value}:{dataset.value}"),
                    artifact_id=None,
                    state=ArtifactState.INVALID,
                    stage=ExecutionStageName(ArtifactStage.PREPROCESSING.value),
                    experiment=experiment,
                    dataset=dataset,
                    elapsed_seconds=time.monotonic() - materialize_started_at,
                )
            )
            continue
        logger.record(
            ExecutionLogEvent(
                occurred_at=datetime.now(UTC),
                cell_coordinates=SemanticCoordinates(f"{experiment.value}:{dataset.value}"),
                artifact_id=None,
                state=ArtifactState.COMPLETED,
                stage=ExecutionStageName(ArtifactStage.PREPROCESSING.value),
                experiment=experiment,
                dataset=dataset,
                elapsed_seconds=time.monotonic() - materialize_started_at,
            )
        )
        persist_dataset_manifest(layout, experiment, dataset, build_dataset_manifest(materialized))
        _execute_client_base_model_pilot(
            store,
            layout,
            experiment,
            relevance,
            dataset,
            materialized,
            confirmatory_seeds,
            request.overwrite_policy,
            device,
            logger,
        )


_SOURCE_RESPONSE_PILOT_CONFIGURATION_SECTIONS = frozenset(
    {ConfigurationSection.MODELS, ConfigurationSection.RESPONSE}
)


def execute_source_response_estimator_pilot(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> ReusableArtifactManifest:
    seed = ExperimentSeed(active_config().scientific.randomness.pilot_seeds[0])
    return _persist_synthetic_experiment_payload(
        store,
        layout,
        request,
        seed,
        lambda fingerprint: _source_response_estimator_payload(layout, request, fingerprint),
        _SOURCE_RESPONSE_PILOT_CONFIGURATION_SECTIONS,
        _MODULE_NAME,
        "source-response-pilot",
    )


def _source_response_estimator_payload(
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    fingerprint: Sha256Digest,
) -> StableJsonPayload:
    return _source_response_estimator_client_results(layout, request, fingerprint)


def _execute_final_source_response_band_validation(
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    selected_root = (
        experiment_workspace(layout, ExperimentName.SOURCE_RESPONSE_ESTIMATOR_PILOT)
        / "artifacts"
        / "fitted"
    )
    checkpoint_root = (
        experiment_workspace(layout, ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT)
        / "checkpoints"
        / "training"
    )
    for dataset in active_config().scientific.datasets.clients:
        try:
            materialized = materialize_client(dataset, raw_root)
        except MaterializationError as error:
            _persist_client_invalid(layout, request.experiment, dataset, InvalidReason(str(error)))
            continue
        selected_path = selected_root / f"{dataset.value}.json"
        if not selected_path.is_file():
            raise ExecutionError(f"missing selected response configuration: {selected_path}")
        selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
        candidate = ResponseCandidate(
            selected_payload["intervention_magnitude"],
            selected_payload["optimizer_step_horizon"],
        )
        eligible_by_group: OrderedDict[CoarseGroup, list[TransferConceptGroup]] = OrderedDict()
        for group in transfer_concept_groups(dataset, materialized):
            if group.source_eligible:
                eligible_by_group.setdefault(TRANSFER_ONTOLOGY[group.concept][0], []).append(group)
        for seed in active_config().scientific.randomness.confirmatory_seeds:
            checkpoint_path = checkpoint_root / dataset.value / f"seed-{seed}" / "checkpoint.pt"
            if not checkpoint_path.is_file():
                raise ExecutionError(f"missing confirmatory checkpoint: {checkpoint_path}")
            checkpoint = load_base_checkpoint(checkpoint_path)
            for coarse_group, groups in eligible_by_group.items():
                node_classes = tuple(group.native_class_indices for group in groups)
                model = create_classifier(
                    dataset,
                    materialized.splits[Split.TRAIN].features.shape[1],
                    materialized.class_manifest.class_count,
                    checkpoint.selected_hyperparameters.dropout_probability,
                    seed,
                )
                checkpoint.state_dict.load_into(model)
                packet = construct_source_packet(
                    PacketConstructionContext(
                        dataset,
                        materialized.splits[Split.TRAIN].features.shape[1],
                        materialized.class_manifest.class_count,
                        coarse_group,
                        tuple(AnonymousNodeDisplayId(group.concept.value) for group in groups),
                        tuple(group.train_support for group in groups),
                        tuple(group.meta_support for group in groups),
                        file_sha256(checkpoint_path),
                        Sha256Digest(hashlib.sha256(selected_path.read_bytes()).hexdigest()),
                        seed,
                    ),
                    checkpoint,
                    model,
                    materialized.splits[Split.TRAIN].features,
                    materialized.splits[Split.TRAIN].targets,
                    materialized.splits[Split.META].features,
                    materialized.splits[Split.META].targets,
                    node_classes,
                    node_classes,
                    checkpoint.train_class_weights,
                    candidate,
                    Rfc3339UtcTimestamp(datetime.now(UTC).isoformat().replace("+00:00", "Z")),
                ).packet
                destination = (
                    experiment_workspace(layout, request.experiment)
                    / "artifacts"
                    / "packets"
                    / dataset.value
                    / f"seed-{seed}"
                    / f"{coarse_group.value.casefold().replace(' ', '-')}.json"
                )
                atomic_write_bytes(destination, (packet.serialized() + "\n").encode("utf-8"))


def _source_response_estimator_client_results(
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    fingerprint: Sha256Digest,
) -> StableJsonPayload:
    raw_root = raw_dataset_root()
    pilot_seeds = active_config().scientific.randomness.pilot_seeds
    base_workspace = experiment_workspace(layout, ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT)
    destination = experiment_workspace(layout, request.experiment) / "artifacts" / "fitted"
    diagnostics_destination = (
        experiment_workspace(layout, request.experiment) / "artifacts" / "derived"
    )
    client_results: list[StableJsonPayload] = []
    for dataset in active_config().scientific.datasets.clients:
        try:
            materialized = materialize_client(dataset, raw_root)
        except MaterializationError as error:
            _persist_client_invalid(layout, request.experiment, dataset, InvalidReason(str(error)))
            client_results.append(
                cast(
                    StableJsonPayload,
                    OrderedDict(dataset=dataset.value, state=ArtifactState.INVALID.value),
                )
            )
            continue
        groups = transfer_concept_groups(dataset, materialized)
        eligible = tuple(group for group in groups if group.source_eligible)
        if len(eligible) < 2:
            _persist_client_invalid(
                layout,
                request.experiment,
                dataset,
                InvalidReason("fewer than two eligible source transfer concepts"),
            )
            client_results.append(
                cast(
                    StableJsonPayload,
                    OrderedDict(dataset=dataset.value, state=ArtifactState.INVALID.value),
                )
            )
            continue
        checkpoints: list[PilotCheckpoint] = []
        for seed in pilot_seeds:
            path = (
                base_workspace
                / "checkpoints"
                / "pilot"
                / dataset.value
                / f"seed-{seed}"
                / "checkpoint.pt"
            )
            if not path.is_file():
                raise ExecutionError(f"missing selected pilot checkpoint: {path}")
            checkpoint = load_base_checkpoint(path)
            model = create_classifier(
                dataset,
                materialized.splits[Split.TRAIN].features.shape[1],
                materialized.class_manifest.class_count,
                checkpoint.selected_hyperparameters.dropout_probability,
                seed,
            )
            checkpoint.state_dict.load_into(model)
            checkpoints.append(PilotCheckpoint(model, checkpoint, seed))
        checkpoint = checkpoints[0].checkpoint
        intervention_classes = tuple(group.native_class_indices for group in eligible)
        data = ResponsePilotData(
            materialized.splits[Split.TRAIN].features,
            materialized.splits[Split.TRAIN].targets,
            materialized.splits[Split.META].features,
            materialized.splits[Split.META].targets,
            intervention_classes,
            checkpoint.train_class_weights,
            checkpoint.selected_hyperparameters.learning_rate,
            checkpoint.selected_hyperparameters.weight_decay,
        )
        results = run_pooled_source_response_pilot(tuple(checkpoints), data, intervention_classes)
        selected = select_response_configuration(results)
        atomic_write_json(
            diagnostics_destination / f"{dataset.value}-candidates.json",
            cast(
                StableJsonPayload,
                OrderedDict(
                    dataset=dataset.value,
                    pilot_checkpoint_seeds=pilot_seeds,
                    candidates=tuple(asdict(result) for result in results),
                ),
            ),
        )
        atomic_write_json(
            destination / f"{dataset.value}.json",
            cast(
                StableJsonPayload,
                OrderedDict(
                    dataset=dataset.value,
                    intervention_magnitude=selected.intervention_magnitude,
                    optimizer_step_horizon=selected.optimizer_step_horizon,
                    pilot_checkpoint_seeds=pilot_seeds,
                ),
            ),
        )
        client_results.append(
            cast(
                StableJsonPayload,
                OrderedDict(
                    dataset=dataset.value,
                    state=ArtifactState.COMPLETED.value,
                    selected_configuration=cast(
                        StableJsonPayload,
                        OrderedDict(
                            intervention_magnitude=selected.intervention_magnitude,
                            optimizer_step_horizon=selected.optimizer_step_horizon,
                        ),
                    ),
                ),
            )
        )
    return cast(
        StableJsonPayload,
        OrderedDict(
            experiment=request.experiment.value,
            dependency_fingerprint_sha256=fingerprint,
            clients=tuple(client_results),
        ),
    )


def execute_final_source_response_band_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    del store
    _execute_final_source_response_band_validation(layout, request)


_PRIMARY_TRANSFER_CONFIGURATION_SECTIONS = frozenset(
    {ConfigurationSection.MODELS, ConfigurationSection.METRICS}
)


def _target_confirmatory_checkpoint_path(
    layout: WorkspaceLayout,
    target: DatasetId,
    seed: RandomSeed,
    checkpoint_source_experiment: ExperimentName = ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT,
) -> Path:
    return (
        experiment_workspace(layout, checkpoint_source_experiment)
        / "checkpoints"
        / CheckpointDirectorySegment.TRAINING
        / target.value
        / f"seed-{seed}"
        / "checkpoint.pt"
    )


def _checkpoint_artifact_id(
    store: ArtifactStore, checkpoint_path: Path
) -> ArtifactIdentifier | None:
    target_path = str(checkpoint_path)
    for manifest in store.all_manifests():
        if manifest.payload_paths == (target_path,):
            return manifest.artifact_id
    return None


def _score_local_only_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    target: DatasetId,
    materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
    checkpoint_source_experiment: ExperimentName = ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT,
) -> tuple[ScoreArtifact, ClassCount, ArtifactIdentifier] | None:
    checkpoint_path = _target_confirmatory_checkpoint_path(
        layout, target, seed, checkpoint_source_experiment
    )
    if not checkpoint_path.is_file():
        return None
    checkpoint_artifact_id = _checkpoint_artifact_id(store, checkpoint_path)
    if checkpoint_artifact_id is None:
        return None
    checkpoint = load_base_checkpoint(checkpoint_path)
    test = materialized.splits[Split.TEST]
    n_classes = materialized.class_manifest.class_count
    model = create_classifier(
        target,
        test.features.shape[1],
        n_classes,
        checkpoint.selected_hyperparameters.dropout_probability,
        seed,
        device,
    )
    checkpoint.state_dict.load_into(model)
    score = score_model(
        ScoringRequest(model, test.features, test.targets, LocalClassCount(n_classes))
    )
    return score, n_classes, checkpoint_artifact_id


def class_metric_sets(
    score: ScoreArtifact, n_classes: ClassCount
) -> tuple[ClassF1Set, ClassRecallSet]:
    predicted = tuple(ClassIndex(row.predicted_class.value) for row in score.rows)
    actual = tuple(ClassIndex(row.target.value) for row in score.rows)
    f1_values: list[ClassF1] = []
    recall_values: list[ClassRecall] = []
    for class_index in range(n_classes):
        counts = confusion_counts(predicted, actual, ClassIndex(class_index))
        recall_values.append(
            ClassRecall(recall_from_counts(counts.true_positives, counts.false_negatives))
        )
        f1_values.append(
            ClassF1(
                f1_from_counts(
                    counts.true_positives, counts.false_positives, counts.false_negatives
                )
            )
        )
    return ClassF1Set(tuple(f1_values)), ClassRecallSet(tuple(recall_values))


def persist_primary_transfer_metric(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    pair_direction: DirectedPairName,
    directed_pair_source: DatasetId,
    directed_pair_target: DatasetId,
    method: TransferMethod,
    seed: RandomSeed,
    metric_name: MetricId,
    metric_value: float,
    metric_unit: MetricUnit,
    direction: MetricDirection,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
    condition: EvaluationConditionName = _PRINCIPAL_CONDITION,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        directed_pair=DirectedPair(source=directed_pair_source, target=directed_pair_target),
        method=method,
        condition=ExperimentCondition(condition),
        seed=ExperimentSeed(seed),
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.EVALUATION,
            cell,
            relevance,
            tuple(identifier.value for identifier in input_artifact_ids),
            _PRIMARY_TRANSFER_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    metric = MetricRecord(
        experiment=experiment,
        pair=pair_direction,
        method=method,
        condition=condition,
        seed=seed,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        direction=direction,
        evaluation_class_set_sha256=Sha256Digest(
            hashlib.sha256(coordinates.encode("utf-8")).hexdigest()
        ),
        input_artifact_ids=tuple(input_artifact_ids),
        dependency_fingerprint_sha256=fingerprint,
        valid=True,
        invalid_reason=None,
    )
    validate_metric_records(MetricRecordCollection((metric,)))
    payload_path = (
        experiment_workspace(layout, experiment)
        / "artifacts"
        / "derived"
        / (
            f"metric.{directed_pair_source.value}-{directed_pair_target.value}"
            f".{method.value}.{condition}.{seed}.{metric_name.value}.json"
        )
    )
    payload = cast(StableJsonPayload, OrderedDict(metric_record=metric.model_dump(mode="json")))
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_PRIMARY_TRANSFER_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(ArtifactStage.EVALUATION).sha256)
    completion = _completion(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=ArtifactStage.EVALUATION,
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.PREDICTION.value), payload, Sha256Digest(fingerprint)
            ),
            artifact_type=ArtifactType.PREDICTION,
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


def execute_primary_strict_cross_telemetry_transfer(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_target: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()
    for directed_pair in primary_pairs:
        target = directed_pair.target
        if target not in materialized_by_target:
            try:
                materialized_by_target[target] = materialize_client(target, raw_root)
            except MaterializationError:
                continue
        materialized = materialized_by_target[target]
        source = directed_pair.source
        if source not in materialized_by_target:
            with contextlib.suppress(MaterializationError):
                materialized_by_target[source] = materialize_client(source, raw_root)
        source_materialized = materialized_by_target.get(source)
        pair_direction = DirectedPairName(
            f"{directed_pair.source.value} -> {directed_pair.target.value}"
        )
        for seed in confirmatory_seeds:
            local_only = _score_local_only_cell(store, layout, target, materialized, seed, device)
            if local_only is not None:
                score, n_classes, checkpoint_artifact_id = local_only
                _persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    directed_pair.source,
                    directed_pair.target,
                    TransferMethod.LOCAL_ONLY,
                    seed,
                    score,
                    n_classes,
                    (checkpoint_artifact_id,),
                )
            local_sir = _score_local_sir_cell(store, layout, target, materialized, seed, device)
            if local_sir is not None:
                score, n_classes, input_artifact_ids = local_sir
                _persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    directed_pair.source,
                    directed_pair.target,
                    TransferMethod.LOCAL_SIR,
                    seed,
                    score,
                    n_classes,
                    input_artifact_ids,
                )
            if source_materialized is not None:
                matched_resource_rectangular = _score_matched_resource_rectangular_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    materialized,
                    seed,
                    device,
                )
                if matched_resource_rectangular is not None:
                    score, n_classes, input_artifact_ids = matched_resource_rectangular
                    _persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        directed_pair.source,
                        directed_pair.target,
                        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                    )
                point_correspondence = _score_point_correspondence_commitment_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    materialized,
                    seed,
                    device,
                )
                if point_correspondence is not None:
                    score, n_classes, input_artifact_ids = point_correspondence
                    _persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        directed_pair.source,
                        directed_pair.target,
                        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                    )
                fedorbit_exact_sparse = _score_fedorbit_exact_sparse_solver_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    materialized,
                    seed,
                    device,
                )
                if fedorbit_exact_sparse is not None:
                    score, n_classes, input_artifact_ids = fedorbit_exact_sparse
                    _persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        directed_pair.source,
                        directed_pair.target,
                        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                    )
                generic_exact_qap = _score_generic_exact_qap_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    materialized,
                    seed,
                    device,
                )
                if generic_exact_qap is not None:
                    score, n_classes, input_artifact_ids = generic_exact_qap
                    _persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        directed_pair.source,
                        directed_pair.target,
                        TransferMethod.GENERIC_EXACT_QAP,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                    )
                exact_map_oracle = _score_exact_map_oracle_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    materialized,
                    seed,
                    device,
                )
                if exact_map_oracle is not None:
                    score, n_classes, input_artifact_ids = exact_map_oracle
                    _persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        directed_pair.source,
                        directed_pair.target,
                        TransferMethod.EXACT_MAP_ORACLE,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                    )


def _persist_primary_transfer_cell_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    pair_direction: DirectedPairName,
    source: DatasetId,
    target: DatasetId,
    method: TransferMethod,
    seed: RandomSeed,
    score: ScoreArtifact,
    n_classes: ClassCount,
    input_artifact_ids: tuple[ArtifactIdentifier, ...],
    condition: EvaluationConditionName = _PRINCIPAL_CONDITION,
) -> None:
    f1_set, recall_set = class_metric_sets(score, n_classes)
    for metric_name, metric_value, metric_unit, direction in (
        (
            MetricId.MACRO_CROSS_ENTROPY,
            float(score.macro_cross_entropy.value),
            MetricUnit("nats"),
            MetricDirection.LOWER_IS_BETTER,
        ),
        (
            MetricId.MACRO_F1,
            float(macro_f1(f1_set).value),
            MetricUnit("fraction"),
            MetricDirection.HIGHER_IS_BETTER,
        ),
        (
            MetricId.BALANCED_ACCURACY,
            float(balanced_accuracy(recall_set).value),
            MetricUnit("fraction"),
            MetricDirection.HIGHER_IS_BETTER,
        ),
    ):
        persist_primary_transfer_metric(
            store,
            layout,
            request.experiment,
            pair_direction,
            source,
            target,
            method,
            seed,
            metric_name,
            metric_value,
            metric_unit,
            direction,
            input_artifact_ids,
            request.overwrite_policy,
            condition,
        )


_STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS = frozenset({ConfigurationSection.METRICS})


@dataclass(frozen=True, slots=True)
class _SeedMetric:
    value: float
    artifact_id: ArtifactIdentifier


def _iter_completed_json_payloads(
    store: ArtifactStore, experiment: ExperimentName, payload_key: str
) -> Iterator[tuple[ReusableArtifactManifest, Mapping[str, StableJsonPayload]]]:
    experiment_value = experiment.value
    for manifest in store.all_manifests():
        if experiment_value not in manifest.semantic_producer_coordinates:
            continue
        try:
            resolved = store.resolve(manifest.artifact_id)
        except ValueError:
            continue
        if resolved.state != ArtifactState.COMPLETED:
            continue
        for payload_path in resolved.payload_paths:
            path = Path(payload_path)
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            record_payload = payload.get(payload_key)
            if record_payload is not None:
                yield resolved, record_payload


def completed_experiment_metric_records(
    store: ArtifactStore, experiment: ExperimentName
) -> tuple[MetricRecord, ...]:
    return tuple(
        MetricRecord.model_validate(payload)
        for _, payload in _iter_completed_json_payloads(store, experiment, "metric_record")
    )


def completed_experiment_metric_records_with_support(
    store: ArtifactStore, experiment: ExperimentName
) -> tuple[tuple[MetricRecord, SupportCount | None], ...]:
    results: list[tuple[MetricRecord, SupportCount | None]] = []
    for manifest, payload in _iter_completed_json_payloads(store, experiment, "metric_record"):
        record = MetricRecord.model_validate(payload)
        coordinates = json.loads(manifest.semantic_producer_coordinates)
        support = coordinates.get("support")
        results.append((record, support))
    return tuple(results)


def completed_primary_transfer_metric_records(store: ArtifactStore) -> tuple[MetricRecord, ...]:
    return tuple(
        MetricRecord.model_validate(payload)
        for _, payload in _iter_completed_json_payloads(
            store, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER, "metric_record"
        )
    )


def completed_primary_transfer_comparison_records(
    store: ArtifactStore,
) -> tuple[PairedComparisonRecord, ...]:
    return tuple(
        PairedComparisonRecord.model_validate(payload)
        for _, payload in _iter_completed_json_payloads(
            store, ExperimentName.STATISTICAL_SYNTHESIS, "comparison_record"
        )
    )


def _completed_primary_transfer_macro_ce(
    store: ArtifactStore,
) -> Mapping[tuple[DirectedPairName, TransferMethod, RandomSeed], _SeedMetric]:
    result: OrderedDict[tuple[DirectedPairName, TransferMethod, RandomSeed], _SeedMetric] = (
        OrderedDict()
    )
    for resolved, record_payload in _iter_completed_json_payloads(
        store, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER, "metric_record"
    ):
        record = MetricRecord.model_validate(record_payload)
        if (
            record.metric_name != MetricId.MACRO_CROSS_ENTROPY
            or not record.valid
            or record.metric_value is None
        ):
            continue
        result[(record.pair, record.method, record.seed)] = _SeedMetric(
            float(record.metric_value), resolved.artifact_id
        )
    return result


def persist_primary_transfer_comparison(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    pair: DirectedPairName,
    method: TransferMethod,
    paired_seed_count: Index,
    mean_difference: float | None,
    median_difference: float | None,
    bca_ci_low: float | None,
    bca_ci_high: float | None,
    raw_p: float | None,
    holm_p: float | None,
    decision: ComparisonDecision,
    input_metric_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        directed_pair=DirectedPair(
            source=DatasetId(pair.split(" -> ")[0]), target=DatasetId(pair.split(" -> ")[1])
        ),
        method=method,
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.STATISTICS,
            cell,
            relevance,
            tuple(identifier.value for identifier in input_metric_artifact_ids),
            _STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    comparison = PairedComparisonRecord(
        contrast_name=ContrastName(f"{method.value} vs {TransferMethod.LOCAL_ONLY.value}: {pair}"),
        family=MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
        pair=DirectedPairName(pair),
        method_a=method,
        method_b=TransferMethod.LOCAL_ONLY,
        metric=MetricId.MACRO_CROSS_ENTROPY,
        paired_seed_count=paired_seed_count,
        mean_difference=mean_difference,
        median_difference=median_difference,
        bca_ci_low=bca_ci_low,
        bca_ci_high=bca_ci_high,
        raw_p=raw_p,
        holm_p=holm_p,
        materiality_threshold=active_config().scientific.materiality.realized_relative_macro_ce,
        equivalence_margin_low=None,
        equivalence_margin_high=None,
        input_metric_artifact_ids=tuple(input_metric_artifact_ids),
        dependency_fingerprint_sha256=fingerprint,
        decision=decision,
    )
    payload_path = (
        experiment_workspace(layout, experiment)
        / "artifacts"
        / "derived"
        / f"comparison.{pair.replace(' -> ', '-to-')}.{method.value}.json"
    )
    payload = cast(
        StableJsonPayload, OrderedDict(comparison_record=comparison.model_dump(mode="json"))
    )
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(ArtifactStage.STATISTICS).sha256)
    completion = _completion(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=ArtifactStage.STATISTICS,
        upstream_artifact_ids=tuple(input_metric_artifact_ids),
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.OTHER.value), payload, Sha256Digest(fingerprint)
            ),
            artifact_type=ArtifactType.OTHER,
            semantic_producer_coordinates=coordinates,
            producer_stage=ArtifactStage.STATISTICS,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=tuple(input_metric_artifact_ids),
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


def execute_statistical_synthesis(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    metrics = _completed_primary_transfer_macro_ce(store)
    pairs = sorted({pair for pair, _, _ in metrics})
    methods = sorted(
        {method for _, method, _ in metrics if method != TransferMethod.LOCAL_ONLY},
        key=lambda method: method.value,
    )
    statistics_config = active_config().scientific.statistics
    for method in methods:
        raw_p_by_pair: OrderedDict[str, float] = OrderedDict()
        contrasts: OrderedDict[
            str,
            tuple[
                Index,
                float | None,
                float | None,
                float | None,
                float | None,
                tuple[ArtifactIdentifier, ...],
            ],
        ] = OrderedDict()
        for pair in pairs:
            local_only_seeds = OrderedDict(
                (seed, entry)
                for (candidate_pair, candidate_method, seed), entry in metrics.items()
                if candidate_pair == pair and candidate_method == TransferMethod.LOCAL_ONLY
            )
            method_seeds = OrderedDict(
                (seed, entry)
                for (candidate_pair, candidate_method, seed), entry in metrics.items()
                if candidate_pair == pair and candidate_method == method
            )
            shared_seeds = sorted(set(local_only_seeds) & set(method_seeds))
            paired_seed_count: Index = len(shared_seeds)
            input_ids = tuple(
                identifier
                for seed in shared_seeds
                for identifier in (
                    local_only_seeds[seed].artifact_id,
                    method_seeds[seed].artifact_id,
                )
            )
            if len(shared_seeds) < statistics_config.minimum_valid_paired_seeds:
                contrasts[pair] = (paired_seed_count, None, None, None, None, input_ids)
                continue
            local_only_values = tuple(local_only_seeds[seed].value for seed in shared_seeds)
            method_values = tuple(method_seeds[seed].value for seed in shared_seeds)
            bca = paired_bca_interval(
                local_only_values,
                method_values,
                statistical_bootstrap_seed(
                    ContrastName(f"{method.value} vs Local-Only"),
                    MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
                    DirectedPairName(pair),
                    MetricId.MACRO_CROSS_ENTROPY,
                    BootstrapPurpose("primary-transfer-gain"),
                ),
            )
            sign_flip = exact_sign_flip_test(local_only_values, method_values)
            raw_p_by_pair[pair] = sign_flip.p_value
            contrasts[pair] = (
                paired_seed_count,
                float(sign_flip.mean_difference),
                float(sign_flip.median_difference),
                None if bca.lower is None else float(bca.lower),
                None if bca.upper is None else float(bca.upper),
                input_ids,
            )
        holm_adjusted = holm_step_down(
            PValueSet(
                tuple(
                    NamedPValue(PValueName(pair), p_value)
                    for pair, p_value in raw_p_by_pair.items()
                )
            )
        )
        criteria = active_config().scientific.evaluation_criteria.strict_cross_telemetry_utility
        for pair in pairs:
            paired_seed_count, mean_difference, median_difference, bca_low, bca_high, input_ids = (
                contrasts[pair]
            )
            if mean_difference is None:
                decision = ComparisonDecision.INSUFFICIENT_EVIDENCE
                raw_p = None
                holm_p = None
            else:
                raw_p = raw_p_by_pair[pair]
                holm_p = holm_adjusted.value_of(PValueName(pair))
                if bca_low is None:
                    decision = ComparisonDecision.DEGENERATE
                elif (
                    holm_p is not None
                    and holm_p <= criteria.holm_adjusted_p_maximum
                    and bca_low > criteria.bca_lower_bound_strictly_greater_than
                ):
                    decision = ComparisonDecision.SUPERIOR
                else:
                    decision = ComparisonDecision.NOT_SUPPORTED
            if not input_ids:
                continue
            persist_primary_transfer_comparison(
                store,
                layout,
                request.experiment,
                pair,
                method,
                paired_seed_count,
                mean_difference,
                median_difference,
                bca_low,
                bca_high,
                raw_p,
                float(holm_p) if holm_p is not None else None,
                decision,
                input_ids,
                request.overwrite_policy,
            )
    gap_metrics = _completed_real_packet_coupling_gap(store)
    coupling_pairs = sorted({pair for pair, _ in gap_metrics})
    coupling_raw_p_by_pair: OrderedDict[str, float] = OrderedDict()
    coupling_contrasts: OrderedDict[
        str,
        tuple[
            Index,
            float | None,
            float | None,
            float | None,
            float | None,
            tuple[ArtifactIdentifier, ...],
        ],
    ] = OrderedDict()
    for pair in coupling_pairs:
        pair_seeds = OrderedDict(
            (seed, entry)
            for (candidate_pair, seed), entry in gap_metrics.items()
            if candidate_pair == pair
        )
        seeds = sorted(pair_seeds)
        paired_seed_count: Index = len(seeds)
        input_ids = tuple(pair_seeds[seed].artifact_id for seed in seeds)
        if len(seeds) < statistics_config.minimum_valid_paired_seeds:
            coupling_contrasts[pair] = (paired_seed_count, None, None, None, None, input_ids)
            continue
        gap_values = tuple(pair_seeds[seed].value for seed in seeds)
        zero_reference = tuple(0.0 for _ in gap_values)
        bca = paired_bca_interval(
            gap_values,
            zero_reference,
            statistical_bootstrap_seed(
                ContrastName(f"Exact correspondence orbit vs Matched-Resource Rectangular: {pair}"),
                MultiplicityFamily.COUPLING_MECHANISM,
                DirectedPairName(pair),
                MetricId.ROBUST_COUPLING_VALUE_GAP,
                BootstrapPurpose("coupling-mechanism-gap"),
            ),
        )
        sign_flip = exact_sign_flip_test(gap_values, zero_reference)
        coupling_raw_p_by_pair[pair] = sign_flip.p_value
        coupling_contrasts[pair] = (
            paired_seed_count,
            float(sign_flip.mean_difference),
            float(sign_flip.median_difference),
            None if bca.lower is None else float(bca.lower),
            None if bca.upper is None else float(bca.upper),
            input_ids,
        )
    coupling_holm_adjusted = holm_step_down(
        PValueSet(
            tuple(
                NamedPValue(PValueName(pair), p_value)
                for pair, p_value in coupling_raw_p_by_pair.items()
            )
        )
    )
    coupling_criteria = active_config().scientific.evaluation_criteria.coupling_mechanism
    for pair in coupling_pairs:
        (
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            input_ids,
        ) = coupling_contrasts[pair]
        if mean_difference is None:
            decision = ComparisonDecision.INSUFFICIENT_EVIDENCE
            raw_p = None
            holm_p = None
        else:
            raw_p = coupling_raw_p_by_pair[pair]
            holm_p = coupling_holm_adjusted.value_of(PValueName(pair))
            if bca_low is None:
                decision = ComparisonDecision.DEGENERATE
            elif (
                holm_p is not None
                and holm_p <= coupling_criteria.holm_adjusted_p_maximum
                and bca_low > 0.0
            ):
                decision = ComparisonDecision.SUPERIOR
            else:
                decision = ComparisonDecision.NOT_SUPPORTED
        if not input_ids:
            continue
        persist_coupling_mechanism_comparison(
            store,
            layout,
            request.experiment,
            DirectedPairName(pair),
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            raw_p,
            float(holm_p) if holm_p is not None else None,
            decision,
            input_ids,
            request.overwrite_policy,
        )


def _completed_real_packet_coupling_gap(
    store: ArtifactStore,
) -> Mapping[tuple[DirectedPairName, RandomSeed], _SeedMetric]:
    result: OrderedDict[tuple[DirectedPairName, RandomSeed], _SeedMetric] = OrderedDict()
    for resolved, record_payload in _iter_completed_json_payloads(
        store, ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION, "metric_record"
    ):
        record = MetricRecord.model_validate(record_payload)
        if (
            record.metric_name != MetricId.ROBUST_COUPLING_VALUE_GAP
            or record.method != TransferMethod.MATCHED_RESOURCE_RECTANGULAR
            or not record.valid
            or record.metric_value is None
        ):
            continue
        result[(record.pair, record.seed)] = _SeedMetric(
            float(record.metric_value), resolved.artifact_id
        )
    return result


def persist_coupling_mechanism_comparison(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    pair: DirectedPairName,
    paired_seed_count: Index,
    mean_difference: float | None,
    median_difference: float | None,
    bca_ci_low: float | None,
    bca_ci_high: float | None,
    raw_p: float | None,
    holm_p: float | None,
    decision: ComparisonDecision,
    input_metric_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        directed_pair=DirectedPair(
            source=DatasetId(pair.split(" -> ")[0]), target=DatasetId(pair.split(" -> ")[1])
        ),
        method=TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.STATISTICS,
            cell,
            relevance,
            tuple(identifier.value for identifier in input_metric_artifact_ids),
            _STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    comparison = PairedComparisonRecord(
        contrast_name=ContrastName(
            f"Exact correspondence orbit vs Matched-Resource Rectangular: {pair}"
        ),
        family=MultiplicityFamily.COUPLING_MECHANISM,
        pair=DirectedPairName(pair),
        method_a=ExperimentLocalMethod.EXACT_ORBIT,
        method_b=TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
        metric=MetricId.ROBUST_COUPLING_VALUE_GAP,
        paired_seed_count=paired_seed_count,
        mean_difference=mean_difference,
        median_difference=median_difference,
        bca_ci_low=bca_ci_low,
        bca_ci_high=bca_ci_high,
        raw_p=raw_p,
        holm_p=holm_p,
        materiality_threshold=active_config().scientific.materiality.coupling_objective_units,
        equivalence_margin_low=None,
        equivalence_margin_high=None,
        input_metric_artifact_ids=tuple(input_metric_artifact_ids),
        dependency_fingerprint_sha256=fingerprint,
        decision=decision,
    )
    payload_path = (
        experiment_workspace(layout, experiment)
        / "artifacts"
        / "derived"
        / f"coupling-comparison.{pair.replace(' -> ', '-to-')}.json"
    )
    payload = cast(
        StableJsonPayload, OrderedDict(comparison_record=comparison.model_dump(mode="json"))
    )
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(ArtifactStage.STATISTICS).sha256)
    completion = _completion(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=ArtifactStage.STATISTICS,
        upstream_artifact_ids=tuple(input_metric_artifact_ids),
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.OTHER.value), payload, Sha256Digest(fingerprint)
            ),
            artifact_type=ArtifactType.OTHER,
            semantic_producer_coordinates=coordinates,
            producer_stage=ArtifactStage.STATISTICS,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=tuple(input_metric_artifact_ids),
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


def _dataset_eligible_groups_by_coarse(
    target: DatasetId, materialized: MaterializedClient
) -> Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]]:
    eligible_by_group: OrderedDict[CoarseGroup, list[TransferConceptGroup]] = OrderedDict()
    for group in transfer_concept_groups(target, materialized):
        if group.source_eligible:
            eligible_by_group.setdefault(TRANSFER_ONTOLOGY[group.concept][0], []).append(group)
    return OrderedDict((coarse, tuple(groups)) for coarse, groups in eligible_by_group.items())


def self_padded_blocks(
    eligible_groups_by_coarse: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]],
) -> PaddedBlockStructure:
    coarse_groups = tuple(eligible_groups_by_coarse.keys())
    counts = OrderedDict(
        (coarse, len(groups)) for coarse, groups in eligible_groups_by_coarse.items()
    )
    return build_padded_block_structure(coarse_groups, counts, counts)


def _load_dataset_source_packet(
    layout: WorkspaceLayout,
    target: DatasetId,
    seed: RandomSeed,
    coarse_group: CoarseGroup,
) -> SourcePacket | None:
    path = (
        experiment_workspace(layout, ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION)
        / "artifacts"
        / "packets"
        / target.value
        / f"seed-{seed}"
        / f"{coarse_group.value.casefold().replace(' ', '-')}.json"
    )
    if not path.is_file():
        return None
    return SourcePacket.from_serialized(SerializedPacket(path.read_text(encoding="utf-8")))


def assemble_self_response_matrix(
    blocks: PaddedBlockStructure,
    packets_by_coarse_group: Mapping[CoarseGroup, SourcePacket],
) -> ResponseMatrix:
    size = blocks.total_padded_nodes
    matrix: ResponseMatrix = np.zeros((size, size), dtype=np.float64)
    for block_index, coarse_group in enumerate(blocks.coarse_groups):
        packet = packets_by_coarse_group.get(coarse_group)
        if packet is None:
            continue
        block_range = blocks.block_index_range(block_index)
        block_size = block_range.stop - block_range.start
        submatrix = packet.lower_matrix()
        if submatrix.shape != (block_size, block_size):
            raise ExecutionError(
                f"source packet for {coarse_group.value} has shape {submatrix.shape}, "
                f"expected {(block_size, block_size)}"
            )
        matrix[block_range.start : block_range.stop, block_range.start : block_range.stop] = (
            submatrix
        )
    return matrix


def target_node_risks(
    blocks: PaddedBlockStructure,
    eligible_groups_by_coarse: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]],
    class_conditional_cross_entropy: tuple[float, ...],
) -> tuple[TransferNodeRisk, ...]:
    risks: list[TransferNodeRisk] = []
    for block_index, coarse_group in enumerate(blocks.coarse_groups):
        groups = eligible_groups_by_coarse.get(coarse_group, ())
        block_range = blocks.block_index_range(block_index)
        for offset, node_index in enumerate(block_range):
            if offset < len(groups):
                relevant = [
                    class_conditional_cross_entropy[class_index]
                    for class_index in groups[offset].native_class_indices
                    if math.isfinite(class_conditional_cross_entropy[class_index])
                ]
                risk = statistics.fmean(relevant) if relevant else 0.0
                risks.append(TransferNodeRisk(node_index, True, risk))
            else:
                risks.append(TransferNodeRisk(node_index, False, 0.0))
    return tuple(risks)


def curriculum_multipliers_from_action(
    action: CurriculumAction,
    blocks: PaddedBlockStructure,
    eligible_groups_by_coarse: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]],
    n_classes: ClassCount,
) -> CurriculumMultipliers:
    values = torch.ones(n_classes, dtype=torch.float64)
    for block_index, coarse_group in enumerate(blocks.coarse_groups):
        groups = eligible_groups_by_coarse.get(coarse_group, ())
        block_range = blocks.block_index_range(block_index)
        for offset, node_index in enumerate(block_range):
            if offset >= len(groups):
                continue
            alpha = float(action.coordinates[node_index])
            if alpha <= 0.0:
                continue
            for class_index in groups[offset].native_class_indices:
                values[class_index] = 1.0 + alpha
    return CurriculumMultipliers(values)


def _confirm_assimilate_and_score(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    checkpoint: BaseCheckpoint,
    train: SplitTensors,
    confirm: SplitTensors,
    test: SplitTensors,
    multipliers: CurriculumMultipliers,
    seed: RandomSeed,
    contrast_coordinates: ContrastCoordinates,
    assimilation_coordinates: AssimilationCoordinates,
    n_classes: ClassCount,
) -> ScoreArtifact:
    pre_confirm = capture_pre_confirm_pair(model, optimizer)
    verdict = run_proposal_confirmation(
        ConfirmationRequest(
            model,
            pre_confirm.baseline,
            pre_confirm.curriculum,
            train.features,
            train.targets,
            confirm.features,
            confirm.targets,
            checkpoint.train_class_weights,
            multipliers,
            checkpoint.selected_hyperparameters,
            seed,
            contrast_coordinates,
        )
    )
    lifecycle = PreTestLifecycle()
    lifecycle.complete_phase(PreTestPhase.SOURCE_SELECTION_FINALIZED)
    lifecycle.complete_phase(PreTestPhase.ACTION_FINALIZED)
    lifecycle.complete_phase(PreTestPhase.CONFIRMATION_DECISION_FINALIZED)
    if verdict.accepted:
        apply_accepted_assimilation(
            model,
            optimizer,
            pre_confirm.curriculum,
            train.features,
            train.targets,
            checkpoint.train_class_weights,
            multipliers,
            seed,
            assimilation_coordinates,
        )
    else:
        settle_rejected_proposal(model, optimizer, pre_confirm.baseline)
    lifecycle.complete_phase(PreTestPhase.ASSIMILATION_SETTLED)
    lifecycle.complete_phase(PreTestPhase.PRE_TEST_ARTIFACTS_COMMITTED)
    lifecycle.open_test()
    lifecycle.assert_opened()
    return score_model(
        ScoringRequest(model, test.features, test.targets, LocalClassCount(n_classes))
    )


def _action_sha256(action: CurriculumAction) -> Sha256Digest:
    return Sha256Digest(
        hashlib.sha256(
            ",".join(f"{value:.17g}" for value in action.coordinates).encode()
        ).hexdigest()
    )


def _score_local_sir_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    target: DatasetId,
    materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    checkpoint_path = _target_confirmatory_checkpoint_path(layout, target, seed)
    if not checkpoint_path.is_file():
        return None
    checkpoint_artifact_id = _checkpoint_artifact_id(store, checkpoint_path)
    if checkpoint_artifact_id is None:
        return None
    eligible_groups_by_coarse = _dataset_eligible_groups_by_coarse(target, materialized)
    if not eligible_groups_by_coarse:
        return None
    blocks = self_padded_blocks(eligible_groups_by_coarse)
    packets_by_coarse: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    for coarse_group in eligible_groups_by_coarse:
        packet = _load_dataset_source_packet(layout, target, seed, coarse_group)
        if packet is not None:
            packets_by_coarse[coarse_group] = packet
    if not packets_by_coarse:
        return None
    response_matrix = assemble_self_response_matrix(blocks, packets_by_coarse)
    checkpoint = load_base_checkpoint(checkpoint_path)
    n_classes = materialized.class_manifest.class_count
    train = materialized.splits[Split.TRAIN]
    meta = materialized.splits[Split.META]
    confirm = materialized.splits[Split.CONFIRM]
    test = materialized.splits[Split.TEST]
    model = create_classifier(
        target,
        train.features.shape[1],
        n_classes,
        checkpoint.selected_hyperparameters.dropout_probability,
        seed,
        device,
    )
    checkpoint.state_dict.load_into(model)
    meta_score = score_model(
        ScoringRequest(model, meta.features, meta.targets, LocalClassCount(n_classes))
    )
    meta_class_ce = tuple(
        entry.value for entry in meta_score.class_conditional_cross_entropy.values
    )
    node_risks = target_node_risks(blocks, eligible_groups_by_coarse, meta_class_ce)
    try:
        target_importance = build_target_importance(node_risks)
    except TargetImportanceError:
        return None
    actionable_nodes = tuple(risk.node_index for risk in node_risks if risk.is_actionable)
    problem = build_robust_action_problem(
        blocks,
        response_matrix,
        response_matrix,
        target_importance.as_vector(blocks.total_padded_nodes),
        actionable_nodes,
    )
    solution = local_sir_action(problem, response_matrix)
    action = solution.selected_action
    multipliers = curriculum_multipliers_from_action(
        action, blocks, eligible_groups_by_coarse, n_classes
    )
    optimizer = make_adamw(
        model,
        checkpoint.selected_hyperparameters.learning_rate,
        checkpoint.selected_hyperparameters.weight_decay,
    )
    input_artifact_ids: tuple[ArtifactIdentifier, ...] = (
        checkpoint_artifact_id,
        *(
            ArtifactIdentifier(packet.packet_integrity_sha256)
            for packet in packets_by_coarse.values()
        ),
    )
    first_packet = next(iter(packets_by_coarse.values()))
    score = _confirm_assimilate_and_score(
        model,
        optimizer,
        checkpoint,
        train,
        confirm,
        test,
        multipliers,
        seed,
        ContrastCoordinates(f"local-sir:{target.value}:{seed}"),
        AssimilationCoordinates(
            target_client=SourceClientName(target.value),
            directed_pair=DirectedPairName(f"{target.value} -> {target.value}"),
            condition=EvaluationConditionName("principal"),
            seed=seed,
            clean_pretransfer_checkpoint_artifact_id=checkpoint_artifact_id,
            source_packet_artifact_id=ArtifactIdentifier(first_packet.packet_integrity_sha256),
            action_artifact_sha256=_action_sha256(action),
        ),
        n_classes,
    )
    return score, n_classes, input_artifact_ids


def _common_eligible_groups(
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
) -> (
    tuple[
        Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]],
        Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]],
    ]
    | None
):
    target_eligible_all = _dataset_eligible_groups_by_coarse(target, target_materialized)
    source_eligible_all = _dataset_eligible_groups_by_coarse(source, source_materialized)
    common_coarse = tuple(group for group in target_eligible_all if group in source_eligible_all)
    if not common_coarse:
        return None
    source_eligible = OrderedDict((group, source_eligible_all[group]) for group in common_coarse)
    target_eligible = OrderedDict((group, target_eligible_all[group]) for group in common_coarse)
    return source_eligible, target_eligible


def cross_client_padded_blocks(
    source_eligible_by_coarse: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]],
    target_eligible_by_coarse: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]],
) -> PaddedBlockStructure:
    coarse_groups = tuple(
        group for group in target_eligible_by_coarse if group in source_eligible_by_coarse
    )
    source_counts = OrderedDict(
        (group, len(source_eligible_by_coarse[group])) for group in coarse_groups
    )
    target_counts = OrderedDict(
        (group, len(target_eligible_by_coarse[group])) for group in coarse_groups
    )
    return build_padded_block_structure(coarse_groups, source_counts, target_counts)


def assemble_cross_client_response_matrix(
    blocks: PaddedBlockStructure,
    source_packets_by_coarse: Mapping[CoarseGroup, SourcePacket],
    array_selector: Callable[[SourcePacket], ResponseMatrix],
) -> ResponseMatrix:
    size = blocks.total_padded_nodes
    matrix: ResponseMatrix = np.zeros((size, size), dtype=np.float64)
    for block_index, coarse_group in enumerate(blocks.coarse_groups):
        packet = source_packets_by_coarse.get(coarse_group)
        if packet is None:
            continue
        block_range = blocks.block_index_range(block_index)
        source_real = blocks.source_real_counts[block_index]
        submatrix = array_selector(packet)
        if submatrix.shape != (source_real, source_real):
            raise ExecutionError(
                f"source packet for {coarse_group.value} has shape {submatrix.shape}, "
                f"expected {(source_real, source_real)}"
            )
        start = block_range.start
        matrix[start : start + source_real, start : start + source_real] = submatrix
    return matrix


def _original_groups_for_bucket(
    common_coarse: tuple[CoarseGroup, ...],
    group_bucket_of: Mapping[CoarseGroup, CoarseGroup],
) -> OrderedDict[CoarseGroup, tuple[CoarseGroup, ...]]:
    by_bucket: OrderedDict[CoarseGroup, list[CoarseGroup]] = OrderedDict()
    for group in common_coarse:
        bucket = group_bucket_of.get(group, group)
        by_bucket.setdefault(bucket, []).append(group)
    return OrderedDict((bucket, tuple(groups)) for bucket, groups in by_bucket.items())


def _regroup_eligible_groups(
    eligible_by_coarse: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]],
    groups_by_bucket: Mapping[CoarseGroup, tuple[CoarseGroup, ...]],
) -> OrderedDict[CoarseGroup, tuple[TransferConceptGroup, ...]]:
    merged: OrderedDict[CoarseGroup, tuple[TransferConceptGroup, ...]] = OrderedDict()
    for bucket, originals in groups_by_bucket.items():
        combined: list[TransferConceptGroup] = []
        for original in originals:
            combined.extend(eligible_by_coarse[original])
        merged[bucket] = tuple(combined)
    return merged


def assemble_merged_cross_client_response_matrix(
    blocks: PaddedBlockStructure,
    groups_by_bucket: Mapping[CoarseGroup, tuple[CoarseGroup, ...]],
    source_eligible_original: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]],
    packets_by_original_group: Mapping[CoarseGroup, SourcePacket],
    array_selector: Callable[[SourcePacket], ResponseMatrix],
) -> ResponseMatrix:
    size = blocks.total_padded_nodes
    matrix: ResponseMatrix = np.zeros((size, size), dtype=np.float64)
    for block_index, bucket in enumerate(blocks.coarse_groups):
        block_range = blocks.block_index_range(block_index)
        offset = 0
        for original in groups_by_bucket[bucket]:
            original_source_real = len(source_eligible_original[original])
            packet = packets_by_original_group.get(original)
            if packet is not None:
                submatrix = array_selector(packet)
                if submatrix.shape != (original_source_real, original_source_real):
                    raise ExecutionError(
                        f"source packet for {original.value} has shape {submatrix.shape}, "
                        f"expected {(original_source_real, original_source_real)}"
                    )
                start = block_range.start + offset
                matrix[
                    start : start + original_source_real, start : start + original_source_real
                ] = submatrix
            offset += original_source_real
    return matrix


def assemble_target_response_matrix(
    blocks: PaddedBlockStructure,
    target_packets_by_coarse: Mapping[CoarseGroup, SourcePacket],
    array_selector: Callable[[SourcePacket], ResponseMatrix],
) -> ResponseMatrix:
    size = blocks.total_padded_nodes
    matrix: ResponseMatrix = np.zeros((size, size), dtype=np.float64)
    for block_index, coarse_group in enumerate(blocks.coarse_groups):
        packet = target_packets_by_coarse.get(coarse_group)
        if packet is None:
            continue
        block_range = blocks.block_index_range(block_index)
        target_real = blocks.target_real_counts[block_index]
        submatrix = array_selector(packet)
        if submatrix.shape != (target_real, target_real):
            raise ExecutionError(
                f"target packet for {coarse_group.value} has shape {submatrix.shape}, "
                f"expected {(target_real, target_real)}"
            )
        start = block_range.start
        matrix[start : start + target_real, start : start + target_real] = submatrix
    return matrix


def _score_matched_resource_rectangular_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    checkpoint_path = _target_confirmatory_checkpoint_path(layout, target, seed)
    if not checkpoint_path.is_file():
        return None
    checkpoint_artifact_id = _checkpoint_artifact_id(store, checkpoint_path)
    if checkpoint_artifact_id is None:
        return None
    common = _common_eligible_groups(source, target, source_materialized, target_materialized)
    if common is None:
        return None
    source_eligible, target_eligible = common
    common_coarse = tuple(target_eligible)
    blocks = cross_client_padded_blocks(source_eligible, target_eligible)
    packets_by_coarse: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    for coarse_group in common_coarse:
        packet = _load_dataset_source_packet(layout, source, seed, coarse_group)
        if packet is not None:
            packets_by_coarse[coarse_group] = packet
    if not packets_by_coarse:
        return None
    lower_matrix = assemble_cross_client_response_matrix(
        blocks, packets_by_coarse, SourcePacket.lower_matrix
    )
    upper_matrix = assemble_cross_client_response_matrix(
        blocks, packets_by_coarse, SourcePacket.upper_matrix
    )
    checkpoint = load_base_checkpoint(checkpoint_path)
    n_classes = target_materialized.class_manifest.class_count
    train = target_materialized.splits[Split.TRAIN]
    meta = target_materialized.splits[Split.META]
    confirm = target_materialized.splits[Split.CONFIRM]
    test = target_materialized.splits[Split.TEST]
    model = create_classifier(
        target,
        train.features.shape[1],
        n_classes,
        checkpoint.selected_hyperparameters.dropout_probability,
        seed,
        device,
    )
    checkpoint.state_dict.load_into(model)
    meta_score = score_model(
        ScoringRequest(model, meta.features, meta.targets, LocalClassCount(n_classes))
    )
    meta_class_ce = tuple(
        entry.value for entry in meta_score.class_conditional_cross_entropy.values
    )
    node_risks = target_node_risks(blocks, target_eligible, meta_class_ce)
    try:
        target_importance = build_target_importance(node_risks)
    except TargetImportanceError:
        return None
    actionable_nodes = tuple(risk.node_index for risk in node_risks if risk.is_actionable)
    problem = build_robust_action_problem(
        blocks,
        lower_matrix,
        upper_matrix,
        target_importance.as_vector(blocks.total_padded_nodes),
        actionable_nodes,
    )
    hull = build_rectangular_hull(blocks, lower_matrix, upper_matrix)
    solution = optimize_against_fixed_matrix(problem, hull.lower_bounds)
    action = solution.selected_action
    multipliers = curriculum_multipliers_from_action(action, blocks, target_eligible, n_classes)
    optimizer = make_adamw(
        model,
        checkpoint.selected_hyperparameters.learning_rate,
        checkpoint.selected_hyperparameters.weight_decay,
    )
    input_artifact_ids: tuple[ArtifactIdentifier, ...] = (
        checkpoint_artifact_id,
        *(
            ArtifactIdentifier(packet.packet_integrity_sha256)
            for packet in packets_by_coarse.values()
        ),
    )
    first_packet = next(iter(packets_by_coarse.values()))
    score = _confirm_assimilate_and_score(
        model,
        optimizer,
        checkpoint,
        train,
        confirm,
        test,
        multipliers,
        seed,
        ContrastCoordinates(
            f"matched-resource-rectangular:{source.value}-to-{target.value}:{seed}"
        ),
        AssimilationCoordinates(
            target_client=SourceClientName(target.value),
            directed_pair=DirectedPairName(f"{source.value} -> {target.value}"),
            condition=EvaluationConditionName("principal"),
            seed=seed,
            clean_pretransfer_checkpoint_artifact_id=checkpoint_artifact_id,
            source_packet_artifact_id=ArtifactIdentifier(first_packet.packet_integrity_sha256),
            action_artifact_sha256=_action_sha256(action),
        ),
        n_classes,
    )
    return score, n_classes, input_artifact_ids


def _score_point_correspondence_commitment_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    checkpoint_path = _target_confirmatory_checkpoint_path(layout, target, seed)
    if not checkpoint_path.is_file():
        return None
    checkpoint_artifact_id = _checkpoint_artifact_id(store, checkpoint_path)
    if checkpoint_artifact_id is None:
        return None
    common = _common_eligible_groups(source, target, source_materialized, target_materialized)
    if common is None:
        return None
    source_eligible, target_eligible = common
    common_coarse = tuple(target_eligible)
    blocks = cross_client_padded_blocks(source_eligible, target_eligible)
    source_packets: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    target_packets: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    for coarse_group in common_coarse:
        source_packet = _load_dataset_source_packet(layout, source, seed, coarse_group)
        if source_packet is not None:
            source_packets[coarse_group] = source_packet
        target_packet = _load_dataset_source_packet(layout, target, seed, coarse_group)
        if target_packet is not None:
            target_packets[coarse_group] = target_packet
    if not source_packets or not target_packets:
        return None
    source_matrix = assemble_cross_client_response_matrix(
        blocks, source_packets, SourcePacket.lower_matrix
    )
    target_matrix = assemble_target_response_matrix(
        blocks, target_packets, SourcePacket.lower_matrix
    )
    qap_result = point_correspondence_commitment(source_matrix, target_matrix, blocks)
    if not qap_result.certified or qap_result.correspondence is None:
        return None
    committed_matrix = qap_result.correspondence.permute_response_matrix(source_matrix)
    checkpoint = load_base_checkpoint(checkpoint_path)
    n_classes = target_materialized.class_manifest.class_count
    train = target_materialized.splits[Split.TRAIN]
    meta = target_materialized.splits[Split.META]
    confirm = target_materialized.splits[Split.CONFIRM]
    test = target_materialized.splits[Split.TEST]
    model = create_classifier(
        target,
        train.features.shape[1],
        n_classes,
        checkpoint.selected_hyperparameters.dropout_probability,
        seed,
        device,
    )
    checkpoint.state_dict.load_into(model)
    meta_score = score_model(
        ScoringRequest(model, meta.features, meta.targets, LocalClassCount(n_classes))
    )
    meta_class_ce = tuple(
        entry.value for entry in meta_score.class_conditional_cross_entropy.values
    )
    node_risks = target_node_risks(blocks, target_eligible, meta_class_ce)
    try:
        target_importance = build_target_importance(node_risks)
    except TargetImportanceError:
        return None
    actionable_nodes = tuple(risk.node_index for risk in node_risks if risk.is_actionable)
    problem = build_robust_action_problem(
        blocks,
        committed_matrix,
        committed_matrix,
        target_importance.as_vector(blocks.total_padded_nodes),
        actionable_nodes,
    )
    solution = optimize_against_fixed_matrix(problem, committed_matrix)
    action = solution.selected_action
    multipliers = curriculum_multipliers_from_action(action, blocks, target_eligible, n_classes)
    optimizer = make_adamw(
        model,
        checkpoint.selected_hyperparameters.learning_rate,
        checkpoint.selected_hyperparameters.weight_decay,
    )
    input_artifact_ids: tuple[ArtifactIdentifier, ...] = (
        checkpoint_artifact_id,
        *(
            ArtifactIdentifier(packet.packet_integrity_sha256)
            for packet in (*source_packets.values(), *target_packets.values())
        ),
    )
    first_source_packet = next(iter(source_packets.values()))
    score = _confirm_assimilate_and_score(
        model,
        optimizer,
        checkpoint,
        train,
        confirm,
        test,
        multipliers,
        seed,
        ContrastCoordinates(
            f"point-correspondence-commitment:{source.value}-to-{target.value}:{seed}"
        ),
        AssimilationCoordinates(
            target_client=SourceClientName(target.value),
            directed_pair=DirectedPairName(f"{source.value} -> {target.value}"),
            condition=EvaluationConditionName("principal"),
            seed=seed,
            clean_pretransfer_checkpoint_artifact_id=checkpoint_artifact_id,
            source_packet_artifact_id=ArtifactIdentifier(
                first_source_packet.packet_integrity_sha256
            ),
            action_artifact_sha256=_action_sha256(action),
        ),
        n_classes,
    )
    return score, n_classes, input_artifact_ids


@dataclass(frozen=True, slots=True)
class PrincipalActionAssembly:
    problem: RobustActionProblem
    blocks: PaddedBlockStructure
    action: CurriculumAction
    checkpoint: BaseCheckpoint
    checkpoint_artifact_id: ArtifactIdentifier
    model: torch.nn.Module
    train: SplitTensors
    confirm: SplitTensors
    test: SplitTensors
    n_classes: ClassCount
    target_eligible: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]]
    input_artifact_ids: tuple[ArtifactIdentifier, ...]
    first_packet_artifact_id: ArtifactIdentifier


def _assemble_principal_action(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
    solve_action: Callable[[RobustActionProblem, RandomSeed], CurriculumAction | None],
    group_bucket_of: Mapping[CoarseGroup, CoarseGroup] | None = None,
    perturb: Callable[
        [PaddedBlockStructure, ResponseMatrix, ResponseMatrix],
        tuple[ResponseMatrix, ResponseMatrix],
    ]
    | None = None,
    checkpoint_source_experiment: ExperimentName = ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT,
) -> PrincipalActionAssembly | None:
    checkpoint_path = _target_confirmatory_checkpoint_path(
        layout, target, seed, checkpoint_source_experiment
    )
    if not checkpoint_path.is_file():
        return None
    checkpoint_artifact_id = _checkpoint_artifact_id(store, checkpoint_path)
    if checkpoint_artifact_id is None:
        return None
    common = _common_eligible_groups(source, target, source_materialized, target_materialized)
    if common is None:
        return None
    source_eligible_original, target_eligible_original = common
    common_coarse = tuple(target_eligible_original)
    packets_by_coarse: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    for coarse_group in common_coarse:
        packet = _load_dataset_source_packet(layout, source, seed, coarse_group)
        if packet is not None:
            packets_by_coarse[coarse_group] = packet
    if not packets_by_coarse:
        return None
    if group_bucket_of is None:
        source_eligible = source_eligible_original
        target_eligible = target_eligible_original
        groups_by_bucket: Mapping[CoarseGroup, tuple[CoarseGroup, ...]] = OrderedDict(
            (group, (group,)) for group in common_coarse
        )
    else:
        groups_by_bucket = _original_groups_for_bucket(common_coarse, group_bucket_of)
        source_eligible = _regroup_eligible_groups(source_eligible_original, groups_by_bucket)
        target_eligible = _regroup_eligible_groups(target_eligible_original, groups_by_bucket)
    blocks = cross_client_padded_blocks(source_eligible, target_eligible)
    lower_matrix = assemble_merged_cross_client_response_matrix(
        blocks,
        groups_by_bucket,
        source_eligible_original,
        packets_by_coarse,
        SourcePacket.lower_matrix,
    )
    upper_matrix = assemble_merged_cross_client_response_matrix(
        blocks,
        groups_by_bucket,
        source_eligible_original,
        packets_by_coarse,
        SourcePacket.upper_matrix,
    )
    if perturb is not None:
        lower_matrix, upper_matrix = perturb(blocks, lower_matrix, upper_matrix)
    checkpoint = load_base_checkpoint(checkpoint_path)
    n_classes = target_materialized.class_manifest.class_count
    train = target_materialized.splits[Split.TRAIN]
    meta = target_materialized.splits[Split.META]
    confirm = target_materialized.splits[Split.CONFIRM]
    test = target_materialized.splits[Split.TEST]
    model = create_classifier(
        target,
        train.features.shape[1],
        n_classes,
        checkpoint.selected_hyperparameters.dropout_probability,
        seed,
        device,
    )
    checkpoint.state_dict.load_into(model)
    meta_score = score_model(
        ScoringRequest(model, meta.features, meta.targets, LocalClassCount(n_classes))
    )
    meta_class_ce = tuple(
        entry.value for entry in meta_score.class_conditional_cross_entropy.values
    )
    node_risks = target_node_risks(blocks, target_eligible, meta_class_ce)
    try:
        target_importance = build_target_importance(node_risks)
    except TargetImportanceError:
        return None
    actionable_nodes = tuple(risk.node_index for risk in node_risks if risk.is_actionable)
    problem = build_robust_action_problem(
        blocks,
        lower_matrix,
        upper_matrix,
        target_importance.as_vector(blocks.total_padded_nodes),
        actionable_nodes,
    )
    action = solve_action(problem, seed)
    if action is None:
        return None
    input_artifact_ids: tuple[ArtifactIdentifier, ...] = (
        checkpoint_artifact_id,
        *(
            ArtifactIdentifier(packet.packet_integrity_sha256)
            for packet in packets_by_coarse.values()
        ),
    )
    first_packet = next(iter(packets_by_coarse.values()))
    return PrincipalActionAssembly(
        problem=problem,
        blocks=blocks,
        action=action,
        checkpoint=checkpoint,
        checkpoint_artifact_id=checkpoint_artifact_id,
        model=model,
        train=train,
        confirm=confirm,
        test=test,
        n_classes=n_classes,
        target_eligible=target_eligible,
        input_artifact_ids=input_artifact_ids,
        first_packet_artifact_id=ArtifactIdentifier(first_packet.packet_integrity_sha256),
    )


def _score_robust_action_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
    method_slug: str,
    solve_action: Callable[[RobustActionProblem, RandomSeed], CurriculumAction | None],
    settle_and_score: Callable[
        [
            torch.nn.Module,
            torch.optim.AdamW,
            BaseCheckpoint,
            SplitTensors,
            SplitTensors,
            SplitTensors,
            CurriculumMultipliers,
            CurriculumAction,
            RandomSeed,
            ContrastCoordinates,
            AssimilationCoordinates,
            ClassCount,
        ],
        ScoreArtifact,
    ]
    | None = None,
    group_bucket_of: Mapping[CoarseGroup, CoarseGroup] | None = None,
    perturb: Callable[
        [PaddedBlockStructure, ResponseMatrix, ResponseMatrix],
        tuple[ResponseMatrix, ResponseMatrix],
    ]
    | None = None,
    checkpoint_source_experiment: ExperimentName = ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    assembly = _assemble_principal_action(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        solve_action,
        group_bucket_of,
        perturb,
        checkpoint_source_experiment,
    )
    if assembly is None:
        return None
    multipliers = curriculum_multipliers_from_action(
        assembly.action, assembly.blocks, assembly.target_eligible, assembly.n_classes
    )
    optimizer = make_adamw(
        assembly.model,
        assembly.checkpoint.selected_hyperparameters.learning_rate,
        assembly.checkpoint.selected_hyperparameters.weight_decay,
    )
    contrast_coordinates = ContrastCoordinates(
        f"{method_slug}:{source.value}-to-{target.value}:{seed}"
    )
    assimilation_coordinates = AssimilationCoordinates(
        target_client=SourceClientName(target.value),
        directed_pair=DirectedPairName(f"{source.value} -> {target.value}"),
        condition=EvaluationConditionName("principal"),
        seed=seed,
        clean_pretransfer_checkpoint_artifact_id=assembly.checkpoint_artifact_id,
        source_packet_artifact_id=assembly.first_packet_artifact_id,
        action_artifact_sha256=_action_sha256(assembly.action),
    )
    if settle_and_score is None:
        score = _confirm_assimilate_and_score(
            assembly.model,
            optimizer,
            assembly.checkpoint,
            assembly.train,
            assembly.confirm,
            assembly.test,
            multipliers,
            seed,
            contrast_coordinates,
            assimilation_coordinates,
            assembly.n_classes,
        )
    else:
        score = settle_and_score(
            assembly.model,
            optimizer,
            assembly.checkpoint,
            assembly.train,
            assembly.confirm,
            assembly.test,
            multipliers,
            assembly.action,
            seed,
            contrast_coordinates,
            assimilation_coordinates,
            assembly.n_classes,
        )
    return score, assembly.n_classes, assembly.input_artifact_ids


def _solve_fedorbit_exact_sparse_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    del seed
    solution = solve_robust_action(problem)
    return solution.selected_action


def _solve_generic_exact_qap_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    del seed
    outcome = solve_robust_action_qap(problem)
    if outcome.certified_solution is None:
        return None
    return outcome.certified_solution.certified_action


def _solve_exact_map_oracle_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    del seed
    identity = BlockCorrespondence.lexicographically_smallest(problem.blocks)
    committed_matrix = identity.permute_response_matrix(problem.lower_response_matrix)
    return optimize_against_fixed_matrix(problem, committed_matrix).selected_action


def _score_fedorbit_exact_sparse_solver_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return _score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        "fedorbit-exact-sparse-solver",
        _solve_fedorbit_exact_sparse_action,
    )


def _settle_without_confirmation_and_score(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    checkpoint: BaseCheckpoint,
    train: SplitTensors,
    confirm: SplitTensors,
    test: SplitTensors,
    multipliers: CurriculumMultipliers,
    action: CurriculumAction,
    seed: RandomSeed,
    contrast_coordinates: ContrastCoordinates,
    assimilation_coordinates: AssimilationCoordinates,
    n_classes: ClassCount,
) -> ScoreArtifact:
    del confirm, contrast_coordinates
    pre_confirm = capture_pre_confirm_pair(model, optimizer)
    if action.realized_support_size > 0:
        apply_accepted_assimilation(
            model,
            optimizer,
            pre_confirm.curriculum,
            train.features,
            train.targets,
            checkpoint.train_class_weights,
            multipliers,
            seed,
            assimilation_coordinates,
        )
    else:
        settle_rejected_proposal(model, optimizer, pre_confirm.baseline)
    return score_model(
        ScoringRequest(model, test.features, test.targets, LocalClassCount(n_classes))
    )


def _score_fedorbit_without_confirmation_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return _score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        "fedorbit-without-confirmation",
        _solve_fedorbit_exact_sparse_action,
        _settle_without_confirmation_and_score,
    )


def _score_generic_exact_qap_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return _score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        "generic-exact-qap",
        _solve_generic_exact_qap_action,
    )


def _score_exact_map_oracle_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return _score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        "exact-map-oracle",
        _solve_exact_map_oracle_action,
    )


def solve_coarse_block_mean_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    del seed
    summary = coarse_block_mean_matrix(problem.blocks, problem.lower_response_matrix)
    return optimize_against_fixed_matrix(problem, summary.matrix).selected_action


def solve_coarse_block_min_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    del seed
    summary = coarse_block_min_matrix(problem.blocks, problem.lower_response_matrix)
    return optimize_against_fixed_matrix(problem, summary.matrix).selected_action


def solve_orbit_mean_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    del seed
    summary = orbit_mean_matrix(problem.blocks, problem.lower_response_matrix)
    return optimize_against_fixed_matrix(problem, summary.matrix).selected_action


def solve_coupling_destroyed_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    destroyed = coupling_destroyed_matrices(
        problem.blocks,
        problem.lower_response_matrix,
        problem.upper_response_matrix,
        seed,
        ContrastCoordinates(f"coupling-destroyed:{seed}"),
    )
    destroyed_problem = RobustActionProblem(
        blocks=problem.blocks,
        lower_response_matrix=destroyed.lower_response_matrix,
        upper_response_matrix=destroyed.upper_response_matrix,
        target_importance=problem.target_importance,
        coordinate_caps=problem.coordinate_caps,
        linear_costs=problem.linear_costs,
        total_budget=problem.total_budget,
        principal_support=problem.principal_support,
    )
    return solve_robust_action(destroyed_problem).selected_action


def _score_coarse_block_mean_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return _score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        "coarse-block-mean",
        solve_coarse_block_mean_action,
    )


def _score_coarse_block_min_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return _score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        "coarse-block-min",
        solve_coarse_block_min_action,
    )


def _score_orbit_mean_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return _score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        "orbit-mean",
        solve_orbit_mean_action,
    )


def _score_coupling_destroyed_fedorbit_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return _score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        "coupling-destroyed-fedorbit",
        solve_coupling_destroyed_action,
    )


def _score_local_sir_cell_adapter(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    del source, source_materialized
    return _score_local_sir_cell(store, layout, target, target_materialized, seed, device)


def execute_mechanism_ablations(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = materialize_client(dataset, raw_root)
        return materialized_by_dataset.get(dataset)

    scorers: tuple[
        tuple[
            TransferMethod,
            Callable[
                [
                    ArtifactStore,
                    WorkspaceLayout,
                    DatasetId,
                    DatasetId,
                    MaterializedClient,
                    MaterializedClient,
                    RandomSeed,
                    torch.device,
                ],
                tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None,
            ],
        ],
        ...,
    ] = (
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, _score_fedorbit_exact_sparse_solver_cell),
        (TransferMethod.MATCHED_RESOURCE_RECTANGULAR, _score_matched_resource_rectangular_cell),
        (
            TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
            _score_point_correspondence_commitment_cell,
        ),
        (TransferMethod.COUPLING_DESTROYED_FEDORBIT, _score_coupling_destroyed_fedorbit_cell),
        (TransferMethod.COARSE_BLOCK_MEAN, _score_coarse_block_mean_cell),
        (TransferMethod.COARSE_BLOCK_MIN, _score_coarse_block_min_cell),
        (TransferMethod.ORBIT_MEAN, _score_orbit_mean_cell),
        (TransferMethod.LOCAL_SIR, _score_local_sir_cell_adapter),
    )
    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        for seed in confirmatory_seeds:
            for method, scorer in scorers:
                scored = scorer(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    target_materialized,
                    seed,
                    device,
                )
                if scored is None:
                    continue
                score, n_classes, input_artifact_ids = scored
                _persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    method,
                    seed,
                    score,
                    n_classes,
                    input_artifact_ids,
                )


def _confirm_assimilate_score_capturing_verdict(
    verdicts: list[ConfirmationVerdict],
) -> Callable[
    [
        torch.nn.Module,
        torch.optim.AdamW,
        BaseCheckpoint,
        SplitTensors,
        SplitTensors,
        SplitTensors,
        CurriculumMultipliers,
        CurriculumAction,
        RandomSeed,
        ContrastCoordinates,
        AssimilationCoordinates,
        ClassCount,
    ],
    ScoreArtifact,
]:
    def settle_and_score(
        model: torch.nn.Module,
        optimizer: torch.optim.AdamW,
        checkpoint: BaseCheckpoint,
        train: SplitTensors,
        confirm: SplitTensors,
        test: SplitTensors,
        multipliers: CurriculumMultipliers,
        action: CurriculumAction,
        seed: RandomSeed,
        contrast_coordinates: ContrastCoordinates,
        assimilation_coordinates: AssimilationCoordinates,
        n_classes: ClassCount,
    ) -> ScoreArtifact:
        del action
        pre_confirm = capture_pre_confirm_pair(model, optimizer)
        verdict = run_proposal_confirmation(
            ConfirmationRequest(
                model,
                pre_confirm.baseline,
                pre_confirm.curriculum,
                train.features,
                train.targets,
                confirm.features,
                confirm.targets,
                checkpoint.train_class_weights,
                multipliers,
                checkpoint.selected_hyperparameters,
                seed,
                contrast_coordinates,
            )
        )
        verdicts.append(verdict)
        lifecycle = PreTestLifecycle()
        lifecycle.complete_phase(PreTestPhase.SOURCE_SELECTION_FINALIZED)
        lifecycle.complete_phase(PreTestPhase.ACTION_FINALIZED)
        lifecycle.complete_phase(PreTestPhase.CONFIRMATION_DECISION_FINALIZED)
        if verdict.accepted:
            apply_accepted_assimilation(
                model,
                optimizer,
                pre_confirm.curriculum,
                train.features,
                train.targets,
                checkpoint.train_class_weights,
                multipliers,
                seed,
                assimilation_coordinates,
            )
        else:
            settle_rejected_proposal(model, optimizer, pre_confirm.baseline)
        lifecycle.complete_phase(PreTestPhase.ASSIMILATION_SETTLED)
        lifecycle.complete_phase(PreTestPhase.PRE_TEST_ARTIFACTS_COMMITTED)
        lifecycle.open_test()
        lifecycle.assert_opened()
        return score_model(
            ScoringRequest(model, test.features, test.targets, LocalClassCount(n_classes))
        )

    return settle_and_score


def _score_fedorbit_with_confirmation_verdict_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
    verdicts: list[ConfirmationVerdict],
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return _score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        "fedorbit-exact-sparse-solver",
        _solve_fedorbit_exact_sparse_action,
        _confirm_assimilate_score_capturing_verdict(verdicts),
    )


def execute_target_confirmation_and_portability(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    directed_pairs = (
        *active_config().scientific.datasets.primary_directed_pairs,
        *active_config().scientific.datasets.secondary_directed_pairs,
    )
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = materialize_client(dataset, raw_root)
        return materialized_by_dataset.get(dataset)

    for directed_pair in directed_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        for seed in confirmatory_seeds:
            verdicts: list[ConfirmationVerdict] = []
            with_confirmation = _score_fedorbit_with_confirmation_verdict_cell(
                store,
                layout,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                device,
                verdicts,
            )
            if with_confirmation is not None:
                score, n_classes, input_artifact_ids = with_confirmation
                _persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                    seed,
                    score,
                    n_classes,
                    input_artifact_ids,
                )
                for verdict in verdicts:
                    persist_primary_transfer_metric(
                        store,
                        layout,
                        request.experiment,
                        pair_direction,
                        source,
                        target,
                        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                        seed,
                        MetricId.PROPOSAL_ACCEPTANCE_RATE,
                        1.0 if verdict.accepted else 0.0,
                        MetricUnit("fraction"),
                        MetricDirection.DESCRIPTIVE,
                        input_artifact_ids,
                        request.overwrite_policy,
                    )
            without_confirmation = _score_fedorbit_without_confirmation_cell(
                store,
                layout,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                device,
            )
            if without_confirmation is not None:
                score, n_classes, input_artifact_ids = without_confirmation
                _persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    TransferMethod.FEDORBIT_WITHOUT_CONFIRMATION,
                    seed,
                    score,
                    n_classes,
                    input_artifact_ids,
                )


def _score_local_only_cell_adapter(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    del source, source_materialized
    scored = _score_local_only_cell(store, layout, target, target_materialized, seed, device)
    if scored is None:
        return None
    score, n_classes, artifact_id = scored
    return score, n_classes, (artifact_id,)


def execute_secondary_cross_modality_generalization(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    secondary_pairs = active_config().scientific.datasets.secondary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = materialize_client(dataset, raw_root)
        return materialized_by_dataset.get(dataset)

    scorers = (
        (TransferMethod.LOCAL_ONLY, _score_local_only_cell_adapter),
        (TransferMethod.LOCAL_SIR, _score_local_sir_cell_adapter),
        (TransferMethod.MATCHED_RESOURCE_RECTANGULAR, _score_matched_resource_rectangular_cell),
        (
            TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
            _score_point_correspondence_commitment_cell,
        ),
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, _score_fedorbit_exact_sparse_solver_cell),
    )
    for directed_pair in secondary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        for seed in confirmatory_seeds:
            for method, scorer in scorers:
                scored = scorer(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    target_materialized,
                    seed,
                    device,
                )
                if scored is None:
                    continue
                score, n_classes, input_artifact_ids = scored
                _persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    method,
                    seed,
                    score,
                    n_classes,
                    input_artifact_ids,
                )


def persist_synthetic_benchmark_metric(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportCount,
    method: TransferMethod,
    seed: RandomSeed,
    metric_name: MetricId,
    metric_value: float,
    metric_unit: MetricUnit,
    direction: MetricDirection,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        method=method,
        condition=ExperimentCondition(condition),
        support=SupportSize(support),
        seed=ExperimentSeed(seed),
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.EVALUATION,
            cell,
            relevance,
            tuple(identifier.value for identifier in input_artifact_ids),
            _THEOREM_VALIDATION_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    metric = MetricRecord(
        experiment=experiment,
        pair=DirectedPairName("synthetic"),
        method=method,
        condition=condition,
        seed=seed,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        direction=direction,
        evaluation_class_set_sha256=Sha256Digest(
            hashlib.sha256(coordinates.encode("utf-8")).hexdigest()
        ),
        input_artifact_ids=tuple(input_artifact_ids),
        dependency_fingerprint_sha256=fingerprint,
        valid=True,
        invalid_reason=None,
    )
    validate_metric_records(MetricRecordCollection((metric,)))
    payload_path = (
        experiment_workspace(layout, experiment)
        / "artifacts"
        / "derived"
        / f"metric.{condition}.support-{support}.{method.value}.{seed}.{metric_name.value}.json"
    )
    payload = cast(StableJsonPayload, OrderedDict(metric_record=metric.model_dump(mode="json")))
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_THEOREM_VALIDATION_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(ArtifactStage.EVALUATION).sha256)
    completion = _completion(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=ArtifactStage.EVALUATION,
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.PREDICTION.value), payload, Sha256Digest(fingerprint)
            ),
            artifact_type=ArtifactType.PREDICTION,
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


def _synthetic_solver_instance(
    node_count: ConceptCount,
    block_pattern: ScalabilityBlockPattern,
    support: SupportCount,
    seed: RandomSeed,
) -> tuple[RobustActionProblem, CurriculumAction, PaddedBlockStructure]:
    instance = generate_scalability_instance(
        ScalabilityInstanceRequest(node_count, block_pattern, support, seed)
    )
    groups = tuple(CoarseGroup)[:2]
    counts = OrderedDict(zip(groups, instance.block_pattern, strict=True))
    blocks = build_padded_block_structure(groups, counts, counts)
    problem = build_robust_action_problem(
        blocks,
        instance.lower_response_matrix,
        instance.lower_response_matrix,
        instance.target_importance,
        tuple(range(support)),
    )
    action = CurriculumAction(problem=problem, coordinates=instance.fixed_action)
    return problem, action, blocks


def _solver_benchmark_reference_truth(
    blocks: PaddedBlockStructure,
    action: CurriculumAction,
    exhaustive_truth_correspondence_count_maximum: StepCount,
) -> Score | None:
    if blocks.orbit_size > exhaustive_truth_correspondence_count_maximum:
        return None
    return min(
        evaluate_objective(action, correspondence)
        for correspondence in enumerate_block_permutations(blocks)
    )


def _persist_solver_benchmark_error_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportCount,
    method: TransferMethod,
    seed: RandomSeed,
    reference_truth: Score | None,
    reported_objective: Score,
    exact_validation_absolute_tolerance: Tolerance,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> None:
    if reference_truth is None:
        return
    absolute_error = abs(reported_objective - reference_truth)
    persist_synthetic_benchmark_metric(
        store,
        layout,
        experiment,
        condition,
        support,
        method,
        seed,
        MetricId.ABSOLUTE_OBJECTIVE_ERROR,
        float(absolute_error),
        MetricUnit("score"),
        MetricDirection.LOWER_IS_BETTER,
        input_artifact_ids,
        overwrite_policy,
    )
    relative_error = absolute_error / abs(reference_truth) if reference_truth != 0.0 else 0.0
    persist_synthetic_benchmark_metric(
        store,
        layout,
        experiment,
        condition,
        support,
        method,
        seed,
        MetricId.RELATIVE_OBJECTIVE_ERROR,
        float(relative_error),
        MetricUnit("fraction"),
        MetricDirection.LOWER_IS_BETTER,
        input_artifact_ids,
        overwrite_policy,
    )
    persist_synthetic_benchmark_metric(
        store,
        layout,
        experiment,
        condition,
        support,
        method,
        seed,
        MetricId.CORRESPONDENCE_CERTIFICATE_VALIDITY,
        1.0
        if verify_exactness_certificate(
            reported_objective, reference_truth, exact_validation_absolute_tolerance
        )
        else 0.0,
        MetricUnit("boolean"),
        MetricDirection.HIGHER_IS_BETTER,
        input_artifact_ids,
        overwrite_policy,
    )


def _persist_solver_benchmark_efficiency_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportCount,
    method: TransferMethod,
    seed: RandomSeed,
    measurement: EfficiencyMeasurement,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> None:
    for metric_name, metric_value, metric_unit in (
        (MetricId.WALL_TIME, measurement.wall_time_seconds, MetricUnit("seconds")),
        (MetricId.PEAK_HOST_RSS, measurement.peak_host_rss_mib, MetricUnit("mib")),
        (
            MetricId.PEAK_CUDA_ALLOCATED_BYTES,
            float(measurement.peak_cuda_allocated_bytes),
            MetricUnit("bytes"),
        ),
    ):
        persist_synthetic_benchmark_metric(
            store,
            layout,
            experiment,
            condition,
            support,
            method,
            seed,
            metric_name,
            float(metric_value),
            metric_unit,
            MetricDirection.LOWER_IS_BETTER,
            input_artifact_ids,
            overwrite_policy,
        )


def _score_exact_sparse_solver_benchmark_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportCount,
    seed: RandomSeed,
    problem: RobustActionProblem,
    action: CurriculumAction,
    reference_truth: Score | None,
    methods: tuple[MethodName, ...],
    overwrite_policy: OverwritePolicy,
) -> None:
    input_artifact_ids = (ArtifactIdentifier("synthetic-generator"),)
    solver_config = active_config().solvers.exact_sparse
    if TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER in methods:
        with measure_efficiency() as efficiency:
            outcome = fixed_action_worst_correspondence(
                problem,
                action,
                solver_config.lap_objective_tie_tolerance,
                solver_config.action_tie_tolerance,
            )
        _persist_solver_benchmark_efficiency_metrics(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            seed,
            efficiency.result,
            input_artifact_ids,
            overwrite_policy,
        )
        _persist_solver_benchmark_error_metrics(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            seed,
            reference_truth,
            outcome.separator_objective,
            solver_config.exact_validation_absolute_tolerance,
            input_artifact_ids,
            overwrite_policy,
        )
        persist_synthetic_benchmark_metric(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            seed,
            MetricId.ACTIVE_IMAGE_CANDIDATES,
            float(outcome.active_image_candidates),
            MetricUnit("count"),
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            overwrite_policy,
        )
        persist_synthetic_benchmark_metric(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            seed,
            MetricId.LAP_CALLS,
            float(outcome.lap_calls),
            MetricUnit("count"),
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            overwrite_policy,
        )
    if TransferMethod.GENERIC_EXACT_QAP in methods:
        with measure_efficiency() as efficiency:
            qap_result = fixed_action_worst_correspondence_qap(problem, action)
        _persist_solver_benchmark_efficiency_metrics(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.GENERIC_EXACT_QAP,
            seed,
            efficiency.result,
            input_artifact_ids,
            overwrite_policy,
        )
        persist_synthetic_benchmark_metric(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.GENERIC_EXACT_QAP,
            seed,
            MetricId.TIMEOUT_INDICATOR,
            1.0 if qap_result.terminal_state == TerminalState.TIME_LIMIT else 0.0,
            MetricUnit("boolean"),
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            overwrite_policy,
        )
        if qap_result.certified and qap_result.objective_value is not None:
            _persist_solver_benchmark_error_metrics(
                store,
                layout,
                experiment,
                condition,
                support,
                TransferMethod.GENERIC_EXACT_QAP,
                seed,
                reference_truth,
                qap_result.objective_value,
                solver_config.exact_validation_absolute_tolerance,
                input_artifact_ids,
                overwrite_policy,
            )
    if TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK in methods:
        with measure_efficiency() as efficiency:
            dense_outcome = solve_dense_ccp(
                problem,
                seed,
                SemanticCoordinates(f"{experiment.value}:{condition}:{support}:{seed}"),
            )
        _persist_solver_benchmark_efficiency_metrics(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
            seed,
            efficiency.result,
            input_artifact_ids,
            overwrite_policy,
        )
        for metric_name, metric_value in (
            (MetricId.DENSE_RELAXATION_BOUND, dense_outcome.relaxation_lower_bound),
            (MetricId.DENSE_PROJECTED_OBJECTIVE, dense_outcome.best_projected_response_objective),
            (MetricId.DENSE_BOUND_GAP, dense_outcome.dense_bound_gap),
            (MetricId.DENSE_INTEGRALITY_RESIDUAL, dense_outcome.integrality_residual),
        ):
            persist_synthetic_benchmark_metric(
                store,
                layout,
                experiment,
                condition,
                support,
                TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
                seed,
                metric_name,
                float(metric_value),
                MetricUnit("score"),
                MetricDirection.DESCRIPTIVE,
                input_artifact_ids,
                overwrite_policy,
            )
        _persist_solver_benchmark_error_metrics(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
            seed,
            reference_truth,
            dense_outcome.best_projected_response_objective,
            solver_config.exact_validation_absolute_tolerance,
            input_artifact_ids,
            overwrite_policy,
        )


def execute_exact_sparse_solver_benchmark(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    config = active_config().experiments.exact_sparse_solver_benchmark
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    k_values: tuple[ConceptCount, ...] = tuple(
        range(config.synthetic_k.minimum, config.synthetic_k.maximum + 1)
    )
    for node_count in k_values:
        for pattern in config.block_patterns:
            for support in config.supports:
                condition = EvaluationConditionName(f"k{node_count}-{pattern.value}")
                for seed in confirmatory_seeds:
                    try:
                        problem, action, blocks = _synthetic_solver_instance(
                            node_count, pattern, support, seed
                        )
                    except ScalabilityGenerationError:
                        continue
                    reference_truth = _solver_benchmark_reference_truth(
                        blocks, action, config.exhaustive_truth_correspondence_count_maximum
                    )
                    _score_exact_sparse_solver_benchmark_cell(
                        store,
                        layout,
                        request.experiment,
                        condition,
                        support,
                        seed,
                        problem,
                        action,
                        reference_truth,
                        config.methods,
                        request.overwrite_policy,
                    )


def execute_scalability_and_efficiency(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    config = active_config().experiments.scalability_and_efficiency
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    methods = (
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.GENERIC_EXACT_QAP,
    )
    for node_count in config.k_values:
        for pattern in config.block_patterns:
            for support in config.exact_qap_supports:
                condition = EvaluationConditionName(f"k{node_count}-{pattern.value}")
                for seed in confirmatory_seeds:
                    try:
                        problem, action, _ = _synthetic_solver_instance(
                            node_count, pattern, support, seed
                        )
                    except ScalabilityGenerationError:
                        continue
                    _score_exact_sparse_solver_benchmark_cell(
                        store,
                        layout,
                        request.experiment,
                        condition,
                        support,
                        seed,
                        problem,
                        action,
                        None,
                        methods,
                        request.overwrite_policy,
                    )


def persist_synthetic_diagnostic_metric(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    seed: RandomSeed,
    metric_name: MetricId,
    metric_value: float,
    metric_unit: MetricUnit,
    direction: MetricDirection,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        condition=ExperimentCondition(condition),
        seed=ExperimentSeed(seed),
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.EVALUATION,
            cell,
            relevance,
            tuple(identifier.value for identifier in input_artifact_ids),
            _THEOREM_VALIDATION_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    metric = MetricRecord(
        experiment=experiment,
        pair=DirectedPairName("synthetic"),
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        condition=condition,
        seed=seed,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        direction=direction,
        evaluation_class_set_sha256=Sha256Digest(
            hashlib.sha256(coordinates.encode("utf-8")).hexdigest()
        ),
        input_artifact_ids=tuple(input_artifact_ids),
        dependency_fingerprint_sha256=fingerprint,
        valid=True,
        invalid_reason=None,
    )
    validate_metric_records(MetricRecordCollection((metric,)))
    payload_path = (
        experiment_workspace(layout, experiment)
        / "artifacts"
        / "derived"
        / f"metric.{condition}.{seed}.{metric_name.value}.json"
    )
    payload = cast(StableJsonPayload, OrderedDict(metric_record=metric.model_dump(mode="json")))
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_THEOREM_VALIDATION_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(ArtifactStage.EVALUATION).sha256)
    completion = _completion(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=ArtifactStage.EVALUATION,
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.PREDICTION.value), payload, Sha256Digest(fingerprint)
            ),
            artifact_type=ArtifactType.PREDICTION,
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


def _fixture_seed(
    base_seed: RandomSeed, world_kind: UnresolvedMapWorldKind, fixture_index: Index
) -> RandomSeed:
    derived: RandomSeed = derive_seed32(
        SeedDerivationRequest(
            base_seed,
            RngNamespace.SYNTHETIC_INSTANCE,
            cast(
                StableJsonPayload,
                OrderedDict(world_kind=world_kind.value, fixture_index=fixture_index),
            ),
        )
    )
    return derived


def _persist_map_world_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    world_kind: UnresolvedMapWorldKind,
    fixture_index: Index,
    seed: RandomSeed,
    world: UnresolvedMapWorld,
    overwrite_policy: OverwritePolicy,
) -> None:
    condition = EvaluationConditionName(f"{world_kind.value}-{fixture_index}")
    input_artifact_ids = (ArtifactIdentifier("synthetic-generator"),)
    for metric_name, metric_value in (
        (MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE, world.certified_robust_value),
        (MetricId.EXACT_MAP_ACTION_VALUE, world.diagnostics.exact_map_action_value),
        (MetricId.ORBIT_RADIUS_MAP_BOUND, world.diagnostics.bound),
    ):
        persist_synthetic_diagnostic_metric(
            store,
            layout,
            experiment,
            condition,
            seed,
            metric_name,
            float(metric_value),
            MetricUnit("score"),
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            overwrite_policy,
        )


def _run_unresolved_map_fixtures(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    world_kind: UnresolvedMapWorldKind,
    fixtures_per_seed: ReplicateCount,
) -> None:
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    for seed in confirmatory_seeds:
        for fixture_index in range(fixtures_per_seed):
            index: Index = fixture_index
            fixture_seed = _fixture_seed(seed, world_kind, index)
            world = generate_unresolved_map_world(
                UnresolvedMapWorldRequest(world_kind, fixture_seed)
            )
            _persist_map_world_metrics(
                store,
                layout,
                request.experiment,
                world_kind,
                index,
                seed,
                world,
                request.overwrite_policy,
            )


def execute_common_action_under_unidentified_map(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    config = active_config().experiments.common_action_under_unidentified_map
    _run_unresolved_map_fixtures(
        store, layout, request, UnresolvedMapWorldKind.COMMON_ACTION, config.fixtures_per_seed
    )


def execute_robust_compromise_under_unidentified_map(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    config = active_config().experiments.robust_compromise_under_unidentified_map
    _run_unresolved_map_fixtures(
        store, layout, request, UnresolvedMapWorldKind.ROBUST_COMPROMISE, config.fixtures_per_seed
    )


def execute_map_dependent_action_boundary(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    config = active_config().experiments.map_dependent_action_boundary
    _run_unresolved_map_fixtures(
        store, layout, request, UnresolvedMapWorldKind.MAP_DEPENDENT, config.fixtures_per_seed
    )


def execute_exact_map_value_bound_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    config = active_config().experiments.exact_map_value_bound_validation
    _run_unresolved_map_fixtures(
        store,
        layout,
        request,
        UnresolvedMapWorldKind.COMMON_ACTION,
        config.zero_map_value_fixtures_per_seed,
    )
    _run_unresolved_map_fixtures(
        store,
        layout,
        request,
        UnresolvedMapWorldKind.MAP_DEPENDENT,
        config.high_map_value_fixtures_per_seed,
    )


def _synthetic_coupling_problem(
    instance: CouplingInstance,
) -> tuple[RobustActionProblem, tuple[BlockCorrespondence, ...], CurriculumAction, RectangularHull]:
    groups = tuple(CoarseGroup)[: len(instance.block_pattern)]
    counts = OrderedDict(zip(groups, instance.block_pattern, strict=True))
    blocks = build_padded_block_structure(groups, counts, counts)
    orbit = tuple(enumerate_block_permutations(blocks))
    problem = build_robust_action_problem(
        blocks,
        instance.lower_response_matrix,
        instance.lower_response_matrix,
        instance.target_importance,
        tuple(range(sum(instance.block_pattern))),
    )
    alpha = CurriculumAction(problem=problem, coordinates=instance.active_action)
    hull = build_rectangular_hull(
        blocks, instance.lower_response_matrix, instance.lower_response_matrix
    )
    return problem, orbit, alpha, hull


def _persist_coupling_mechanism_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportCount,
    method: TransferMethod,
    seed: RandomSeed,
    problem: RobustActionProblem,
    orbit: tuple[BlockCorrespondence, ...],
    alpha: CurriculumAction,
    hull: RectangularHull,
    overwrite_policy: OverwritePolicy,
) -> None:
    input_artifact_ids = (ArtifactIdentifier("synthetic-generator"),)
    candidates = (alpha, zero_action(problem))
    for metric_name, metric_value in (
        (
            MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP,
            fixed_action_rectangularization_gap(alpha, orbit, hull.lower_bounds),
        ),
        (
            MetricId.ROBUST_COUPLING_VALUE_GAP,
            robust_coupling_gap(candidates, problem, orbit, hull),
        ),
        (
            MetricId.COUPLING_UPPER_BOUND_DIAGNOSTIC,
            rectangular_value_over_candidates(candidates, problem, hull),
        ),
    ):
        persist_synthetic_benchmark_metric(
            store,
            layout,
            experiment,
            condition,
            support,
            method,
            seed,
            metric_name,
            float(metric_value),
            MetricUnit("score"),
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            overwrite_policy,
        )


def execute_synthetic_coupling_mechanism_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    coupling_config = active_config().generators.coupling_structure
    seeds = active_config().scientific.randomness.confirmatory_seeds
    for compatibility in coupling_config.compatibility:
        eligible_supports = eligible_coupling_support_sizes(compatibility, coupling_config.supports)
        for support in eligible_supports:
            for heterogeneity in coupling_config.response_heterogeneity:
                for asymmetry in coupling_config.directed_asymmetry:
                    for sparsity in coupling_config.response_sparsity:
                        for block_pattern in coupling_config.block_patterns:
                            condition = EvaluationConditionName(
                                f"{compatibility.value}-{heterogeneity}-{asymmetry}-"
                                f"{sparsity}-{'x'.join(str(size) for size in block_pattern)}"
                            )
                            for seed in seeds:
                                coupling_request = CouplingInstanceRequest(
                                    compatibility=compatibility,
                                    response_heterogeneity=heterogeneity,
                                    directed_asymmetry=asymmetry,
                                    response_sparsity=sparsity,
                                    block_pattern=tuple(block_pattern),
                                    support_size=support,
                                    seed=seed,
                                    instance_index=0,
                                )
                                try:
                                    instance = generate_coupling_instance(coupling_request)
                                except CouplingGenerationError:
                                    continue
                                problem, orbit, alpha, hull = _synthetic_coupling_problem(instance)
                                _persist_coupling_mechanism_metrics(
                                    store,
                                    layout,
                                    request.experiment,
                                    condition,
                                    support,
                                    TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                                    seed,
                                    problem,
                                    orbit,
                                    alpha,
                                    hull,
                                    request.overwrite_policy,
                                )
                                destroyed = coupling_destroyed_matrices(
                                    problem.blocks,
                                    instance.lower_response_matrix,
                                    instance.lower_response_matrix,
                                    seed,
                                    condition,
                                )
                                destroyed_problem = build_robust_action_problem(
                                    problem.blocks,
                                    destroyed.lower_response_matrix,
                                    destroyed.lower_response_matrix,
                                    instance.target_importance,
                                    tuple(range(sum(instance.block_pattern))),
                                )
                                destroyed_alpha = CurriculumAction(
                                    problem=destroyed_problem, coordinates=instance.active_action
                                )
                                destroyed_hull = build_rectangular_hull(
                                    problem.blocks,
                                    destroyed.lower_response_matrix,
                                    destroyed.lower_response_matrix,
                                )
                                _persist_coupling_mechanism_metrics(
                                    store,
                                    layout,
                                    request.experiment,
                                    condition,
                                    support,
                                    TransferMethod.COUPLING_DESTROYED_FEDORBIT,
                                    seed,
                                    destroyed_problem,
                                    orbit,
                                    destroyed_alpha,
                                    destroyed_hull,
                                    request.overwrite_policy,
                                )


def _solve_fedorbit_exact_sparse_action_at_support(
    support_limit: SupportCount,
) -> Callable[[RobustActionProblem, RandomSeed], CurriculumAction | None]:
    def solve(problem: RobustActionProblem, seed: RandomSeed) -> CurriculumAction | None:
        del seed
        return solve_robust_action(problem, support_limit).selected_action

    return solve


def _solve_dense_ccp_fallback_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    outcome = solve_dense_ccp(
        problem, seed, SemanticCoordinates(f"sparsity-and-dense-fallback:{seed}")
    )
    return outcome.selected_action


def execute_sparsity_and_dense_fallback(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = materialize_client(dataset, raw_root)
        return materialized_by_dataset.get(dataset)

    conditions: tuple[
        tuple[
            str,
            TransferMethod,
            Callable[[RobustActionProblem, RandomSeed], CurriculumAction | None],
        ],
        ...,
    ] = (
        (
            "exact sparse s=1",
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            _solve_fedorbit_exact_sparse_action_at_support(1),
        ),
        (
            "exact sparse s=2",
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            _solve_fedorbit_exact_sparse_action_at_support(2),
        ),
        (
            "exact sparse s=3",
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            _solve_fedorbit_exact_sparse_action_at_support(3),
        ),
        ("dense CCP", TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK, _solve_dense_ccp_fallback_action),
    )
    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        for seed in confirmatory_seeds:
            for condition_label, method, solve_action in conditions:
                scored = _score_robust_action_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    target_materialized,
                    seed,
                    device,
                    "sparsity-and-dense-fallback",
                    solve_action,
                )
                if scored is None:
                    continue
                score, n_classes, input_artifact_ids = scored
                _persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    method,
                    seed,
                    score,
                    n_classes,
                    input_artifact_ids,
                    EvaluationConditionName(condition_label),
                )


def execute_real_packet_coupling_mechanism_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = materialize_client(dataset, raw_root)
        return materialized_by_dataset.get(dataset)

    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        for seed in confirmatory_seeds:
            assembly = _assemble_principal_action(
                store,
                layout,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                device,
                _solve_fedorbit_exact_sparse_action,
            )
            if assembly is None:
                continue
            orbit = tuple(enumerate_block_permutations(assembly.blocks))
            hull = build_rectangular_hull(
                assembly.blocks,
                assembly.problem.lower_response_matrix,
                assembly.problem.upper_response_matrix,
            )
            candidates = (assembly.action, zero_action(assembly.problem))
            for metric_name, metric_value in (
                (
                    MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP,
                    fixed_action_rectangularization_gap(assembly.action, orbit, hull.lower_bounds),
                ),
                (
                    MetricId.ROBUST_COUPLING_VALUE_GAP,
                    robust_coupling_gap(candidates, assembly.problem, orbit, hull),
                ),
                (
                    MetricId.COUPLING_UPPER_BOUND_DIAGNOSTIC,
                    rectangular_value_over_candidates(candidates, assembly.problem, hull),
                ),
            ):
                persist_primary_transfer_metric(
                    store,
                    layout,
                    request.experiment,
                    pair_direction,
                    source,
                    target,
                    TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                    seed,
                    metric_name,
                    float(metric_value),
                    MetricUnit("score"),
                    MetricDirection.DESCRIPTIVE,
                    assembly.input_artifact_ids,
                    request.overwrite_policy,
                )


def execute_multi_source_selection_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    config = active_config().experiments.multi_source_selection_validation
    all_clients = active_config().scientific.datasets.clients
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = materialize_client(dataset, raw_root)
        return materialized_by_dataset.get(dataset)

    for target in config.targets:
        target_materialized = materialized(target)
        if target_materialized is None:
            continue
        candidate_sources = tuple(client for client in all_clients if client != target)
        for seed in confirmatory_seeds:
            assemblies_by_source: OrderedDict[DatasetId, PrincipalActionAssembly] = OrderedDict()
            proposals: list[SourceProposal] = []
            for source in candidate_sources:
                source_materialized = materialized(source)
                if source_materialized is None:
                    continue
                assembly = _assemble_principal_action(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    target_materialized,
                    seed,
                    device,
                    _solve_fedorbit_exact_sparse_action,
                )
                if assembly is None:
                    continue
                solution = solve_robust_action(assembly.problem)
                assemblies_by_source[source] = assembly
                proposals.append(
                    SourceProposal(SourceClientName(source.value), solution.certified_robust_value)
                )
            if not proposals:
                continue
            ranked = rank_source_proposals(proposals)

            def confirmation_decision(
                proposal: SourceProposal,
                *,
                target: DatasetId = target,
                seed: RandomSeed = seed,
                assemblies_by_source: Mapping[
                    DatasetId, PrincipalActionAssembly
                ] = assemblies_by_source,
            ) -> bool:
                source = DatasetId(proposal.source_client_name)
                assembly = assemblies_by_source[source]
                multipliers = curriculum_multipliers_from_action(
                    assembly.action, assembly.blocks, assembly.target_eligible, assembly.n_classes
                )
                optimizer = make_adamw(
                    assembly.model,
                    assembly.checkpoint.selected_hyperparameters.learning_rate,
                    assembly.checkpoint.selected_hyperparameters.weight_decay,
                )
                pre_confirm = capture_pre_confirm_pair(assembly.model, optimizer)
                contrast_coordinates = ContrastCoordinates(
                    f"multi-source-selection:{source.value}-to-{target.value}:{seed}"
                )
                verdict = run_proposal_confirmation(
                    ConfirmationRequest(
                        assembly.model,
                        pre_confirm.baseline,
                        pre_confirm.curriculum,
                        assembly.train.features,
                        assembly.train.targets,
                        assembly.confirm.features,
                        assembly.confirm.targets,
                        assembly.checkpoint.train_class_weights,
                        multipliers,
                        assembly.checkpoint.selected_hyperparameters,
                        seed,
                        contrast_coordinates,
                    )
                )
                if verdict.accepted:
                    apply_accepted_assimilation(
                        assembly.model,
                        optimizer,
                        pre_confirm.curriculum,
                        assembly.train.features,
                        assembly.train.targets,
                        assembly.checkpoint.train_class_weights,
                        multipliers,
                        seed,
                        AssimilationCoordinates(
                            target_client=SourceClientName(target.value),
                            directed_pair=DirectedPairName(f"{source.value} -> {target.value}"),
                            condition=EvaluationConditionName("principal"),
                            seed=seed,
                            clean_pretransfer_checkpoint_artifact_id=assembly.checkpoint_artifact_id,
                            source_packet_artifact_id=assembly.first_packet_artifact_id,
                            action_artifact_sha256=_action_sha256(assembly.action),
                        ),
                    )
                else:
                    settle_rejected_proposal(assembly.model, optimizer, pre_confirm.baseline)
                return verdict.accepted

            decision = select_source_sequentially(ranked, confirmation_decision)
            accepted_source = (
                assemblies_by_source[DatasetId(decision.accepted_proposal.source_client_name)]
                if decision.accepted_proposal is not None
                else None
            )
            input_artifact_ids = (
                accepted_source.input_artifact_ids
                if accepted_source is not None
                else next(iter(assemblies_by_source.values())).input_artifact_ids
            )
            persist_synthetic_diagnostic_metric(
                store,
                layout,
                request.experiment,
                EvaluationConditionName(target.value),
                seed,
                MetricId.PROPOSAL_ACCEPTANCE_RATE,
                1.0 if decision.accepted_proposal is not None else 0.0,
                MetricUnit("fraction"),
                MetricDirection.DESCRIPTIVE,
                input_artifact_ids,
                request.overwrite_policy,
            )


def _solve_matched_resource_rectangular_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    del seed
    hull = build_rectangular_hull(
        problem.blocks, problem.lower_response_matrix, problem.upper_response_matrix
    )
    return optimize_against_fixed_matrix(problem, hull.lower_bounds).selected_action


_SEMANTIC_PARTITION_MERGE: Mapping[CoarseGroup, CoarseGroup] = OrderedDict(
    (
        (CoarseGroup.DISRUPTION, CoarseGroup.DISRUPTION),
        (CoarseGroup.EXPLOITATION, CoarseGroup.DISRUPTION),
        (CoarseGroup.ACCESS_AND_DISCOVERY, CoarseGroup.ACCESS_AND_DISCOVERY),
    )
)
_SEMANTIC_PARTITION_SUPERGROUP: Mapping[CoarseGroup, CoarseGroup] = OrderedDict(
    (group, CoarseGroup.DISRUPTION) for group in CoarseGroup
)
_SEMANTIC_PARTITION_PRINCIPAL: Mapping[CoarseGroup, CoarseGroup] = OrderedDict(
    (group, group) for group in CoarseGroup
)


def _semantic_partition_bucket_of(
    partition: str | tuple[str, ...],
) -> Mapping[CoarseGroup, CoarseGroup] | None:
    if partition == "principal_three_coarse_groups":
        return _SEMANTIC_PARTITION_PRINCIPAL
    if partition == "one_attack_supergroup":
        return _SEMANTIC_PARTITION_SUPERGROUP
    if isinstance(partition, tuple) and set(partition) == {
        "Disruption or Exploitation",
        "Access and Discovery",
    }:
        return _SEMANTIC_PARTITION_MERGE
    return None


def _semantic_partition_label(partition: str | tuple[str, ...]) -> str:
    return partition if isinstance(partition, str) else "|".join(partition)


def execute_semantic_sufficiency_frontier(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    config = active_config().experiments.semantic_sufficiency_frontier
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = materialize_client(dataset, raw_root)
        return materialized_by_dataset.get(dataset)

    scorers = (
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, _solve_fedorbit_exact_sparse_action),
        (TransferMethod.MATCHED_RESOURCE_RECTANGULAR, _solve_matched_resource_rectangular_action),
        (TransferMethod.EXACT_MAP_ORACLE, _solve_exact_map_oracle_action),
    )
    for partition in config.partitions:
        bucket_of = _semantic_partition_bucket_of(partition)
        if bucket_of is None:
            continue
        condition = EvaluationConditionName(_semantic_partition_label(partition))
        for directed_pair in primary_pairs:
            source = directed_pair.source
            target = directed_pair.target
            source_materialized = materialized(source)
            target_materialized = materialized(target)
            if source_materialized is None or target_materialized is None:
                continue
            pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
            for seed in confirmatory_seeds:
                for method, solve_action in scorers:
                    scored = _score_robust_action_cell(
                        store,
                        layout,
                        source,
                        target,
                        source_materialized,
                        target_materialized,
                        seed,
                        device,
                        "semantic-sufficiency-frontier",
                        solve_action,
                        None,
                        bucket_of,
                    )
                    if scored is None:
                        continue
                    score, n_classes, input_artifact_ids = scored
                    _persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                        condition,
                    )


def _response_scale_perturbation(
    scale: float,
) -> Callable[
    [PaddedBlockStructure, ResponseMatrix, ResponseMatrix], tuple[ResponseMatrix, ResponseMatrix]
]:
    def perturb(
        blocks: PaddedBlockStructure, lower: ResponseMatrix, upper: ResponseMatrix
    ) -> tuple[ResponseMatrix, ResponseMatrix]:
        del blocks
        midpoint = (lower + upper) / 2.0
        half_width = (upper - lower) / 2.0
        return scale * midpoint - half_width, scale * midpoint + half_width

    return perturb


def _ci_half_width_perturbation(
    multiplier: float,
) -> Callable[
    [PaddedBlockStructure, ResponseMatrix, ResponseMatrix], tuple[ResponseMatrix, ResponseMatrix]
]:
    def perturb(
        blocks: PaddedBlockStructure, lower: ResponseMatrix, upper: ResponseMatrix
    ) -> tuple[ResponseMatrix, ResponseMatrix]:
        del blocks
        midpoint = (lower + upper) / 2.0
        half_width = (upper - lower) / 2.0
        return midpoint - multiplier * half_width, midpoint + multiplier * half_width

    return perturb


def _response_heterogeneity_perturbation(
    multiplier: float,
) -> Callable[
    [PaddedBlockStructure, ResponseMatrix, ResponseMatrix], tuple[ResponseMatrix, ResponseMatrix]
]:
    def perturb(
        blocks: PaddedBlockStructure, lower: ResponseMatrix, upper: ResponseMatrix
    ) -> tuple[ResponseMatrix, ResponseMatrix]:
        midpoint = (lower + upper) / 2.0
        half_width = (upper - lower) / 2.0
        new_midpoint = midpoint.copy()
        block_count = len(blocks.padded_size_tuple)
        for row_block in range(block_count):
            row_range = blocks.block_index_range(row_block)
            for col_block in range(block_count):
                col_range = blocks.block_index_range(col_block)
                segment = midpoint[
                    row_range.start : row_range.stop, col_range.start : col_range.stop
                ]
                block_mean = segment.mean()
                new_midpoint[row_range.start : row_range.stop, col_range.start : col_range.stop] = (
                    block_mean + multiplier * (segment - block_mean)
                )
        return new_midpoint - half_width, new_midpoint + half_width

    return perturb


_WeakSignalPerturbation = Callable[
    [PaddedBlockStructure, ResponseMatrix, ResponseMatrix], tuple[ResponseMatrix, ResponseMatrix]
]


def execute_weak_signal_support_and_heterogeneity_boundaries(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    config = active_config().experiments.weak_signal_support_and_heterogeneity_boundaries
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = materialize_client(dataset, raw_root)
        return materialized_by_dataset.get(dataset)

    conditions: list[tuple[str, _WeakSignalPerturbation | None, SupportCount | None]] = []
    for scale in config.response_scales:
        conditions.append((f"response-scale-{scale}", _response_scale_perturbation(scale), None))
    for multiplier in config.ci_half_width_multipliers:
        conditions.append(
            (f"ci-half-width-{multiplier}", _ci_half_width_perturbation(multiplier), None)
        )
    for multiplier in config.response_heterogeneity_multipliers:
        conditions.append(
            (
                f"response-heterogeneity-{multiplier}",
                _response_heterogeneity_perturbation(multiplier),
                None,
            )
        )
    for support in config.support_budgets:
        conditions.append((f"support-budget-{support}", None, support))

    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        for seed in confirmatory_seeds:
            local_only = _score_local_only_cell_adapter(
                store,
                layout,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                device,
            )
            for condition_label, perturb, support_limit in conditions:
                exact_sparse_solve = (
                    _solve_fedorbit_exact_sparse_action_at_support(support_limit)
                    if support_limit is not None
                    else _solve_fedorbit_exact_sparse_action
                )
                condition = EvaluationConditionName(condition_label)
                for method, solve_action in (
                    (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, exact_sparse_solve),
                    (
                        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                        _solve_matched_resource_rectangular_action,
                    ),
                ):
                    scored = _score_robust_action_cell(
                        store,
                        layout,
                        source,
                        target,
                        source_materialized,
                        target_materialized,
                        seed,
                        device,
                        "weak-signal-boundaries",
                        solve_action,
                        None,
                        None,
                        perturb,
                    )
                    if scored is None:
                        continue
                    score, n_classes, input_artifact_ids = scored
                    _persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                        condition,
                    )
                if local_only is not None:
                    score, n_classes, input_artifact_ids = local_only
                    _persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        TransferMethod.LOCAL_ONLY,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                        condition,
                    )

    for fraction in config.target_usable_support_fractions:
        condition = EvaluationConditionName(f"target-usable-support-fraction-{fraction}")
        for directed_pair in primary_pairs:
            source = directed_pair.source
            target = directed_pair.target
            source_materialized = materialized(source)
            target_materialized = materialized(target)
            if source_materialized is None or target_materialized is None:
                continue
            pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
            subsample_seed: RandomSeed = derive_seed32(
                SeedDerivationRequest(
                    confirmatory_seeds[0],
                    RngNamespace.SYNTHETIC_INSTANCE,
                    cast(
                        StableJsonPayload,
                        OrderedDict(target=target.value, fraction=fraction),
                    ),
                )
            )
            subsampled_target = subsampled_materialized_client(
                target_materialized, fraction, subsample_seed
            )
            _execute_client_base_model_pilot(
                store,
                layout,
                request.experiment,
                experiment_relevance(request.experiment),
                target,
                subsampled_target,
                confirmatory_seeds,
                request.overwrite_policy,
                device,
                execution_logger(),
            )
            for seed in confirmatory_seeds:
                local_only = _score_local_only_cell(
                    store, layout, target, subsampled_target, seed, device, request.experiment
                )
                for method, solve_action in (
                    (
                        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                        _solve_fedorbit_exact_sparse_action,
                    ),
                    (
                        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                        _solve_matched_resource_rectangular_action,
                    ),
                ):
                    scored = _score_robust_action_cell(
                        store,
                        layout,
                        source,
                        target,
                        source_materialized,
                        subsampled_target,
                        seed,
                        device,
                        "weak-signal-boundaries",
                        solve_action,
                        None,
                        None,
                        None,
                        request.experiment,
                    )
                    if scored is None:
                        continue
                    score, n_classes, input_artifact_ids = scored
                    _persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                        condition,
                    )
                if local_only is not None:
                    score, n_classes, artifact_id = local_only
                    _persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        TransferMethod.LOCAL_ONLY,
                        seed,
                        score,
                        n_classes,
                        (artifact_id,),
                        condition,
                    )


def _human_audit_researcher_id(index: Index) -> str:
    return f"researcher-{index + 1}"


def _human_audit_directory(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    source: DatasetId,
    target: DatasetId,
    researcher_id: str,
) -> Path:
    return (
        experiment_workspace(layout, experiment)
        / "artifacts"
        / "fitted"
        / "human_audit"
        / f"{source.value}-to-{target.value}"
        / researcher_id
    )


def _score_packet_only_recovery_attempt(
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
) -> tuple[bool, tuple[ArtifactIdentifier, ...]] | None:
    common = _common_eligible_groups(source, target, source_materialized, target_materialized)
    if common is None:
        return None
    source_eligible, target_eligible = common
    common_coarse = tuple(target_eligible)
    blocks = cross_client_padded_blocks(source_eligible, target_eligible)
    source_packets: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    target_packets: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    for coarse_group in common_coarse:
        source_packet = _load_dataset_source_packet(layout, source, seed, coarse_group)
        if source_packet is not None:
            source_packets[coarse_group] = source_packet
        target_packet = _load_dataset_source_packet(layout, target, seed, coarse_group)
        if target_packet is not None:
            target_packets[coarse_group] = target_packet
    if not source_packets or not target_packets:
        return None
    source_matrix = assemble_cross_client_response_matrix(
        blocks, source_packets, SourcePacket.lower_matrix
    )
    target_matrix = assemble_target_response_matrix(
        blocks, target_packets, SourcePacket.lower_matrix
    )
    qap_result = point_correspondence_commitment(source_matrix, target_matrix, blocks)
    if not qap_result.certified or qap_result.correspondence is None:
        return None
    oracle_correspondence = BlockCorrespondence.lexicographically_smallest(blocks)
    recovered = qap_result.correspondence.images == oracle_correspondence.images
    input_artifact_ids = tuple(
        ArtifactIdentifier(packet.packet_integrity_sha256)
        for packet in (*source_packets.values(), *target_packets.values())
    )
    return recovered, input_artifact_ids


def execute_map_availability_applicability_audit(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    config = active_config().experiments.map_availability_applicability_audit
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    public_labels = documented_public_labels()
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = materialize_client(dataset, raw_root)
        return materialized_by_dataset.get(dataset)

    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        domain_pair = DirectedPair(source=source, target=target)
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is not None and target_materialized is not None:
            pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
            for seed in confirmatory_seeds:
                attempt = _score_packet_only_recovery_attempt(
                    layout, source, target, source_materialized, target_materialized, seed
                )
                if attempt is None:
                    continue
                recovered, input_artifact_ids = attempt
                persist_primary_transfer_metric(
                    store,
                    layout,
                    request.experiment,
                    pair_direction,
                    source,
                    target,
                    TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
                    seed,
                    MetricId.PACKET_ONLY_RECOVERY_ACCURACY,
                    1.0 if recovered else 0.0,
                    MetricUnit("boolean"),
                    MetricDirection.HIGHER_IS_BETTER,
                    input_artifact_ids,
                    request.overwrite_policy,
                )
        submissions: list[MapAvailabilityAuditSubmission] = []
        for researcher_index in range(config.independent_researchers):
            index: Index = researcher_index
            researcher_id = _human_audit_researcher_id(index)
            directory = _human_audit_directory(
                layout, request.experiment, source, target, researcher_id
            )
            submission_path = directory / "submission.json"
            if not submission_path.is_file():
                template = blank_audit_template(researcher_id, domain_pair)
                atomic_write_json(
                    directory / "template.json",
                    cast(StableJsonPayload, template.model_dump(mode="json")),
                )
                continue
            submission = MapAvailabilityAuditSubmission.model_validate_json(
                submission_path.read_text(encoding="utf-8")
            )
            failures = validate_submission(
                submission, config.minutes_per_researcher_per_pair, public_labels
            )
            if failures:
                continue
            submissions.append(submission)
        if len(submissions) != config.independent_researchers or not distinct_researcher_ids(
            tuple(submissions)
        ):
            continue
        for submission in submissions:
            atomic_write_json(
                _human_audit_directory(
                    layout, request.experiment, source, target, submission.researcher_id
                )
                / "validated.sha256.json",
                cast(
                    StableJsonPayload,
                    OrderedDict(sha256=submission_sha256(submission)),
                ),
            )


def _deterministic_replay_consistent(
    node_count: ConceptCount,
    pattern: ScalabilityBlockPattern,
    support: SupportCount,
    seed: RandomSeed,
) -> bool:
    problem_a, action_a, _ = _synthetic_solver_instance(node_count, pattern, support, seed)
    problem_b, action_b, _ = _synthetic_solver_instance(node_count, pattern, support, seed)
    solver_config = active_config().solvers.exact_sparse
    outcome_a = fixed_action_worst_correspondence(
        problem_a,
        action_a,
        solver_config.lap_objective_tie_tolerance,
        solver_config.action_tie_tolerance,
    )
    outcome_b = fixed_action_worst_correspondence(
        problem_b,
        action_b,
        solver_config.lap_objective_tie_tolerance,
        solver_config.action_tie_tolerance,
    )
    return (
        outcome_a.separator_objective == outcome_b.separator_objective
        and outcome_a.worst_correspondence.images == outcome_b.worst_correspondence.images
    )


def execute_baseline_and_oracle_correctness_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    validation_seeds = (confirmatory_seeds[0], confirmatory_seeds[4])
    tractable_config = active_config().experiments.exact_sparse_solver_benchmark
    tractable_k: ConceptCount = tractable_config.synthetic_k.minimum
    tractable_pattern = tractable_config.block_patterns[0]
    tractable_support = tractable_config.supports[0]
    for seed in validation_seeds:
        replay_consistent = _deterministic_replay_consistent(
            tractable_k, tractable_pattern, tractable_support, seed
        )
        persist_synthetic_diagnostic_metric(
            store,
            layout,
            request.experiment,
            EvaluationConditionName(f"deterministic-replay-k{tractable_k}"),
            seed,
            MetricId.DETERMINISTIC_REPLAY_CONSISTENCY,
            1.0 if replay_consistent else 0.0,
            MetricUnit("boolean"),
            MetricDirection.HIGHER_IS_BETTER,
            (ArtifactIdentifier("synthetic-generator"),),
            request.overwrite_policy,
        )
        problem, action, blocks = _synthetic_solver_instance(
            tractable_k, tractable_pattern, tractable_support, seed
        )
        exhaustive_truth = _solver_benchmark_reference_truth(
            blocks, action, tractable_config.exhaustive_truth_correspondence_count_maximum
        )
        if exhaustive_truth is not None:
            qap_result = fixed_action_worst_correspondence_qap(problem, action)
            if qap_result.certified and qap_result.objective_value is not None:
                persist_synthetic_diagnostic_metric(
                    store,
                    layout,
                    request.experiment,
                    EvaluationConditionName(f"generic-qap-vs-exhaustive-truth-k{tractable_k}"),
                    seed,
                    MetricId.ABSOLUTE_OBJECTIVE_ERROR,
                    float(abs(qap_result.objective_value - exhaustive_truth)),
                    MetricUnit("score"),
                    MetricDirection.LOWER_IS_BETTER,
                    (ArtifactIdentifier("synthetic-generator"),),
                    request.overwrite_policy,
                )
            source_matrix = problem.lower_response_matrix
            pc_result = point_correspondence_commitment(source_matrix, source_matrix, blocks)
            if (
                pc_result.certified
                and pc_result.correspondence is not None
                and pc_result.objective_value is not None
            ):
                pc_truth = min(
                    -float(
                        np.sum(
                            correspondence.permute_response_matrix(source_matrix) * source_matrix
                        )
                    )
                    for correspondence in enumerate_block_permutations(blocks)
                )
                persist_synthetic_diagnostic_metric(
                    store,
                    layout,
                    request.experiment,
                    EvaluationConditionName(f"point-map-qap-correctness-k{tractable_k}"),
                    seed,
                    MetricId.ABSOLUTE_OBJECTIVE_ERROR,
                    float(abs(pc_result.objective_value - pc_truth)),
                    MetricUnit("score"),
                    MetricDirection.LOWER_IS_BETTER,
                    (ArtifactIdentifier("synthetic-generator"),),
                    request.overwrite_policy,
                )
            exhaustive_hull = build_rectangular_hull(blocks, source_matrix, source_matrix)
            analytic_lower, analytic_upper = analytic_rectangular_hull_bounds(
                blocks, source_matrix, source_matrix
            )
            hull_max_error = float(
                max(
                    np.max(np.abs(exhaustive_hull.lower_bounds - analytic_lower)),
                    np.max(np.abs(exhaustive_hull.upper_bounds - analytic_upper)),
                )
            )
            persist_synthetic_diagnostic_metric(
                store,
                layout,
                request.experiment,
                EvaluationConditionName(f"rectangular-baseline-vs-analytical-k{tractable_k}"),
                seed,
                MetricId.ABSOLUTE_OBJECTIVE_ERROR,
                hull_max_error,
                MetricUnit("score"),
                MetricDirection.LOWER_IS_BETTER,
                (ArtifactIdentifier("synthetic-generator"),),
                request.overwrite_policy,
            )
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = materialize_client(dataset, raw_root)
        return materialized_by_dataset.get(dataset)

    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        try:
            validate_disjoint_feature_namespaces(
                frozenset(str(name) for name in source_materialized.feature_names),
                frozenset(str(name) for name in target_materialized.feature_names),
            )
            validate_oracle_acl_isolation(
                cell_is_oracle_validation_context=True, oracle_information_accessed=True
            )
            valid = True
        except StrictResourceViolationError:
            valid = False
        input_artifact_ids = (ArtifactIdentifier("materialized-client-schema"),)
        for seed in validation_seeds:
            persist_primary_transfer_metric(
                store,
                layout,
                request.experiment,
                pair_direction,
                source,
                target,
                TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                seed,
                MetricId.STRICT_RESOURCE_VALIDITY,
                1.0 if valid else 0.0,
                MetricUnit("boolean"),
                MetricDirection.HIGHER_IS_BETTER,
                input_artifact_ids,
                request.overwrite_policy,
            )


def _execute_client_base_model_pilot(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    relevance: frozenset[SemanticCoordinate],
    dataset: DatasetId,
    materialized: MaterializedClient,
    confirmatory_seeds: tuple[RandomSeed, ...],
    overwrite_policy: OverwritePolicy,
    device: torch.device,
    logger: ExecutionLogger,
) -> None:
    train = materialized.splits[Split.TRAIN]
    valid = materialized.splits[Split.VALID]
    n_classes = materialized.class_manifest.class_count
    pilot_data = PilotData(train.features, train.targets, valid.features, valid.targets, n_classes)
    class_weights = ClassWeights.from_targets(train.targets, n_classes)
    with principal_determinism():
        pilot_started_at = time.monotonic()
        logger.record(
            ExecutionLogEvent(
                occurred_at=datetime.now(UTC),
                cell_coordinates=SemanticCoordinates(f"{experiment.value}:{dataset.value}:pilot"),
                artifact_id=None,
                state=ArtifactState.RUNNING,
                stage=ExecutionStageName(ArtifactStage.PILOT_SELECTION.value),
                experiment=experiment,
                dataset=dataset,
            )
        )
        pilot_results = run_base_model_pilot(pilot_data, dataset, device)
        selection = select_pilot_configuration(pilot_results)
        logger.record(
            ExecutionLogEvent(
                occurred_at=datetime.now(UTC),
                cell_coordinates=SemanticCoordinates(f"{experiment.value}:{dataset.value}:pilot"),
                artifact_id=None,
                state=ArtifactState.COMPLETED,
                stage=ExecutionStageName(ArtifactStage.PILOT_SELECTION.value),
                experiment=experiment,
                dataset=dataset,
                elapsed_seconds=time.monotonic() - pilot_started_at,
            )
        )
        for pilot_result in pilot_results:
            if pilot_result.configuration != selection.configuration:
                continue
            _persist_base_checkpoint(
                store,
                layout,
                experiment,
                relevance,
                dataset,
                pilot_result.seed,
                pilot_result.outcome.checkpoint,
                ArtifactStage.PILOT_SELECTION,
                CheckpointDirectorySegment.PILOT,
                overwrite_policy,
            )
        for seed_index, seed in enumerate(confirmatory_seeds):
            checkpoint_coordinates = SemanticCoordinates(
                f"{experiment.value}:{dataset.value}:confirmatory:{seed}"
            )
            checkpoint_cell = SemanticCell(
                experiment=experiment,
                dataset=dataset,
                source_client=dataset,
                seed=ExperimentSeed(seed),
            )
            fingerprint = stage_dependency_fingerprint(
                ArtifactStage.TRAINING,
                checkpoint_cell,
                relevance,
                (),
                _BASE_MODEL_PILOT_CONFIGURATION_SECTIONS,
                _MODULE_NAME,
            )
            if overwrite_policy == OverwritePolicy.REUSE:
                existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
                if existing is not None:
                    logger.record(
                        ExecutionLogEvent(
                            occurred_at=datetime.now(UTC),
                            cell_coordinates=checkpoint_coordinates,
                            artifact_id=existing.artifact_id,
                            state=ArtifactState.COMPLETED,
                            stage=ExecutionStageName(ArtifactStage.TRAINING.value),
                            experiment=experiment,
                            dataset=dataset,
                            seed=seed,
                            reuse_decision=ReuseDecision("reused"),
                        )
                    )
                    continue
            logger.record(
                ExecutionLogEvent(
                    occurred_at=datetime.now(UTC),
                    cell_coordinates=checkpoint_coordinates,
                    artifact_id=None,
                    state=ArtifactState.RUNNING,
                    stage=ExecutionStageName(ArtifactStage.TRAINING.value),
                    experiment=experiment,
                    dataset=dataset,
                    seed=seed,
                    reuse_decision=ReuseDecision(
                        f"{seed_index + 1}/{len(confirmatory_seeds)} confirmatory checkpoints"
                    ),
                )
            )
            checkpoint_started_at = time.monotonic()
            with principal_determinism(), measure_efficiency() as efficiency:
                model = create_classifier(
                    dataset,
                    train.features.shape[1],
                    n_classes,
                    selection.configuration.dropout,
                    seed,
                    device,
                )
                outcome = train_base_model(
                    model,
                    train.features,
                    train.targets,
                    valid.features,
                    valid.targets,
                    class_weights,
                    seed,
                    selection.configuration.hyperparameters(),
                )
            _persist_training_efficiency(layout, experiment, dataset, seed, efficiency.result)
            _persist_base_checkpoint(
                store,
                layout,
                experiment,
                relevance,
                dataset,
                seed,
                outcome.checkpoint,
                ArtifactStage.TRAINING,
                CheckpointDirectorySegment.TRAINING,
                overwrite_policy,
            )
            logger.record(
                ExecutionLogEvent(
                    occurred_at=datetime.now(UTC),
                    cell_coordinates=checkpoint_coordinates,
                    artifact_id=None,
                    state=ArtifactState.COMPLETED,
                    stage=ExecutionStageName(ArtifactStage.TRAINING.value),
                    experiment=experiment,
                    dataset=dataset,
                    seed=seed,
                    elapsed_seconds=time.monotonic() - checkpoint_started_at,
                )
            )


def _persist_training_efficiency(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    dataset: DatasetId,
    seed: RandomSeed,
    measurement: EfficiencyMeasurement,
) -> Path:
    record = EfficiencyRecord(
        wall_time_seconds=measurement.wall_time_seconds,
        peak_host_rss_mib=measurement.peak_host_rss_mib,
        peak_cuda_allocated_bytes=measurement.peak_cuda_allocated_bytes,
        packet_serialized_byte_count=0,
        source_response_optimizer_steps=0,
        target_confirmation_optimizer_steps=0,
        live_assimilation_optimizer_steps=0,
        timeout_indicator=False,
        resource_limit_indicator=False,
    )
    destination = (
        experiment_workspace(layout, experiment)
        / "artifacts"
        / "derived"
        / f"training-efficiency.{dataset.value}.{seed}.json"
    )
    atomic_write_json(destination, OrderedDict[str, StableJsonPayload](asdict(record)))
    return destination


def _persist_base_checkpoint(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    relevance: frozenset[SemanticCoordinate],
    dataset: DatasetId,
    seed: RandomSeed,
    checkpoint: BaseCheckpoint,
    stage: ArtifactStage,
    directory_segment: CheckpointDirectorySegment,
    overwrite_policy: OverwritePolicy,
) -> None:
    checkpoint_cell = SemanticCell(
        experiment=experiment,
        dataset=dataset,
        source_client=dataset,
        seed=ExperimentSeed(seed),
    )
    coordinates = SemanticCoordinateText(checkpoint_cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            stage,
            checkpoint_cell,
            relevance,
            (),
            _BASE_MODEL_PILOT_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return
    payload_path = (
        experiment_workspace(layout, experiment)
        / "checkpoints"
        / directory_segment
        / dataset.value
        / f"seed-{seed}"
        / "checkpoint.pt"
    )
    save_base_checkpoint(checkpoint, payload_path)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_BASE_MODEL_PILOT_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(stage).sha256)
    completion = _completion(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=stage,
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.CHECKPOINT.value),
                cast(
                    StableJsonPayload,
                    OrderedDict(coordinates=coordinates, payload_sha256=payload_sha256),
                ),
                Sha256Digest(fingerprint),
            ),
            artifact_type=ArtifactType.CHECKPOINT,
            semantic_producer_coordinates=coordinates,
            producer_stage=stage,
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


class PrimitiveValidationError(ValueError):
    pass


_EXPERIMENT = ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION
_STAGE = ArtifactStage.EVALUATION
_CONFIGURATION_SECTIONS = frozenset(
    {ConfigurationSection.ACTION, ConfigurationSection.GENERATORS, ConfigurationSection.METRICS}
)


def execute_primitive_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    overwrite_policy: OverwritePolicy = OverwritePolicy.REUSE,
) -> ReusableArtifactManifest:
    configuration = active_config()
    seed = ExperimentSeed(0)
    cell = SemanticCell(experiment=_EXPERIMENT, seed=seed)
    relevance = experiment_relevance(_EXPERIMENT)
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            _STAGE,
            cell,
            relevance,
            (),
            _CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    payload_path = _payload_path(layout, fingerprint)
    payload = _validation_payload(
        configuration.generators.exact_separator_theorem.block_patterns[0]
    )
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(configuration_subset_digest(_CONFIGURATION_SECTIONS))
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(_STAGE).sha256)
    completion = _completion(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.OTHER.value), payload, Sha256Digest(fingerprint)
            ),
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


def _payload_path(layout: WorkspaceLayout, fingerprint: Sha256Digest) -> Path:
    return (
        experiment_workspace(layout, _EXPERIMENT)
        / "artifacts"
        / "derived"
        / f"primitive-validation.{fingerprint[:16]}.json"
    )


def _validation_payload(block_pattern: tuple[ConceptCount, ...]) -> StableJsonPayload:
    source_seed = active_config().scientific.randomness.pilot_seeds[0]
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
            seed=source_seed,
            response_shape=list(instance.lower_response_matrix.shape),
            lower_bound_not_above_upper_bound=True,
            assignment_is_bijective=True,
            assignment_objective=assignment.objective_value,
            scoring_row_count=len(scoring.rows),
            scoring_macro_cross_entropy=scoring.macro_cross_entropy.value,
        ),
    )


def _score_deterministic_validation_batch() -> ScoreArtifact:
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
    coordinates: SemanticCoordinateText,
    fingerprint: Sha256Digest,
    payload_path: ArtifactPath,
    payload_sha256: Sha256Digest,
    configuration_sha256: Sha256Digest,
    code_sha256: Sha256Digest,
    runtime_sha256: Sha256Digest,
    stage: ArtifactStage = _STAGE,
    upstream_artifact_ids: ArtifactIdentifiers = (),
) -> CompletionManifest:
    completion = CompletionManifest.model_validate(
        OrderedDict(
            schema_version="1.0",
            semantic_experiment_coordinates=coordinates,
            producer_stage=stage,
            terminal_state=TerminalState.COMPLETED,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=upstream_artifact_ids,
            mandatory_artifact_paths=(str(payload_path),),
            mandatory_artifact_sha256=payload_sha256,
            scientific_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            material_runtime_sha256=runtime_sha256,
            upstream_lineage=stable_json(OrderedDict[str, StableJsonPayload]()),
            completion_validation_state="validated",
            completion_written_last=True,
            completion_manifest_sha256="",
        )
    )
    return completion.model_copy(
        update=OrderedDict(completion_manifest_sha256=completion_manifest_self_hash(completion))
    )
