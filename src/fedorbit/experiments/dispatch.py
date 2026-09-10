from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
from torch import nn

from fedorbit.config.loading import active_config
from fedorbit.experiments.audit import execute_map_availability_applicability_audit
from fedorbit.experiments.classification import execute_evidence_classification
from fedorbit.experiments.protocol import ExperimentExecutionRequest
from fedorbit.experiments.solvers import (
    execute_common_action_under_unidentified_map,
    execute_exact_map_value_bound_validation,
    execute_exact_sparse_solver_benchmark,
    execute_map_dependent_action_boundary,
    execute_robust_compromise_under_unidentified_map,
    execute_scalability_and_efficiency,
    execute_synthetic_coupling_mechanism_validation,
)
from fedorbit.experiments.synthesis import execute_statistical_synthesis
from fedorbit.experiments.synthetic import (
    ExactSeparatorInstanceRequest,
    ScalabilityInstanceRequest,
    UnresolvedMapWorldKind,
    UnresolvedMapWorldRequest,
    generate_exact_separator_instance,
    generate_scalability_instance,
    generate_unresolved_map_world,
)
from fedorbit.experiments.training import (
    execute_base_model_pilot,
    execute_final_source_response_band_validation,
    execute_source_response_estimator_pilot,
)
from fedorbit.experiments.transfer import (
    execute_mechanism_ablations,
    execute_multi_source_selection_validation,
    execute_primary_strict_cross_telemetry_transfer,
    execute_real_packet_coupling_mechanism_validation,
    execute_secondary_cross_modality_generalization,
    execute_semantic_sufficiency_frontier,
    execute_sparsity_and_dense_fallback,
    execute_target_confirmation_and_portability,
    execute_weak_signal_support_and_heterogeneity_boundaries,
)
from fedorbit.experiments.validation import (
    execute_baseline_and_oracle_correctness_validation,
    execute_coupling_and_map_bound_validation,
    execute_dataset_client_and_resource_validation,
    execute_exact_sparse_theorem_exhaustive_validation,
    execute_primitive_validation,
)
from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
    ExecutionError,
    RecoveryBoundary,
    execution_store,
)
from fedorbit.infrastructure.evidence import VerifiedEvidenceWriter
from fedorbit.infrastructure.failures import (
    InfrastructureFailureError,
    RetryPolicy,
    classify_failure,
)
from fedorbit.infrastructure.manifests import (
    ReusableArtifactManifest,
)
from fedorbit.infrastructure.runtime import (
    ExecutionLogEvent,
    execution_logger,
)
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
    build_layout,
)
from fedorbit.methods.assimilation import (
    capture_pre_confirm_pair,
)
from fedorbit.response.packet import (
    build_source_packet,
)
from fedorbit.response.uncertainty import FinalResponseEntry, FinalResponseEstimate
from fedorbit.types import (
    AnonymousNodeDisplayId,
    ArtifactState,
    ElapsedSeconds,
    ExperimentName,
    ExposedCoarseGroupId,
    InfrastructureLogCoordinate,
    OverwritePolicy,
    ProducerModuleName,
    ReuseDecision,
    Rfc3339UtcTimestamp,
    ScalabilityBlockPattern,
    SemanticCoordinates,
    Sha256Digest,
    StableJsonPayload,
)

_MODULE_NAME = ProducerModuleName("fedorbit.experiments.dispatch")


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


def _latest_completed_manifest(
    store: ArtifactStore, experiment: ExperimentName
) -> ReusableArtifactManifest | None:
    candidates: list[ReusableArtifactManifest] = []
    for manifest in store.all_manifests():
        if experiment.value not in manifest.semantic_producer_coordinates:
            continue
        try:
            resolved = store.resolve(manifest.artifact_id)
        except ValueError:
            continue
        if resolved.state == ArtifactState.COMPLETED:
            candidates.append(resolved)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda manifest: max(
            Path(payload).stat().st_mtime_ns for payload in manifest.payload_paths
        ),
    )


def _write_immediate_experiment_evidence(
    store: ArtifactStore, layout: WorkspaceLayout, experiment: ExperimentName
) -> None:
    manifest = _latest_completed_manifest(store, experiment)
    if manifest is None:
        return
    writer = VerifiedEvidenceWriter(store, layout)
    writer.write(
        experiment,
        manifest.artifact_id,
        cast(
            StableJsonPayload,
            OrderedDict(
                experiment=experiment.value,
                artifact_id=manifest.artifact_id,
                state=manifest.state.value,
                dependency_fingerprint_sha256=manifest.dependency_fingerprint_sha256,
            ),
        ),
        overwrite=True,
    )
    writer.write_metric_exports(experiment, manifest.artifact_id)


def _execute_producer_with_retry(
    producer: Callable[[], ReusableArtifactManifest | None],
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
) -> None:
    policy = RetryPolicy(
        active_config().runtime.failure_handling.retries_after_initial_infrastructure_failure
    )
    logger = execution_logger()
    attempt = 0
    while True:
        started_at = time.perf_counter()
        logger.record(
            ExecutionLogEvent(
                occurred_at=datetime.now(UTC),
                cell_coordinates=SemanticCoordinates(f"{experiment.value}:producer"),
                artifact_id=None,
                state=ArtifactState.RUNNING,
                experiment=experiment,
            )
        )
        try:
            producer()
            elapsed: ElapsedSeconds = time.perf_counter() - started_at
            logger.record(
                ExecutionLogEvent(
                    occurred_at=datetime.now(UTC),
                    cell_coordinates=SemanticCoordinates(f"{experiment.value}:producer"),
                    artifact_id=None,
                    state=ArtifactState.COMPLETED,
                    experiment=experiment,
                    elapsed_seconds=elapsed,
                )
            )
            _write_immediate_experiment_evidence(store, layout, experiment)
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
    producers[ExperimentName.EVIDENCE_CLASSIFICATION] = lambda: execute_evidence_classification(
        store, layout, request
    )
    return producers


def run_experiment(request: ExperimentExecutionRequest) -> None:
    store = execution_store()
    layout = build_layout()
    RecoveryBoundary(store).discard_interrupted_staging()
    logger = execution_logger()
    logger.event(
        "experiment_start",
        experiment=request.experiment.value,
        classification=request.definition.classification.value,
        planned_cells=int(request.definition.derived_planned_cells),
        overwrite_policy=request.overwrite_policy.value,
        seeds=list(request.definition.seeds),
    )
    producer = _registered_experiment_producers(store, layout, request).get(request.experiment)
    if producer is None:
        raise ExecutionError(
            f"registered experiment has no scientific producer: {request.experiment.value}"
        )
    started_at = time.perf_counter()
    _execute_producer_with_retry(producer, store, layout, request.experiment)
    elapsed: ElapsedSeconds = time.perf_counter() - started_at
    logger.event(
        "experiment_end",
        experiment=request.experiment.value,
        elapsed_seconds=elapsed,
        state=ArtifactState.COMPLETED.value,
    )
