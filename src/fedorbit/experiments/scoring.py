from __future__ import annotations

import hashlib
import math
import statistics
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from fedorbit.analysis.metrics import (
    ClassF1,
    ClassF1Set,
    ClassRecall,
    ClassRecallSet,
    balanced_accuracy,
    confusion_counts,
    f1_from_counts,
    macro_f1,
    recall_from_counts,
)
from fedorbit.analysis.records import (
    MetricDirection,
    MetricRecord,
    MetricRecordCollection,
    validate_metric_records,
)
from fedorbit.datasets.common import (
    file_sha256,
)
from fedorbit.datasets.materialization import (
    MaterializedClient,
    SplitTensors,
    TransferConceptGroup,
    transfer_concept_groups,
)
from fedorbit.datasets.ontology import TRANSFER_ONTOLOGY
from fedorbit.experiments.cells import experiment_relevance
from fedorbit.experiments.protocol import ExperimentExecutionRequest
from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
    ExecutionError,
)
from fedorbit.infrastructure.environment import environment_snapshot
from fedorbit.infrastructure.manifests import (
    CompletionManifest,
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
from fedorbit.infrastructure.runtime import (
    RandomSeed,
    current_code_revision,
)
from fedorbit.infrastructure.storage import atomic_write_json
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
    experiment_workspace,
)
from fedorbit.interface import (
    AccessLogger,
    ResourceKind,
    validate_dynamic_access_log_scan,
)
from fedorbit.learning.checkpoints import load_base_checkpoint
from fedorbit.learning.pilot import (
    create_classifier,
)
from fedorbit.learning.scoring import LocalClassCount, ScoreArtifact, ScoringRequest, score_model
from fedorbit.learning.training import (
    BaseCheckpoint,
    make_adamw,
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
from fedorbit.methods.target import (
    CurriculumMultipliers,
    TargetImportanceError,
    TransferNodeRisk,
    build_target_importance,
)
from fedorbit.optimization.certificates import (
    build_rectangular_hull,
)
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    PaddedBlockStructure,
    ResponseMatrix,
    build_padded_block_structure,
    correspondence_block_id,
)
from fedorbit.optimization.dense_ccp import solve_dense_ccp
from fedorbit.optimization.exact_qap import (
    point_correspondence_commitment,
    solve_robust_action_qap,
)
from fedorbit.optimization.exact_sparse import (
    solve_robust_action,
)
from fedorbit.optimization.objective import (
    CurriculumAction,
    RobustActionProblem,
    build_robust_action_problem,
)
from fedorbit.response.packet import (
    SourcePacket,
)
from fedorbit.types import (
    ArtifactFingerprint,
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactPath,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    ArtifactTypeName,
    CheckpointDirectorySegment,
    ClassCount,
    ClassIndex,
    ClientRole,
    CoarseGroup,
    ConfigurationSection,
    ContrastCoordinates,
    CorrespondenceBlockId,
    DatasetId,
    DirectedPair,
    DirectedPairName,
    Estimate,
    EvaluationConditionName,
    ExperimentCondition,
    ExperimentName,
    ExperimentSeed,
    FilesystemSlug,
    Index,
    InvalidReason,
    MetricId,
    MetricUnit,
    MutableCell,
    OracleTransferConcept,
    OverwritePolicy,
    ProducerModuleName,
    SampleCount,
    ScaleFactor,
    Score,
    SemanticCell,
    SemanticCoordinates,
    SemanticCoordinateText,
    SemanticPartitionId,
    SerializedPacket,
    Sha256Digest,
    SourceClientName,
    Split,
    StableJsonPayload,
    SupportCount,
    TerminalState,
    TransferMethod,
    stable_json,
)

_MODULE_NAME = ProducerModuleName("fedorbit.experiments.scoring")


def build_completion_manifest(
    coordinates: SemanticCoordinateText,
    fingerprint: Sha256Digest,
    payload_path: ArtifactPath,
    payload_sha256: Sha256Digest,
    configuration_sha256: Sha256Digest,
    code_sha256: Sha256Digest,
    runtime_sha256: Sha256Digest,
    stage: ArtifactStage = ArtifactStage.EVALUATION,
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


def latest_completed_manifest(
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


_PRINCIPAL_CONDITION = EvaluationConditionName("principal")
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


def score_local_only_cell(
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
    metric_value: float | None,
    metric_unit: MetricUnit,
    direction: MetricDirection,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
    condition: EvaluationConditionName = _PRINCIPAL_CONDITION,
    valid: bool = True,
    invalid_reason: InvalidReason | None = None,
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
            (*(identifier.value for identifier in input_artifact_ids), metric_name.value),
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
        valid=valid,
        invalid_reason=invalid_reason,
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
    completion = build_completion_manifest(
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


def persist_ineligible_transfer_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    pair_direction: DirectedPairName,
    directed_pair_source: DatasetId,
    directed_pair_target: DatasetId,
    method: TransferMethod,
    seed: RandomSeed,
    overwrite_policy: OverwritePolicy,
    condition: EvaluationConditionName = _PRINCIPAL_CONDITION,
) -> ReusableArtifactManifest | None:
    return persist_primary_transfer_metric(
        store,
        layout,
        experiment,
        pair_direction,
        directed_pair_source,
        directed_pair_target,
        method,
        seed,
        MetricId.ABSTENTION_INDICATOR,
        None,
        MetricUnit("boolean"),
        MetricDirection.DESCRIPTIVE,
        (ArtifactIdentifier("ineligible-cell"),),
        overwrite_policy,
        condition,
        valid=False,
        invalid_reason=InvalidReason("INELIGIBLE/ABSTAIN"),
    )


def persist_primary_transfer_cell_metrics(
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


def persist_boundary_diagnostic_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    pair_direction: DirectedPairName,
    source: DatasetId,
    target: DatasetId,
    method: TransferMethod,
    seed: RandomSeed,
    input_artifact_ids: tuple[ArtifactIdentifier, ...],
    condition: EvaluationConditionName,
    certified_value: Score | None,
    action: CurriculumAction | None,
    confirmation_accepted: bool | None,
) -> None:
    diagnostics: list[tuple[MetricId, float, MetricUnit, MetricDirection]] = []
    if certified_value is not None:
        diagnostics.append(
            (
                MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE,
                float(certified_value),
                MetricUnit("score"),
                MetricDirection.DESCRIPTIVE,
            )
        )
    if action is not None:
        abstained = 1.0 if bool(np.all(action.coordinates == 0.0)) else 0.0
        diagnostics.append(
            (
                MetricId.ABSTENTION_INDICATOR,
                abstained,
                MetricUnit("boolean"),
                MetricDirection.DESCRIPTIVE,
            )
        )
        zero_cap_mask: NDArray[np.bool_] = action.problem.coordinate_caps == 0.0
        null_node_count = float(np.sum(zero_cap_mask))
        diagnostics.append(
            (
                MetricId.NULL_NODE_COUNT,
                null_node_count,
                MetricUnit("count"),
                MetricDirection.DESCRIPTIVE,
            )
        )
        diagnostics.append(
            (
                MetricId.ORBIT_SIZE,
                float(action.problem.blocks.orbit_size),
                MetricUnit("count"),
                MetricDirection.DESCRIPTIVE,
            )
        )
    if confirmation_accepted is not None:
        diagnostics.append(
            (
                MetricId.PROPOSAL_ACCEPTANCE_RATE,
                1.0 if confirmation_accepted else 0.0,
                MetricUnit("fraction"),
                MetricDirection.HIGHER_IS_BETTER,
            )
        )
    for metric_name, metric_value, metric_unit, direction in diagnostics:
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


def load_dataset_source_packet(
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


def _packet_for_block(
    packets_by_coarse_group: Mapping[CoarseGroup, SourcePacket],
    block: CorrespondenceBlockId,
) -> SourcePacket | None:
    for key, packet in packets_by_coarse_group.items():
        if correspondence_block_id(key) == block:
            return packet
    return None


def assemble_self_response_matrix(
    blocks: PaddedBlockStructure,
    packets_by_coarse_group: Mapping[CoarseGroup, SourcePacket],
) -> ResponseMatrix:
    size = blocks.total_padded_nodes
    matrix: ResponseMatrix = np.zeros((size, size), dtype=np.float64)
    for block_index, coarse_group in enumerate(blocks.coarse_groups):
        packet = _packet_for_block(packets_by_coarse_group, coarse_group)
        if packet is None:
            continue
        block_range = blocks.block_index_range(block_index)
        block_size = block_range.stop - block_range.start
        submatrix = packet.lower_matrix()
        if submatrix.shape != (block_size, block_size):
            raise ExecutionError(
                f"source packet for {coarse_group} has shape {submatrix.shape}, "
                f"expected {(block_size, block_size)}"
            )
        matrix[block_range.start : block_range.stop, block_range.start : block_range.stop] = (
            submatrix
        )
    return matrix


def _eligible_groups_for_block(
    eligible_groups_by_coarse: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]]
    | Mapping[CorrespondenceBlockId, tuple[TransferConceptGroup, ...]],
    block: CorrespondenceBlockId,
) -> tuple[TransferConceptGroup, ...]:
    for key, groups in eligible_groups_by_coarse.items():
        if correspondence_block_id(key) == block:
            return groups
    return ()


def target_node_risks(
    blocks: PaddedBlockStructure,
    eligible_groups_by_coarse: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]]
    | Mapping[CorrespondenceBlockId, tuple[TransferConceptGroup, ...]],
    class_conditional_cross_entropy: tuple[Estimate, ...],
) -> tuple[TransferNodeRisk, ...]:
    risks: list[TransferNodeRisk] = []
    for block_index, coarse_group in enumerate(blocks.coarse_groups):
        groups = _eligible_groups_for_block(eligible_groups_by_coarse, coarse_group)
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
    eligible_groups_by_coarse: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]]
    | Mapping[CorrespondenceBlockId, tuple[TransferConceptGroup, ...]],
    n_classes: ClassCount,
) -> CurriculumMultipliers:
    values = torch.ones(n_classes, dtype=torch.float64)
    for block_index, coarse_group in enumerate(blocks.coarse_groups):
        groups = _eligible_groups_for_block(eligible_groups_by_coarse, coarse_group)
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
    confirmation_verdict_sink: MutableCell[bool] | None = None,
) -> ScoreArtifact:
    access = AccessLogger()
    access.record(ClientRole.TARGET, ResourceKind.TRAIN)
    access.record(ClientRole.TARGET, ResourceKind.CONFIRM)
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
    if confirmation_verdict_sink is not None:
        confirmation_verdict_sink.value = verdict.accepted
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
    access.record(ClientRole.TARGET, ResourceKind.TEST, transfer_finalized=True)
    validate_dynamic_access_log_scan(access.trace())
    return score_model(
        ScoringRequest(model, test.features, test.targets, LocalClassCount(n_classes))
    )


def action_sha256(action: CurriculumAction) -> Sha256Digest:
    return Sha256Digest(
        hashlib.sha256(
            ",".join(f"{value:.17g}" for value in action.coordinates).encode()
        ).hexdigest()
    )


def score_local_sir_cell(
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
        packet = load_dataset_source_packet(layout, target, seed, coarse_group)
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
            action_artifact_sha256=action_sha256(action),
        ),
        n_classes,
    )
    return score, n_classes, input_artifact_ids


def common_eligible_groups(
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
        packet = _packet_for_block(source_packets_by_coarse, coarse_group)
        if packet is None:
            continue
        block_range = blocks.block_index_range(block_index)
        source_real = blocks.source_real_counts[block_index]
        submatrix = array_selector(packet)
        if submatrix.shape != (source_real, source_real):
            raise ExecutionError(
                f"source packet for {coarse_group} has shape {submatrix.shape}, "
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
        originals = next(
            (
                groups
                for key, groups in groups_by_bucket.items()
                if correspondence_block_id(key) == bucket
            ),
            (),
        )
        for original in originals:
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
        packet = _packet_for_block(target_packets_by_coarse, coarse_group)
        if packet is None:
            continue
        block_range = blocks.block_index_range(block_index)
        target_real = blocks.target_real_counts[block_index]
        submatrix = array_selector(packet)
        if submatrix.shape != (target_real, target_real):
            raise ExecutionError(
                f"target packet for {coarse_group} has shape {submatrix.shape}, "
                f"expected {(target_real, target_real)}"
            )
        start = block_range.start
        matrix[start : start + target_real, start : start + target_real] = submatrix
    return matrix


def score_matched_resource_rectangular_cell(
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
    common = common_eligible_groups(source, target, source_materialized, target_materialized)
    if common is None:
        return None
    source_eligible, target_eligible = common
    common_coarse = tuple(target_eligible)
    blocks = cross_client_padded_blocks(source_eligible, target_eligible)
    packets_by_coarse: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    for coarse_group in common_coarse:
        packet = load_dataset_source_packet(layout, source, seed, coarse_group)
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
            action_artifact_sha256=action_sha256(action),
        ),
        n_classes,
    )
    return score, n_classes, input_artifact_ids


def score_point_correspondence_commitment_cell(
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
    common = common_eligible_groups(source, target, source_materialized, target_materialized)
    if common is None:
        return None
    source_eligible, target_eligible = common
    common_coarse = tuple(target_eligible)
    blocks = cross_client_padded_blocks(source_eligible, target_eligible)
    source_packets: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    target_packets: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    for coarse_group in common_coarse:
        source_packet = load_dataset_source_packet(layout, source, seed, coarse_group)
        if source_packet is not None:
            source_packets[coarse_group] = source_packet
        target_packet = load_dataset_source_packet(layout, target, seed, coarse_group)
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
            action_artifact_sha256=action_sha256(action),
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
    target_eligible: (
        Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]]
        | Mapping[CorrespondenceBlockId, tuple[TransferConceptGroup, ...]]
    )
    input_artifact_ids: tuple[ArtifactIdentifier, ...]
    first_packet_artifact_id: ArtifactIdentifier


def assemble_principal_action(
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
    fine_singleton: bool = False,
) -> PrincipalActionAssembly | None:
    checkpoint_path = _target_confirmatory_checkpoint_path(
        layout, target, seed, checkpoint_source_experiment
    )
    if not checkpoint_path.is_file():
        return None
    checkpoint_artifact_id = _checkpoint_artifact_id(store, checkpoint_path)
    if checkpoint_artifact_id is None:
        return None
    common = common_eligible_groups(source, target, source_materialized, target_materialized)
    if common is None:
        return None
    source_eligible_original, target_eligible_original = common
    common_coarse = tuple(target_eligible_original)
    packets_by_coarse: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    for coarse_group in common_coarse:
        packet = load_dataset_source_packet(layout, source, seed, coarse_group)
        if packet is not None:
            packets_by_coarse[coarse_group] = packet
    if not packets_by_coarse:
        return None
    source_eligible: (
        Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]]
        | Mapping[CorrespondenceBlockId, tuple[TransferConceptGroup, ...]]
    )
    target_eligible: (
        Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]]
        | Mapping[CorrespondenceBlockId, tuple[TransferConceptGroup, ...]]
    )
    if fine_singleton:
        source_by_concept: OrderedDict[OracleTransferConcept, TransferConceptGroup] = OrderedDict()
        target_by_concept: OrderedDict[OracleTransferConcept, TransferConceptGroup] = OrderedDict()
        source_local_index: OrderedDict[OracleTransferConcept, Index] = OrderedDict()
        for groups in source_eligible_original.values():
            for local_index, group in enumerate(groups):
                source_by_concept[group.concept] = group
                source_local_index[group.concept] = local_index
        for groups in target_eligible_original.values():
            for group in groups:
                target_by_concept[group.concept] = group
        common_concepts = tuple(
            concept
            for concept in OracleTransferConcept
            if concept in source_by_concept and concept in target_by_concept
        )
        if not common_concepts:
            return None
        block_ids = tuple(CorrespondenceBlockId(concept.value) for concept in common_concepts)
        unit_counts: OrderedDict[CorrespondenceBlockId, SampleCount] = OrderedDict(
            (block_id, 1) for block_id in block_ids
        )
        blocks = build_padded_block_structure(block_ids, unit_counts, unit_counts)
        source_eligible = OrderedDict(
            (CorrespondenceBlockId(concept.value), (source_by_concept[concept],))
            for concept in common_concepts
        )
        target_eligible = OrderedDict(
            (CorrespondenceBlockId(concept.value), (target_by_concept[concept],))
            for concept in common_concepts
        )
        node_count = len(common_concepts)
        lower_matrix = np.zeros((node_count, node_count), dtype=np.float64)
        upper_matrix = np.zeros((node_count, node_count), dtype=np.float64)
        for index, concept in enumerate(common_concepts):
            coarse_group, _, _ = TRANSFER_ONTOLOGY[concept]
            packet = packets_by_coarse.get(coarse_group)
            if packet is None:
                continue
            local_index = source_local_index[concept]
            lower = packet.lower_matrix()
            upper = packet.upper_matrix()
            if local_index < lower.shape[0] and local_index < upper.shape[0]:
                lower_matrix[index, index] = lower[local_index, local_index]
                upper_matrix[index, index] = upper[local_index, local_index]
    elif group_bucket_of is None:
        source_eligible = source_eligible_original
        target_eligible = target_eligible_original
        groups_by_bucket: Mapping[CoarseGroup, tuple[CoarseGroup, ...]] = OrderedDict(
            (group, (group,)) for group in common_coarse
        )
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


def score_robust_action_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
    method_slug: FilesystemSlug,
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
    action_sink: MutableCell[CurriculumAction] | None = None,
    confirmation_verdict_sink: MutableCell[bool] | None = None,
    fine_singleton: bool = False,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    assembly = assemble_principal_action(
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
        fine_singleton,
    )
    if assembly is None:
        return None
    if action_sink is not None:
        action_sink.value = assembly.action
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
        action_artifact_sha256=action_sha256(assembly.action),
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
            confirmation_verdict_sink=confirmation_verdict_sink,
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


def solve_fedorbit_exact_sparse_action(
    problem: RobustActionProblem,
    seed: RandomSeed,
    certified_value_sink: MutableCell[Score] | None = None,
) -> CurriculumAction | None:
    del seed
    solution = solve_robust_action(problem)
    if certified_value_sink is not None:
        certified_value_sink.value = solution.certified_robust_value
    return solution.selected_action


def _solve_generic_exact_qap_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    del seed
    outcome = solve_robust_action_qap(problem)
    if outcome.certified_solution is None:
        return None
    return outcome.certified_solution.certified_action


def solve_exact_map_oracle_action(
    problem: RobustActionProblem,
    seed: RandomSeed,
    certified_value_sink: MutableCell[Score] | None = None,
) -> CurriculumAction | None:
    del seed
    identity = BlockCorrespondence.lexicographically_smallest(problem.blocks)
    committed_matrix = identity.permute_response_matrix(problem.lower_response_matrix)
    solution = optimize_against_fixed_matrix(problem, committed_matrix)
    if certified_value_sink is not None:
        certified_value_sink.value = solution.objective_value
    return solution.selected_action


def score_fedorbit_exact_sparse_solver_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("fedorbit-exact-sparse-solver"),
        solve_fedorbit_exact_sparse_action,
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
    access = AccessLogger()
    access.record(ClientRole.TARGET, ResourceKind.TRAIN)
    lifecycle = PreTestLifecycle()
    lifecycle.complete_phase(PreTestPhase.SOURCE_SELECTION_FINALIZED)
    lifecycle.complete_phase(PreTestPhase.ACTION_FINALIZED)
    lifecycle.complete_phase(PreTestPhase.CONFIRMATION_DECISION_FINALIZED)
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
    lifecycle.complete_phase(PreTestPhase.ASSIMILATION_SETTLED)
    lifecycle.complete_phase(PreTestPhase.PRE_TEST_ARTIFACTS_COMMITTED)
    lifecycle.open_test()
    lifecycle.assert_opened()
    access.record(ClientRole.TARGET, ResourceKind.TEST, transfer_finalized=True)
    validate_dynamic_access_log_scan(access.trace())
    return score_model(
        ScoringRequest(model, test.features, test.targets, LocalClassCount(n_classes))
    )


def score_fedorbit_without_confirmation_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("fedorbit-without-confirmation"),
        solve_fedorbit_exact_sparse_action,
        _settle_without_confirmation_and_score,
    )


def score_generic_exact_qap_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("generic-exact-qap"),
        _solve_generic_exact_qap_action,
    )


def score_exact_map_oracle_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("exact-map-oracle"),
        solve_exact_map_oracle_action,
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


def score_coarse_block_mean_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("coarse-block-mean"),
        solve_coarse_block_mean_action,
    )


def score_coarse_block_min_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("coarse-block-min"),
        solve_coarse_block_min_action,
    )


def score_orbit_mean_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("orbit-mean"),
        solve_orbit_mean_action,
    )


def score_coupling_destroyed_fedorbit_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("coupling-destroyed-fedorbit"),
        solve_coupling_destroyed_action,
    )


def score_local_sir_cell_adapter(
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
    return score_local_sir_cell(store, layout, target, target_materialized, seed, device)


def _confirm_assimilate_score_capturing_verdict(
    verdicts: MutableCell[ConfirmationVerdict],
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
        access = AccessLogger()
        access.record(ClientRole.TARGET, ResourceKind.TRAIN)
        access.record(ClientRole.TARGET, ResourceKind.CONFIRM)
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
        verdicts.value = verdict
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
        access.record(ClientRole.TARGET, ResourceKind.TEST, transfer_finalized=True)
        validate_dynamic_access_log_scan(access.trace())
        return score_model(
            ScoringRequest(model, test.features, test.targets, LocalClassCount(n_classes))
        )

    return settle_and_score


def score_fedorbit_with_confirmation_verdict_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
    verdicts: MutableCell[ConfirmationVerdict],
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("fedorbit-exact-sparse-solver"),
        solve_fedorbit_exact_sparse_action,
        _confirm_assimilate_score_capturing_verdict(verdicts),
    )


def score_local_only_cell_adapter(
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
    scored = score_local_only_cell(store, layout, target, target_materialized, seed, device)
    if scored is None:
        return None
    score, n_classes, artifact_id = scored
    return score, n_classes, (artifact_id,)


def solve_fedorbit_exact_sparse_action_at_support(
    support_limit: SupportCount,
) -> Callable[..., CurriculumAction | None]:
    def solve(
        problem: RobustActionProblem,
        seed: RandomSeed,
        certified_value_sink: list[Score] | None = None,
    ) -> CurriculumAction | None:
        del seed
        solution = solve_robust_action(problem, support_limit)
        if certified_value_sink is not None:
            certified_value_sink.append(solution.certified_robust_value)
        return solution.selected_action

    return solve


def solve_dense_ccp_fallback_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    outcome = solve_dense_ccp(
        problem, seed, SemanticCoordinates(f"sparsity-and-dense-fallback:{seed}")
    )
    return outcome.selected_action


def solve_matched_resource_rectangular_action(
    problem: RobustActionProblem,
    seed: RandomSeed,
    certified_value_sink: MutableCell[Score] | None = None,
) -> CurriculumAction | None:
    del seed
    hull = build_rectangular_hull(
        problem.blocks, problem.lower_response_matrix, problem.upper_response_matrix
    )
    solution = optimize_against_fixed_matrix(problem, hull.lower_bounds)
    if certified_value_sink is not None:
        certified_value_sink.value = solution.objective_value
    return solution.selected_action


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


def semantic_partition_bucket_of(
    partition: str | tuple[str, ...],
) -> Mapping[CoarseGroup, CoarseGroup] | None:
    if partition == SemanticPartitionId.PRINCIPAL_THREE_COARSE_GROUPS:
        return _SEMANTIC_PARTITION_PRINCIPAL
    if partition == SemanticPartitionId.ONE_ATTACK_SUPERGROUP:
        return _SEMANTIC_PARTITION_SUPERGROUP
    if isinstance(partition, tuple) and set(partition) == {
        "Disruption or Exploitation",
        "Access and Discovery",
    }:
        return _SEMANTIC_PARTITION_MERGE
    return None


def semantic_partition_label(partition: str | tuple[str, ...]) -> EvaluationConditionName:
    text = partition if isinstance(partition, str) else "|".join(partition)
    return EvaluationConditionName(text)


def response_scale_perturbation(
    scale: ScaleFactor,
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


def ci_half_width_perturbation(
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


def response_heterogeneity_perturbation(
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


WeakSignalPerturbation = Callable[
    [PaddedBlockStructure, ResponseMatrix, ResponseMatrix], tuple[ResponseMatrix, ResponseMatrix]
]
