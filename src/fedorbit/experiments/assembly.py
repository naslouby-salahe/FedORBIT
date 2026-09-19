from __future__ import annotations

import math
import statistics
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from fedorbit.config.loading import active_config
from fedorbit.datasets.common import (
    FieldRole,
)
from fedorbit.datasets.materialization import (
    MaterializedClient,
    SplitTensors,
    TransferConceptGroup,
    transfer_concept_groups,
)
from fedorbit.datasets.ontology import TRANSFER_ONTOLOGY
from fedorbit.infrastructure.artifacts import (
    ArtifactAccessContext,
    ArtifactPayloadRequest,
    ArtifactStore,
    ExecutionError,
)
from fedorbit.infrastructure.runtime import (
    RandomSeed,
)
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
    experiment_workspace,
)
from fedorbit.interface import (
    validate_disjoint_feature_namespaces,
    validate_no_cross_client_entity_ids,
    validate_no_cross_client_timestamp_pairing,
)
from fedorbit.learning.checkpoints import load_base_checkpoint
from fedorbit.learning.pilot import (
    create_classifier,
)
from fedorbit.learning.scoring import LocalClassCount, ScoringRequest, score_model
from fedorbit.learning.training import (
    BaseCheckpoint,
)
from fedorbit.methods.target import (
    CurriculumMultipliers,
    TargetImportance,
    TargetImportanceError,
    TransferNodeRisk,
    build_target_importance,
)
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    PaddedBlockStructure,
    ResponseMatrix,
    build_padded_block_structure,
    correspondence_block_id,
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
    ArtifactDirectorySegment,
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactPath,
    ArtifactStage,
    ArtifactType,
    CheckpointDirectorySegment,
    CheckpointFileName,
    ClassCount,
    ClientRole,
    CoarseGroup,
    CorrespondenceBlockId,
    DatasetId,
    Estimate,
    EvaluationConditionName,
    ExperimentName,
    Index,
    OracleTransferConcept,
    OverwritePolicy,
    PairSeedIneligibilityReason,
    SampleCount,
    ScaleFactor,
    ScientificSubsystem,
    SemanticCoordinateText,
    SemanticPartitionId,
    SemanticPartitionSpecification,
    SerializedPacket,
    Split,
    StableJsonPayload,
    StorageLayoutSegment,
    StrictResourceValidity,
    StrictResourceViolationError,
    TabularColumnName,
    stable_json,
)


def persist_or_reuse_target_importance(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    target: DatasetId,
    seed: RandomSeed,
    node_risks: tuple[TransferNodeRisk, ...],
    overwrite_policy: OverwritePolicy,
    upstream_artifact_ids: ArtifactIdentifiers,
) -> TargetImportance | None:
    try:
        importance = build_target_importance(node_risks)
    except TargetImportanceError:
        return None
    payload = cast(
        StableJsonPayload,
        OrderedDict(
            (str(node_index), weight)
            for node_index, weight in importance.weights_by_node_index.items()
        ),
    )
    destination = (
        experiment_workspace(layout, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
        / StorageLayoutSegment.ARTIFACTS
        / ArtifactDirectorySegment.FITTED
        / "importance"
        / target.value
        / f"seed-{seed}.json"
    )
    serialized = SerializedPacket(stable_json(payload))
    coordinates = SemanticCoordinateText(
        stable_json(cast(StableJsonPayload, OrderedDict(dataset=target.value, seed=seed)))
    )
    store.register_artifact_payload(
        ArtifactPayloadRequest(
            artifact_type=ArtifactType.TARGET_IMPORTANCE,
            coordinates=cast(StableJsonPayload, OrderedDict(dataset=target.value, seed=seed)),
            semantic_coordinates=coordinates,
            stage=ArtifactStage.TARGET_IMPORTANCE,
            payload_path=ArtifactPath(destination),
            serialized=serialized,
            upstream_artifact_ids=upstream_artifact_ids,
            access=ArtifactAccessContext(subsystem=ScientificSubsystem.TARGET_IMPORTANCE),
        ),
        overwrite_policy,
    )
    return importance


def target_confirmatory_checkpoint_path(
    layout: WorkspaceLayout,
    target: DatasetId,
    seed: RandomSeed,
    checkpoint_source_experiment: ExperimentName = ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT,
) -> Path:
    return (
        experiment_workspace(layout, checkpoint_source_experiment)
        / StorageLayoutSegment.CHECKPOINTS
        / CheckpointDirectorySegment.TRAINING
        / target.value
        / f"seed-{seed}"
        / CheckpointFileName.CHECKPOINT
    )


def resolve_checkpoint_artifact_id(
    store: ArtifactStore, checkpoint_path: Path
) -> ArtifactIdentifier | None:
    target_path = str(checkpoint_path)
    for manifest in store.all_manifests():
        if manifest.payload_paths == (target_path,):
            return manifest.artifact_id
    return None


def eligible_groups_by_coarse(
    dataset: DatasetId,
    materialized: MaterializedClient,
    role: ClientRole,
) -> Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]]:
    eligible_by_group: OrderedDict[CoarseGroup, list[TransferConceptGroup]] = OrderedDict()
    for group in transfer_concept_groups(dataset, materialized):
        eligible = group.source_eligible if role is ClientRole.SOURCE else group.target_eligible
        if eligible:
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
        / StorageLayoutSegment.ARTIFACTS
        / ArtifactDirectorySegment.PACKETS
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
    target_eligible_all = eligible_groups_by_coarse(target, target_materialized, ClientRole.TARGET)
    source_eligible_all = eligible_groups_by_coarse(source, source_materialized, ClientRole.SOURCE)
    common_coarse = tuple(group for group in target_eligible_all if group in source_eligible_all)
    if not common_coarse:
        return None
    source_eligible = OrderedDict((group, source_eligible_all[group]) for group in common_coarse)
    target_eligible = OrderedDict((group, target_eligible_all[group]) for group in common_coarse)
    return source_eligible, target_eligible


@dataclass(frozen=True, slots=True)
class PairSeedStructure:
    source_eligible: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]]
    target_eligible: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]]
    ineligibility_reason: PairSeedIneligibilityReason | None

    @property
    def is_eligible(self) -> bool:
        return self.ineligibility_reason is None


def pair_seed_structure(
    source_eligible: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]],
    target_eligible: Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]],
    available_response_blocks: frozenset[CoarseGroup],
    strict_resource_validity: StrictResourceValidity,
) -> PairSeedStructure:
    common_coarse = tuple(
        coarse
        for coarse in target_eligible
        if coarse in source_eligible and coarse in available_response_blocks
    )
    selected_source = OrderedDict((coarse, source_eligible[coarse]) for coarse in common_coarse)
    selected_target = OrderedDict((coarse, target_eligible[coarse]) for coarse in common_coarse)
    if not strict_resource_validity:
        return PairSeedStructure(
            selected_source,
            selected_target,
            PairSeedIneligibilityReason.STRICT_RESOURCE_VIOLATION,
        )
    if not common_coarse:
        return PairSeedStructure(
            selected_source,
            selected_target,
            PairSeedIneligibilityReason.NO_SHARED_ELIGIBLE_COARSE_GROUP,
        )
    support = active_config().scientific.transfer_support
    target_concept_count = sum(len(groups) for groups in selected_target.values())
    if target_concept_count < support.minimum_actionable_target_concepts:
        return PairSeedStructure(
            selected_source,
            selected_target,
            PairSeedIneligibilityReason.INSUFFICIENT_ACTIONABLE_TARGET_CONCEPTS,
        )
    if not any(
        len(groups) >= support.minimum_nontrivial_block_size for groups in selected_target.values()
    ):
        return PairSeedStructure(
            selected_source,
            selected_target,
            PairSeedIneligibilityReason.NO_NONTRIVIAL_TARGET_BLOCK,
        )
    if not any(
        len(groups) >= support.minimum_nontrivial_block_size for groups in selected_source.values()
    ):
        return PairSeedStructure(
            selected_source,
            selected_target,
            PairSeedIneligibilityReason.NO_NONTRIVIAL_SOURCE_RESPONSE_BLOCK,
        )
    return PairSeedStructure(selected_source, selected_target, None)


def strict_pair_resource_validity(
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
) -> StrictResourceValidity:
    source_identity_columns = frozenset(
        column
        for column, role in source_materialized.schema.roles.items()
        if role is FieldRole.FORBIDDEN_IDENTITY
    )
    target_identity_columns = frozenset(
        column
        for column, role in target_materialized.schema.roles.items()
        if role is FieldRole.FORBIDDEN_IDENTITY
    )
    source_timestamp_columns: frozenset[TabularColumnName] = (
        frozenset()
        if source_materialized.schema.timestamp_column is None
        else frozenset({source_materialized.schema.timestamp_column})
    )
    target_timestamp_columns: frozenset[TabularColumnName] = (
        frozenset()
        if target_materialized.schema.timestamp_column is None
        else frozenset({target_materialized.schema.timestamp_column})
    )
    try:
        validate_disjoint_feature_namespaces(
            frozenset(source_materialized.feature_names),
            frozenset(target_materialized.feature_names),
        )
        validate_no_cross_client_entity_ids(source_identity_columns, target_identity_columns)
        validate_no_cross_client_timestamp_pairing(
            source_timestamp_columns,
            target_timestamp_columns,
        )
    except StrictResourceViolationError:
        return StrictResourceValidity(False)
    return StrictResourceValidity(True)


def assess_pair_seed_structure(
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
) -> PairSeedStructure:
    common = common_eligible_groups(
        source,
        target,
        source_materialized,
        target_materialized,
    )
    if common is None:
        return PairSeedStructure(
            OrderedDict(),
            OrderedDict(),
            PairSeedIneligibilityReason.NO_SHARED_ELIGIBLE_COARSE_GROUP,
        )
    source_eligible, target_eligible = common
    available_response_blocks = frozenset(
        coarse
        for coarse in target_eligible
        if load_dataset_source_packet(layout, source, seed, coarse) is not None
    )
    return pair_seed_structure(
        source_eligible,
        target_eligible,
        available_response_blocks,
        strict_pair_resource_validity(source_materialized, target_materialized),
    )


def _block_keyed_eligible_groups(
    eligible_by_coarse: (
        Mapping[CoarseGroup, tuple[TransferConceptGroup, ...]]
        | Mapping[CorrespondenceBlockId, tuple[TransferConceptGroup, ...]]
    ),
) -> Mapping[CorrespondenceBlockId, tuple[TransferConceptGroup, ...]]:
    return OrderedDict(
        (correspondence_block_id(group), groups) for group, groups in eligible_by_coarse.items()
    )


def registered_exact_correspondence(
    blocks: PaddedBlockStructure,
    source_eligible_by_coarse: Mapping[CorrespondenceBlockId, tuple[TransferConceptGroup, ...]],
    target_eligible_by_coarse: Mapping[CorrespondenceBlockId, tuple[TransferConceptGroup, ...]],
    source_packets_by_coarse: Mapping[CoarseGroup, SourcePacket],
    seed: RandomSeed,
) -> BlockCorrespondence:
    images: list[Index] = []
    for block_index, block_id in enumerate(blocks.coarse_groups):
        coarse_group = CoarseGroup(str(block_id))
        block_range = blocks.block_index_range(block_index)
        block_key = correspondence_block_id(coarse_group)
        source_groups = source_eligible_by_coarse.get(block_key, ())
        target_groups = target_eligible_by_coarse.get(block_key, ())
        packet = source_packets_by_coarse.get(coarse_group)
        source_concept_positions: OrderedDict[OracleTransferConcept, Index] = OrderedDict()
        if packet is not None:
            positions = packet.registered_node_order(seed).anonymous_position_of_semantic_index()
            for semantic_index, group in enumerate(source_groups):
                if semantic_index < len(positions):
                    source_concept_positions[group.concept] = positions[semantic_index]
        assigned: OrderedDict[Index, Index] = OrderedDict()
        for target_offset, group in enumerate(target_groups):
            source_offset = source_concept_positions.get(group.concept)
            if source_offset is None or source_offset >= len(block_range):
                continue
            source_node = block_range.start + source_offset
            if source_node in assigned.values():
                continue
            assigned[block_range.start + target_offset] = source_node
        taken = set(assigned.values())
        remaining_sources = [node for node in block_range if node not in taken]
        remaining_targets = [node for node in block_range if node not in assigned]
        for target_node, source_node in zip(remaining_targets, remaining_sources, strict=True):
            assigned[target_node] = source_node
        images.extend(assigned[node] for node in block_range)
    return BlockCorrespondence(blocks=blocks, images=tuple(images))


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
    uses_registered_exact_map: bool = False,
) -> PrincipalActionAssembly | None:
    checkpoint_path = target_confirmatory_checkpoint_path(
        layout, target, seed, checkpoint_source_experiment
    )
    if not checkpoint_path.is_file():
        return None
    checkpoint_artifact_id = resolve_checkpoint_artifact_id(store, checkpoint_path)
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
    pair_structure = pair_seed_structure(
        source_eligible_original,
        target_eligible_original,
        frozenset(packets_by_coarse),
        strict_pair_resource_validity(source_materialized, target_materialized),
    )
    if not pair_structure.is_eligible:
        return None
    source_eligible_original = pair_structure.source_eligible
    target_eligible_original = pair_structure.target_eligible
    common_coarse = tuple(target_eligible_original)
    packets_by_coarse = OrderedDict((coarse, packets_by_coarse[coarse]) for coarse in common_coarse)
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
            semantic_index = source_local_index[concept]
            anonymous_positions = packet.registered_node_order(
                seed
            ).anonymous_position_of_semantic_index()
            local_index = anonymous_positions[semantic_index]
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
    if uses_registered_exact_map and not fine_singleton:
        exact_correspondence = registered_exact_correspondence(
            blocks,
            _block_keyed_eligible_groups(source_eligible),
            _block_keyed_eligible_groups(target_eligible),
            packets_by_coarse,
            seed,
        )
        lower_matrix = exact_correspondence.permute_response_matrix(lower_matrix)
        upper_matrix = exact_correspondence.permute_response_matrix(upper_matrix)
    checkpoint = load_base_checkpoint(checkpoint_path)
    n_classes = target_materialized.class_manifest.class_count
    train = target_materialized.splits[Split.TRAIN]
    meta = target_materialized.splits[Split.META]
    confirm = target_materialized.splits[Split.CONFIRM]
    test = target_materialized.splits[Split.TEST]
    model = create_classifier(
        target,
        target_materialized.feature_count,
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
    target_importance = persist_or_reuse_target_importance(
        store,
        layout,
        target,
        seed,
        node_risks,
        OverwritePolicy.REUSE,
        (checkpoint_artifact_id,),
    )
    if target_importance is None:
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
    partition: SemanticPartitionSpecification,
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


def semantic_partition_label(
    partition: SemanticPartitionSpecification,
) -> EvaluationConditionName:
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
