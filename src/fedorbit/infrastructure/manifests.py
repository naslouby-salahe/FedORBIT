from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from enum import StrEnum
from typing import cast

from pydantic import Field

from fedorbit.config.models import FrozenModel
from fedorbit.types import (
    AnonymousNodeDisplayId,
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactLineage,
    ArtifactPathTexts,
    ArtifactSchemaVersion,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    ArtifactTypeName,
    ClientComponentName,
    CoarseGroup,
    DatasetId,
    DatasetPreprocessingState,
    DirectedPairName,
    EvaluationConditionName,
    ExperimentName,
    FeatureCount,
    FieldDescription,
    FineLabel,
    GitRevision,
    Index,
    ManifestValidationState,
    NonNegativeInt,
    OracleTransferConcept,
    RandomSeed,
    RawDatasetPath,
    SampleCount,
    SemanticCoordinateText,
    SerializedPacket,
    Sha256Digest,
    StableJsonPayload,
    TabularColumnName,
    TabularColumns,
    TerminalState,
    TimestampRange,
    TransferMethod,
    ValidationReason,
    stable_json,
)

NATIVE_CLASS_IDS_FIELD = "native_local_class_ids"
FINE_CONCEPT_FIELD = "fine_concept"


class CompletionManifest(FrozenModel):
    schema_version: ArtifactSchemaVersion
    semantic_experiment_coordinates: SemanticCoordinateText
    producer_stage: ArtifactStage
    terminal_state: TerminalState
    dependency_fingerprint_sha256: Sha256Digest
    upstream_artifact_ids: ArtifactIdentifiers
    mandatory_artifact_paths: ArtifactPathTexts
    mandatory_artifact_sha256: Sha256Digest
    scientific_configuration_sha256: Sha256Digest
    relevant_code_sha256: Sha256Digest
    material_runtime_sha256: Sha256Digest
    upstream_lineage: ArtifactLineage
    completion_validation_state: ManifestValidationState
    completion_written_last: bool
    completion_manifest_sha256: Sha256Digest


class ReusableArtifactManifest(FrozenModel):
    artifact_id: ArtifactIdentifier
    artifact_type: ArtifactType
    semantic_producer_coordinates: SemanticCoordinateText
    producer_stage: ArtifactStage
    dependency_fingerprint_sha256: Sha256Digest
    upstream_artifact_ids: ArtifactIdentifiers
    applicable_configuration_sha256: Sha256Digest
    relevant_code_sha256: Sha256Digest
    material_runtime_sha256: Sha256Digest
    payload_paths: ArtifactPathTexts
    payload_sha256: Sha256Digest
    schema_version: ArtifactSchemaVersion
    created_git_commit: GitRevision
    created_environment_sha256: Sha256Digest
    state: ArtifactState
    completion_required: bool = False
    completion_manifest_sha256: Sha256Digest


class FeatureQualityManifest(FrozenModel):
    dropped_feature_count: Index
    candidate_count_before_filtering: FeatureCount
    client_invalid: bool
    client_invalid_reason: ValidationReason | None = None


class DatasetManifest(FrozenModel):
    dataset: DatasetId
    component: ClientComponentName
    raw_files: tuple[RawDatasetPath, ...]
    raw_sha256: Sha256Digest
    raw_counts: Mapping[RawDatasetPath, NonNegativeInt]
    schema_version: ArtifactSchemaVersion = Field(
        serialization_alias="schema", validation_alias="schema"
    )
    adapter_feature_order: TabularColumns
    adapter_feature_roles: Mapping[TabularColumnName, str]
    accepted_schema_aliases: tuple[TabularColumnName, ...]
    adapter_adaptations: tuple[FieldDescription, ...]
    timestamp_field: TabularColumnName
    timestamp_range: TimestampRange
    duplicate_counts: Mapping[str, NonNegativeInt]
    conflicting_duplicate_counts: Mapping[str, NonNegativeInt]
    local_class_counts: Mapping[str, NonNegativeInt]
    transfer_candidate_counts: Mapping[str, SampleCount]
    feature_quality: FeatureQualityManifest
    preprocessing_state: DatasetPreprocessingState
    dependency_fingerprint_sha256: Sha256Digest
    producer_code_sha256: Sha256Digest


class EligibilityCopyKind(StrEnum):
    BUILDER = "builder"
    METHOD_READABLE = "method_readable"
    ORACLE = "oracle"


class TransferEligibilityManifest(FrozenModel):
    client: DatasetId
    seed: RandomSeed
    coarse_group: CoarseGroup
    anonymous_node_id: AnonymousNodeDisplayId
    native_local_class_ids: tuple[FineLabel, ...]
    present: bool
    train_count: Index
    meta_count: Index
    confirm_count: Index
    test_count: Index
    source_eligible: bool
    target_eligible: bool
    null_reason: ValidationReason | None = None
    fine_concept: OracleTransferConcept | None = None


class SemanticCellManifest(FrozenModel):
    experiment: ExperimentName
    dataset: DatasetId
    source_client: DatasetId
    target_client: DatasetId
    directed_pair: DirectedPairName
    method: TransferMethod
    condition: EvaluationConditionName
    support: Index
    seed: RandomSeed
    scientific_configuration_sha256: Sha256Digest
    dependency_fingerprint_sha256: Sha256Digest
    producer_stage: ArtifactStage
    upstream_artifact_ids: ArtifactIdentifiers
    dataset_manifest_sha256: Sha256Digest
    split_sha256: Sha256Digest
    preprocessing_sha256: Sha256Digest
    source_checkpoint_sha256: Sha256Digest
    response_packet_sha256: Sha256Digest
    target_checkpoint_sha256: Sha256Digest
    importance_vector_sha256: Sha256Digest
    resource_manifest_sha256: Sha256Digest
    relevant_code_sha256: Sha256Digest
    material_runtime_sha256: Sha256Digest
    git_commit: GitRevision
    git_dirty: bool
    environment_sha256: Sha256Digest
    state: ArtifactState
    state_reason: ValidationReason | None = None


def eligibility_copy(
    manifest: TransferEligibilityManifest,
    kind: EligibilityCopyKind,
) -> TransferEligibilityManifest:
    if kind == EligibilityCopyKind.METHOD_READABLE:
        return manifest.model_copy(
            update=OrderedDict(((NATIVE_CLASS_IDS_FIELD, ()), (FINE_CONCEPT_FIELD, None)))
        )
    if kind == EligibilityCopyKind.ORACLE:
        if manifest.fine_concept is None:
            raise ValueError("oracle eligibility copy requires the fine concept")
        return manifest.model_copy(update=OrderedDict(((NATIVE_CLASS_IDS_FIELD, ()),)))
    return manifest


def _sha256(payload: SerializedPacket) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(payload.encode("utf-8")).hexdigest())


def dependency_fingerprint(
    coordinates: StableJsonPayload,
    upstream_artifact_ids: ArtifactIdentifiers,
    configuration_sha256: Sha256Digest,
    code_sha256: Sha256Digest,
    runtime_sha256: Sha256Digest,
) -> Sha256Digest:
    payload = stable_json(
        cast(
            StableJsonPayload,
            OrderedDict(
                coordinates=coordinates,
                upstream_artifact_ids=list(upstream_artifact_ids),
                configuration_sha256=configuration_sha256,
                code_sha256=code_sha256,
                runtime_sha256=runtime_sha256,
            ),
        )
    )
    return _sha256(SerializedPacket(payload))


def artifact_id(
    artifact_type: ArtifactTypeName,
    coordinates: StableJsonPayload,
    fingerprint_sha256: Sha256Digest,
) -> ArtifactIdentifier:
    payload = stable_json(
        cast(
            StableJsonPayload,
            OrderedDict(
                artifact_type=artifact_type,
                coordinates=coordinates,
                dependency_fingerprint_sha256=fingerprint_sha256,
            ),
        )
    )
    return ArtifactIdentifier(_sha256(SerializedPacket(payload)))


def completion_manifest_self_hash(manifest: CompletionManifest) -> Sha256Digest:
    payload = stable_json(manifest.model_dump(mode="json", exclude={"completion_manifest_sha256"}))
    return _sha256(SerializedPacket(payload))
