from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from typing import cast

from pydantic import Field

from fedorbit.config.models import FrozenModel
from fedorbit.datasets.common import FieldRole
from fedorbit.infrastructure.environment import HardwareIdentity
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactLineage,
    ArtifactPath,
    ArtifactPathTexts,
    ArtifactSchemaVersion,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    ClientComponentName,
    CompletionValidationState,
    DatasetId,
    DatasetPreprocessingState,
    ExperimentName,
    FeatureCount,
    FieldDescription,
    FineLabel,
    GitRevision,
    Index,
    OracleTransferConcept,
    ProvenanceAccessRole,
    ProvenanceArtifactFamily,
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
    ValidationReason,
    stable_json,
)

NATIVE_CLASS_IDS_FIELD = "native_local_class_ids"
FINE_CONCEPT_FIELD = "fine_concept"


class ProvenanceFamilyEvidence(FrozenModel):
    family: ProvenanceArtifactFamily
    sha256: Sha256Digest


class ProvenanceAccessEvent(FrozenModel):
    family: ProvenanceArtifactFamily
    sha256: Sha256Digest
    role: ProvenanceAccessRole


class ArtifactProvenance(FrozenModel):
    family_evidence: tuple[ProvenanceFamilyEvidence, ...] = ()
    access_trace: tuple[ProvenanceAccessEvent, ...] = ()
    hardware: HardwareIdentity | None = None

    @classmethod
    def observed(cls, hardware: HardwareIdentity | None = None) -> ArtifactProvenance:
        return cls(family_evidence=(), access_trace=(), hardware=hardware)


class ArtifactLineageRecord(FrozenModel):
    upstream_artifact_ids: ArtifactIdentifiers = ()
    provenance: ArtifactProvenance | None = None

    @classmethod
    def from_lineage_text(cls, lineage: ArtifactLineage) -> ArtifactLineageRecord:
        return cls.model_validate_json(lineage)

    def lineage_text(self) -> ArtifactLineage:
        return ArtifactLineage(stable_json(cast(StableJsonPayload, self.model_dump(mode="json"))))


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
    upstream_lineage: ArtifactLineage
    completion_validation_state: CompletionValidationState
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
    raw_counts: Mapping[RawDatasetPath, Index]
    schema_version: ArtifactSchemaVersion = Field(
        serialization_alias="schema", validation_alias="schema"
    )
    adapter_feature_order: TabularColumns
    adapter_feature_roles: Mapping[TabularColumnName, FieldRole]
    accepted_schema_aliases: tuple[TabularColumnName, ...]
    adapter_adaptations: tuple[FieldDescription, ...]
    timestamp_field: TabularColumnName
    timestamp_range: TimestampRange
    duplicate_counts: Mapping[str, Index]
    conflicting_duplicate_counts: Mapping[str, Index]
    local_class_counts: Mapping[FineLabel, Index]
    transfer_candidate_counts: Mapping[OracleTransferConcept, SampleCount]
    feature_quality: FeatureQualityManifest
    preprocessing_state: DatasetPreprocessingState
    dependency_fingerprint_sha256: Sha256Digest
    producer_code_sha256: Sha256Digest


def _sha256(payload: SerializedPacket) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(payload.encode("utf-8")).hexdigest())


def artifact_id(
    artifact_type: ArtifactType,
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


def recorded_experiment(manifest: ReusableArtifactManifest) -> ExperimentName | None:
    try:
        parsed = json.loads(manifest.semantic_producer_coordinates)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("experiment")
    if not isinstance(value, str):
        return None
    try:
        return ExperimentName(value)
    except ValueError:
        return None


def build_artifact_completion(
    coordinates: SemanticCoordinateText,
    fingerprint_sha256: Sha256Digest,
    payload_path: ArtifactPath,
    payload_sha256: Sha256Digest,
    configuration_sha256: Sha256Digest,
    code_sha256: Sha256Digest,
    stage: ArtifactStage,
    upstream_artifact_ids: ArtifactIdentifiers = (),
    provenance: ArtifactProvenance | None = None,
) -> CompletionManifest:
    lineage = ArtifactLineageRecord(
        upstream_artifact_ids=upstream_artifact_ids,
        provenance=provenance,
    )
    draft = CompletionManifest.model_validate(
        OrderedDict(
            schema_version=ArtifactSchemaVersion.V1,
            semantic_experiment_coordinates=coordinates,
            producer_stage=stage,
            terminal_state=TerminalState.COMPLETED,
            dependency_fingerprint_sha256=fingerprint_sha256,
            upstream_artifact_ids=upstream_artifact_ids,
            mandatory_artifact_paths=(str(payload_path),),
            mandatory_artifact_sha256=payload_sha256,
            scientific_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            upstream_lineage=lineage.lineage_text(),
            completion_validation_state=CompletionValidationState.VALIDATED,
            completion_written_last=True,
            completion_manifest_sha256="",
        )
    )
    return draft.model_copy(
        update={"completion_manifest_sha256": completion_manifest_self_hash(draft)}
    )


def completion_manifest_self_hash(manifest: CompletionManifest) -> Sha256Digest:
    payload = stable_json(manifest.model_dump(mode="json", exclude={"completion_manifest_sha256"}))
    return _sha256(SerializedPacket(payload))
