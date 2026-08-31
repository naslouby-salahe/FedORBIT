from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import cast

from pydantic import Field

from fedorbit.config.models import FrozenModel
from fedorbit.domain.enums import ArtifactStage, ArtifactState, ArtifactType, TerminalState
from fedorbit.domain.serialization import StableJsonPayload, stable_json

NATIVE_CLASS_IDS_FIELD = "native_local_class_ids"
FINE_CONCEPT_FIELD = "fine_concept"


class CompletionManifest(FrozenModel):
    schema_version: str
    semantic_experiment_coordinates: str
    producer_stage: ArtifactStage
    terminal_state: TerminalState
    dependency_fingerprint_sha256: str
    upstream_artifact_ids: tuple[str, ...]
    mandatory_artifact_paths: tuple[str, ...]
    mandatory_artifact_sha256: str
    scientific_configuration_sha256: str
    relevant_code_sha256: str
    material_runtime_sha256: str
    upstream_lineage: str
    completion_validation_state: str
    completion_written_last: bool
    completion_manifest_sha256: str


class ReusableArtifactManifest(FrozenModel):
    artifact_id: str
    artifact_type: ArtifactType
    semantic_producer_coordinates: str
    producer_stage: ArtifactStage
    dependency_fingerprint_sha256: str
    upstream_artifact_ids: tuple[str, ...]
    applicable_configuration_sha256: str
    relevant_code_sha256: str
    material_runtime_sha256: str
    payload_paths: tuple[str, ...]
    payload_sha256: str
    schema_version: str
    created_git_commit: str
    created_environment_sha256: str
    state: ArtifactState
    completion_required: bool = False
    completion_manifest_sha256: str


class DatasetManifest(FrozenModel):
    dataset: str
    component: str
    raw_files: tuple[str, ...]
    raw_sha256: str
    raw_counts: Mapping[str, int]
    schema_version: str = Field(serialization_alias="schema", validation_alias="schema")
    adapter_feature_order: tuple[str, ...]
    adapter_feature_roles: Mapping[str, str]
    accepted_schema_aliases: tuple[str, ...]
    adapter_adaptations: tuple[str, ...]
    timestamp_field: str
    timestamp_range: tuple[str, str]
    duplicate_counts: Mapping[str, int]
    conflicting_duplicate_counts: Mapping[str, int]
    local_class_counts: Mapping[str, int]
    transfer_candidate_counts: Mapping[str, int]
    feature_quality: Mapping[str, str | int | float | bool | None]
    preprocessing_state: str
    dependency_fingerprint_sha256: str
    producer_code_sha256: str


class EligibilityCopyKind(StrEnum):
    BUILDER = "builder"
    METHOD_READABLE = "method_readable"
    ORACLE = "oracle"


class TransferEligibilityManifest(FrozenModel):
    client: str
    seed: int
    coarse_group: str
    anonymous_node_id: str
    native_local_class_ids: tuple[str, ...]
    present: bool
    train_count: int
    meta_count: int
    confirm_count: int
    test_count: int
    source_eligible: bool
    target_eligible: bool
    null_reason: str | None = None
    fine_concept: str | None = None


class SemanticCellManifest(FrozenModel):
    experiment: str
    dataset: str
    source_client: str
    target_client: str
    directed_pair: str
    method: str
    condition: str
    support: int
    seed: int
    scientific_configuration_sha256: str
    dependency_fingerprint_sha256: str
    producer_stage: ArtifactStage


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


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_fingerprint(
    coordinates: StableJsonPayload,
    upstream_artifact_ids: tuple[str, ...],
    configuration_sha256: str,
    code_sha256: str,
    runtime_sha256: str,
) -> str:
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
    return _sha256(payload)


def artifact_id(
    artifact_type: str,
    coordinates: StableJsonPayload,
    fingerprint_sha256: str,
) -> str:
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
    return _sha256(payload)


def completion_manifest_self_hash(manifest: CompletionManifest) -> str:
    payload = stable_json(manifest.model_dump(mode="json", exclude={"completion_manifest_sha256"}))
    return _sha256(payload)
