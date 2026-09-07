from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import cast

from pydantic import Field

from fedorbit.config.models import FrozenModel
from fedorbit.types import (
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    CoarseGroup,
    DatasetId,
    ExperimentName,
    Index,
    OracleTransferConcept,
    RandomSeed,
    StableJsonPayload,
    TerminalState,
    TransferMethod,
    stable_json,
)

NATIVE_CLASS_IDS_FIELD = "native_local_class_ids"
FINE_CONCEPT_FIELD = "fine_concept"


class CompletionManifest(FrozenModel):
    schema_version: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    semantic_experiment_coordinates: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    producer_stage: ArtifactStage
    terminal_state: TerminalState
    dependency_fingerprint_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    upstream_artifact_ids: tuple[str, ...] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    mandatory_artifact_paths: tuple[str, ...] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    mandatory_artifact_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    scientific_configuration_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    relevant_code_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    material_runtime_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    upstream_lineage: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    completion_validation_state: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    completion_written_last: bool #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    completion_manifest_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this


class ReusableArtifactManifest(FrozenModel):
    artifact_id: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    artifact_type: ArtifactType
    semantic_producer_coordinates: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    producer_stage: ArtifactStage
    dependency_fingerprint_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    upstream_artifact_ids: tuple[str, ...] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    applicable_configuration_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    relevant_code_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    material_runtime_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    payload_paths: tuple[str, ...] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    payload_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    schema_version: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    created_git_commit: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    created_environment_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    state: ArtifactState
    completion_required: bool = False
    completion_manifest_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this


class DatasetManifest(FrozenModel):
    dataset: DatasetId
    component: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    raw_files: tuple[str, ...] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    raw_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    raw_counts: Mapping[str, int] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    schema_version: str = Field(serialization_alias="schema", validation_alias="schema") #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    adapter_feature_order: tuple[str, ...] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    adapter_feature_roles: Mapping[str, str] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    accepted_schema_aliases: tuple[str, ...] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    adapter_adaptations: tuple[str, ...] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    timestamp_field: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    timestamp_range: tuple[str, str] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    duplicate_counts: Mapping[str, int] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    conflicting_duplicate_counts: Mapping[str, int] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    local_class_counts: Mapping[str, int] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    transfer_candidate_counts: Mapping[str, int] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    feature_quality: Mapping[str, str | int | float | bool | None] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    preprocessing_state: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    dependency_fingerprint_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this 
    producer_code_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this


class EligibilityCopyKind(StrEnum):
    BUILDER = "builder"
    METHOD_READABLE = "method_readable"
    ORACLE = "oracle"


class TransferEligibilityManifest(FrozenModel):
    client: DatasetId
    seed: RandomSeed
    coarse_group: CoarseGroup
    anonymous_node_id: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    native_local_class_ids: tuple[str, ...] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    present: bool
    train_count: Index
    meta_count: Index
    confirm_count: Index
    test_count: Index
    source_eligible: bool
    target_eligible: bool
    null_reason: str | None = None #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    fine_concept: OracleTransferConcept | None = None


class SemanticCellManifest(FrozenModel):
    experiment: ExperimentName
    dataset: DatasetId
    source_client: DatasetId
    target_client: DatasetId
    directed_pair: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    method: TransferMethod
    condition: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    support: Index
    seed: RandomSeed
    scientific_configuration_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    dependency_fingerprint_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    producer_stage: ArtifactStage
    upstream_artifact_ids: tuple[str, ...] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    dataset_manifest_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    split_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    preprocessing_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    source_checkpoint_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    response_packet_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    target_checkpoint_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    importance_vector_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    resource_manifest_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    relevant_code_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    material_runtime_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    git_commit: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    git_dirty: bool
    environment_sha256: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    state: ArtifactState
    state_reason: str | None = None #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this


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


def _sha256(payload: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: payload)
            ) -> str: #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (output return)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path #TODO: optional xxhash fast change-detection pass before the SHA-256 hash of large raw files
                ) -> str: #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (output return)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_fingerprint(
    coordinates: StableJsonPayload,
    upstream_artifact_ids: tuple[str, ...], #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: upstream_artifact_ids)
    configuration_sha256: str, #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: configuration_sha256)
    code_sha256: str, #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: code_sha256)
    runtime_sha256: str, #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: runtime_sha256)
) -> str: #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (output return)
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
    artifact_type: str, #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: artifact_type)
    coordinates: StableJsonPayload,
    fingerprint_sha256: str, #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: fingerprint_sha256)
) -> str: #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (output return)
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


def completion_manifest_self_hash(manifest: CompletionManifest
                                  ) -> str: #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (output return)
    payload = stable_json(manifest.model_dump(mode="json", exclude={"completion_manifest_sha256"}))
    return _sha256(payload)
