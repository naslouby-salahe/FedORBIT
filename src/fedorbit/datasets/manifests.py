from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel as PydanticModel
from pydantic import Field

NATIVE_CLASS_IDS_FIELD = "native_local_class_ids"
FINE_CONCEPT_FIELD = "fine_concept"


class FrozenModel(PydanticModel):
    model_config = {"frozen": True, "extra": "forbid"}


class DatasetManifest(FrozenModel):
    dataset: str
    component: str
    raw_files: tuple[str, ...]
    raw_sha256: str
    raw_counts: dict[str, int]
    schema_version: str = Field(serialization_alias="schema", validation_alias="schema")
    adapter_feature_order: tuple[str, ...]
    adapter_feature_roles: dict[str, str]
    accepted_schema_aliases: tuple[str, ...]
    adapter_adaptations: tuple[str, ...]
    timestamp_field: str
    timestamp_range: tuple[str, str]
    duplicate_counts: dict[str, int]
    conflicting_duplicate_counts: dict[str, int]
    local_class_counts: dict[str, int]
    transfer_candidate_counts: dict[str, int]
    feature_quality: dict[str, str | int | float | bool | None]
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
    producer_stage: str


def eligibility_copy(
    manifest: TransferEligibilityManifest, kind: EligibilityCopyKind
) -> TransferEligibilityManifest:
    if kind == EligibilityCopyKind.METHOD_READABLE:
        return manifest.model_copy(update={NATIVE_CLASS_IDS_FIELD: (), FINE_CONCEPT_FIELD: None})
    if kind == EligibilityCopyKind.ORACLE:
        if manifest.fine_concept is None:
            raise ValueError("oracle eligibility copy requires the fine concept")
        return manifest.model_copy(update={NATIVE_CLASS_IDS_FIELD: ()})
    return manifest
