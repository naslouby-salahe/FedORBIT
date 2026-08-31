from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fedorbit.artifacts.manifests import (
    CompletionManifest,
    DatasetManifest,
    EligibilityCopyKind,
    ReusableArtifactManifest,
    SemanticCellManifest,
    TransferEligibilityManifest,
    artifact_id,
    completion_manifest_self_hash,
    dependency_fingerprint,
    eligibility_copy,
    file_sha256,
)
from fedorbit.artifacts.storage import ArtifactStore
from fedorbit.artifacts.validation import ArtifactValidationError
from fedorbit.domain.enums import ArtifactStage, ArtifactState, TerminalState
from fedorbit.domain.records import ArtifactIdentifier

COORDINATES = {
    "experiment": "Primary Strict Cross-Telemetry Transfer",
    "pair": ["edge_iiotset_network", "ton_iot_network"],
    "seed": 1103,
}
MANIFEST_VERSION = "1.0"


def _completion_payload() -> dict[str, object]:
    return {
        "schema_version": MANIFEST_VERSION,
        "semantic_experiment_coordinates": "{}",
        "producer_stage": "evaluation",
        "terminal_state": TerminalState.COMPLETED,
        "dependency_fingerprint_sha256": "a" * 64,
        "upstream_artifact_ids": (),
        "mandatory_artifact_paths": (),
        "mandatory_artifact_sha256": "b" * 64,
        "scientific_configuration_sha256": "c" * 64,
        "relevant_code_sha256": "d" * 64,
        "material_runtime_sha256": "e" * 64,
        "upstream_lineage": "{}",
        "completion_validation_state": "validated",
        "completion_written_last": True,
        "completion_manifest_sha256": "f" * 64,
    }


def _reusable_payload() -> dict[str, object]:
    return {
        "artifact_id": "id",
        "artifact_type": "prepared_split",
        "semantic_producer_coordinates": "{}",
        "producer_stage": "preprocessing",
        "dependency_fingerprint_sha256": "a" * 64,
        "upstream_artifact_ids": (),
        "applicable_configuration_sha256": "c" * 64,
        "relevant_code_sha256": "d" * 64,
        "material_runtime_sha256": "e" * 64,
        "payload_paths": (),
        "payload_sha256": "b" * 64,
        "schema_version": MANIFEST_VERSION,
        "created_git_commit": "a" * 40,
        "created_environment_sha256": "g" * 64,
        "state": ArtifactState.COMPLETED,
        "completion_manifest_sha256": "f" * 64,
    }


COMPLETION_FIELDS = (
    "semantic_experiment_coordinates",
    "producer_stage",
    "terminal_state",
    "dependency_fingerprint_sha256",
    "upstream_artifact_ids",
    "mandatory_artifact_paths",
    "mandatory_artifact_sha256",
    "scientific_configuration_sha256",
    "relevant_code_sha256",
    "material_runtime_sha256",
    "upstream_lineage",
    "completion_validation_state",
    "completion_written_last",
    "completion_manifest_sha256",
)
REUSABLE_FIELDS = (
    "artifact_id",
    "artifact_type",
    "semantic_producer_coordinates",
    "producer_stage",
    "dependency_fingerprint_sha256",
    "upstream_artifact_ids",
    "applicable_configuration_sha256",
    "relevant_code_sha256",
    "material_runtime_sha256",
    "payload_paths",
    "payload_sha256",
    "schema_version",
    "created_git_commit",
    "created_environment_sha256",
    "state",
    "completion_manifest_sha256",
)


@pytest.mark.parametrize("field", COMPLETION_FIELDS)
def test_completion_manifest_requires_field(field: str) -> None:
    payload = _completion_payload()
    del payload[field]
    with pytest.raises(ValidationError):
        CompletionManifest.model_validate(payload)


@pytest.mark.parametrize("field", REUSABLE_FIELDS)
def test_reusable_manifest_requires_field(field: str) -> None:
    payload = _reusable_payload()
    del payload[field]
    with pytest.raises(ValidationError):
        ReusableArtifactManifest.model_validate(payload)


def test_completion_and_reusable_manifests_round_trip() -> None:
    completion = CompletionManifest.model_validate(_completion_payload())
    reusable = ReusableArtifactManifest.model_validate(_reusable_payload())
    assert CompletionManifest.model_validate(completion.model_dump(mode="json")) == completion
    assert ReusableArtifactManifest.model_validate(reusable.model_dump(mode="json")) == reusable


def test_completion_manifest_rejects_unknown_fields() -> None:
    payload = _completion_payload()
    payload["invented"] = 1
    with pytest.raises(ValidationError):
        CompletionManifest.model_validate(payload)


def test_dependency_fingerprint_and_artifact_identity_are_stable_and_sensitive() -> None:
    first = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    assert first == dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    assert first != dependency_fingerprint(COORDINATES, ("upstream",), "c" * 64, "d" * 64, "e" * 64)
    assert artifact_id("prepared_split", COORDINATES, first) != artifact_id(
        "checkpoint", COORDINATES, first
    )


def test_file_sha256_is_deterministic(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"fedorbit-payload")
    assert file_sha256(payload) == file_sha256(payload)
    assert len(file_sha256(payload)) == 64


def test_storage_validates_payload_checksum_and_terminal_state(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "split.parquet"
    payload.write_bytes(b"payload-v1")
    fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    manifest = ReusableArtifactManifest.model_validate(
        {
            **_reusable_payload(),
            "artifact_id": artifact_id("prepared_split", COORDINATES, fingerprint),
            "dependency_fingerprint_sha256": fingerprint,
            "payload_paths": (str(payload),),
            "payload_sha256": file_sha256(payload),
        }
    )
    store.write_reusable(manifest)
    assert store.resolve(ArtifactIdentifier(manifest.artifact_id)) == manifest
    payload.write_bytes(b"corrupted")
    with pytest.raises(ArtifactValidationError):
        store.resolve(ArtifactIdentifier(manifest.artifact_id))


def test_completion_manifest_self_hash_excludes_own_field() -> None:
    manifest = CompletionManifest.model_validate(_completion_payload())
    self_hash = completion_manifest_self_hash(manifest)
    assert len(self_hash) == 64
    assert (
        completion_manifest_self_hash(
            manifest.model_copy(update={"producer_stage": ArtifactStage.TRAINING})
        )
        != self_hash
    )


DATASET_FIELDS = {
    "dataset": "edge_iiotset_network",
    "component": "network",
    "raw_files": ("ML-EdgeIIoT-dataset.csv",),
    "raw_sha256": "a" * 64,
    "raw_counts": {"rows": 100},
    "schema": "1.0",
    "adapter_feature_order": ("tcp.ack",),
    "adapter_feature_roles": {"tcp.ack": "behavioral_numeric"},
    "accepted_schema_aliases": (),
    "adapter_adaptations": (),
    "timestamp_field": "frame.time",
    "timestamp_range": ("2020-01-01", "2020-01-02"),
    "duplicate_counts": {"group-a": 2},
    "conflicting_duplicate_counts": {},
    "local_class_counts": {"normal": 50, "ddos_tcp": 50},
    "transfer_candidate_counts": {"DDoS": 50},
    "feature_quality": {"dropped": 0},
    "preprocessing_state": "completed",
    "dependency_fingerprint_sha256": "b" * 64,
    "producer_code_sha256": "c" * 64,
}
ELIGIBILITY_FIELDS = {
    "client": "edge_iiotset_network",
    "seed": 1103,
    "coarse_group": "Disruption",
    "anonymous_node_id": "node-0042",
    "native_local_class_ids": ("ddos_tcp",),
    "present": True,
    "train_count": 200,
    "meta_count": 40,
    "confirm_count": 0,
    "test_count": 40,
    "source_eligible": True,
    "target_eligible": False,
    "null_reason": None,
    "fine_concept": "DDoS",
}
CELL_FIELDS = {
    "experiment": "Primary Strict Cross-Telemetry Transfer",
    "dataset": "edge_iiotset_network",
    "source_client": "edge_iiotset_network",
    "target_client": "ton_iot_windows10_host",
    "directed_pair": "edge_iiotset_network -> ton_iot_windows10_host",
    "method": "FedORBIT Exact-Sparse Solver",
    "condition": "principal",
    "support": 2,
    "seed": 1103,
    "scientific_configuration_sha256": "d" * 64,
    "dependency_fingerprint_sha256": "e" * 64,
    "producer_stage": "response",
}


def test_dataset_manifest_round_trips_with_schema_alias() -> None:
    manifest = DatasetManifest.model_validate(DATASET_FIELDS)
    dumped = manifest.model_dump(mode="json", by_alias=True)
    assert dumped["schema"] == "1.0"
    assert DatasetManifest.model_validate(dumped) == manifest


def test_dataset_manifest_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate({**DATASET_FIELDS, "invented": 1})


def test_eligibility_copies_enforce_method_oracle_separation() -> None:
    manifest = TransferEligibilityManifest.model_validate(ELIGIBILITY_FIELDS)
    readable = eligibility_copy(manifest, EligibilityCopyKind.METHOD_READABLE)
    assert readable.native_local_class_ids == ()
    assert readable.fine_concept is None
    oracle = eligibility_copy(manifest, EligibilityCopyKind.ORACLE)
    assert oracle.native_local_class_ids == ()
    assert oracle.fine_concept == "DDoS"
    builder = eligibility_copy(manifest, EligibilityCopyKind.BUILDER)
    assert builder.native_local_class_ids == ("ddos_tcp",)


def test_oracle_eligibility_copy_requires_fine_concept() -> None:
    manifest = TransferEligibilityManifest.model_validate(
        {**ELIGIBILITY_FIELDS, "fine_concept": None}
    )
    with pytest.raises(ValueError):
        eligibility_copy(manifest, EligibilityCopyKind.ORACLE)


def test_semantic_cell_manifest_round_trips() -> None:
    manifest = SemanticCellManifest.model_validate(CELL_FIELDS)
    assert SemanticCellManifest.model_validate(manifest.model_dump(mode="json")) == manifest
