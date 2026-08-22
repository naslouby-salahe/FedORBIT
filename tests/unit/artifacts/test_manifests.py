from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fedorbit.artifacts.manifests import (
    CompletionManifest,
    ReusableArtifactManifest,
    artifact_id,
    completion_manifest_self_hash,
    dependency_fingerprint,
    file_sha256,
)
from fedorbit.artifacts.reuse import ArtifactStore, ReuseError
from fedorbit.domain.enums import ArtifactState, TerminalState

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


def test_completion_manifest_round_trips_canonical_fixture() -> None:
    manifest = CompletionManifest.model_validate(_completion_payload())
    rendered = manifest.model_dump(mode="json")
    restored = CompletionManifest.model_validate(rendered)
    assert restored == manifest


def test_reusable_manifest_round_trips_canonical_fixture() -> None:
    manifest = ReusableArtifactManifest.model_validate(_reusable_payload())
    rendered = manifest.model_dump(mode="json")
    restored = ReusableArtifactManifest.model_validate(rendered)
    assert restored == manifest


def test_completion_manifest_rejects_unknown_fields() -> None:
    payload = _completion_payload()
    payload["invented"] = 1
    with pytest.raises(ValidationError):
        CompletionManifest.model_validate(payload)


def test_dependency_fingerprint_is_stable_and_sensitive() -> None:
    first = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    second = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    assert first == second
    changed = dependency_fingerprint(COORDINATES, ("upstream-1",), "c" * 64, "d" * 64, "e" * 64)
    assert changed != first
    changed_config = dependency_fingerprint(COORDINATES, (), "c2" * 64, "d" * 64, "e" * 64)
    assert changed_config != first


def test_artifact_id_is_stable_and_fingerprint_sensitive() -> None:
    first = artifact_id("prepared_split", COORDINATES, "a" * 64)
    second = artifact_id("prepared_split", COORDINATES, "a" * 64)
    assert first == second
    different = artifact_id("prepared_split", COORDINATES, "b" * 64)
    assert different != first
    other_type = artifact_id("checkpoint", COORDINATES, "a" * 64)
    assert other_type != first


def test_file_sha256_matches_known_digest(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"fedorbit-payload")
    expected = "cfa11ee0a14b0f6f7e5cf8b3c1ab2a94a1266c5e95d80a0efc7e0cdd97c8b4c0"
    assert file_sha256(payload) != expected
    assert len(file_sha256(payload)) == 64


def test_reuse_returns_same_artifact_under_identical_fingerprint(tmp_path: Path) -> None:
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

    resolved = store.find_by_fingerprint(fingerprint)
    assert resolved is not None
    assert resolved.artifact_id == manifest.artifact_id
    assert store.resolve(manifest.artifact_id).artifact_id == manifest.artifact_id


def test_reuse_does_not_duplicate_payload_for_second_cell(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "checkpoint.pt"
    payload.write_bytes(b"checkpoint-v1")
    fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    manifest = ReusableArtifactManifest.model_validate(
        {
            **_reusable_payload(),
            "artifact_type": "checkpoint",
            "artifact_id": artifact_id("checkpoint", COORDINATES, fingerprint),
            "dependency_fingerprint_sha256": fingerprint,
            "payload_paths": (str(payload),),
            "payload_sha256": file_sha256(payload),
        }
    )
    store.write_reusable(manifest)

    second_cell_fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    assert second_cell_fingerprint == fingerprint
    resolved = store.find_by_fingerprint(second_cell_fingerprint)
    assert resolved is not None
    assert resolved.payload_paths == (str(payload),)
    assert len(list((tmp_path / "manifests").glob("*.json"))) == 1


def test_reuse_rejects_corrupted_payload(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "prediction.npz"
    payload.write_bytes(b"payload-v1")
    fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    manifest = ReusableArtifactManifest.model_validate(
        {
            **_reusable_payload(),
            "artifact_type": "prediction",
            "artifact_id": artifact_id("prediction", COORDINATES, fingerprint),
            "dependency_fingerprint_sha256": fingerprint,
            "payload_paths": (str(payload),),
            "payload_sha256": file_sha256(payload),
        }
    )
    store.write_reusable(manifest)
    payload.write_bytes(b"corrupted")

    with pytest.raises(ReuseError):
        store.resolve(manifest.artifact_id)
    with pytest.raises(ReuseError):
        store.find_by_fingerprint(fingerprint)


def test_reuse_rejects_incomplete_state(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "packet.pt"
    payload.write_bytes(b"payload-v1")
    fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    manifest = ReusableArtifactManifest.model_validate(
        {
            **_reusable_payload(),
            "artifact_type": "response_packet",
            "artifact_id": artifact_id("response_packet", COORDINATES, fingerprint),
            "dependency_fingerprint_sha256": fingerprint,
            "payload_paths": (str(payload),),
            "payload_sha256": file_sha256(payload),
            "state": ArtifactState.RUNNING,
        }
    )
    store.write_reusable(manifest)
    with pytest.raises(ReuseError):
        store.resolve(manifest.artifact_id)


def test_manifest_references_do_not_transfer_ownership(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "derived.bin"
    payload.write_bytes(b"payload-v1")
    fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    manifest = ReusableArtifactManifest.model_validate(
        {
            **_reusable_payload(),
            "artifact_type": "solver_result",
            "artifact_id": artifact_id("solver_result", COORDINATES, fingerprint),
            "dependency_fingerprint_sha256": fingerprint,
            "payload_paths": (str(payload),),
            "payload_sha256": file_sha256(payload),
        }
    )
    store.write_reusable(manifest)
    resolved = store.resolve(manifest.artifact_id)
    assert resolved.payload_paths == (str(payload),)
    assert payload.is_file()


def test_completion_manifest_self_hash_excludes_own_field() -> None:
    manifest = CompletionManifest.model_validate(_completion_payload())
    self_hash = completion_manifest_self_hash(manifest)
    assert len(self_hash) == 64
    changed = manifest.model_copy(update={"producer_stage": "training"})
    assert completion_manifest_self_hash(changed) != self_hash
