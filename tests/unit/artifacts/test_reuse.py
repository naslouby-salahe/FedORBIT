from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.datasets.common import file_sha256
from fedorbit.infrastructure.manifests import (
    CompletionManifest,
    ReusableArtifactManifest,
    artifact_id,
    completion_manifest_self_hash,
)
from fedorbit.infrastructure.reuse import (
    ArtifactValidationError,
    validate_completed_artifact,
    validate_reusable_artifact,
)
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactSchemaVersion,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    CompletionValidationState,
    Sha256Digest,
    TerminalState,
)

COORDINATES = "{}"
FINGERPRINT = Sha256Digest("a" * 64)


def _manifest(payload: Path, fingerprint: Sha256Digest = FINGERPRINT) -> ReusableArtifactManifest:
    return ReusableArtifactManifest.model_validate(
        {
            "artifact_id": artifact_id(ArtifactType.PREPARED_SPLIT, COORDINATES, fingerprint),
            "artifact_type": ArtifactType.PREPARED_SPLIT,
            "semantic_producer_coordinates": COORDINATES,
            "producer_stage": ArtifactStage.PREPROCESSING,
            "dependency_fingerprint_sha256": fingerprint,
            "upstream_artifact_ids": (),
            "applicable_configuration_sha256": "b" * 64,
            "relevant_code_sha256": "c" * 64,
            "material_runtime_sha256": "d" * 64,
            "payload_paths": (str(payload),),
            "payload_sha256": file_sha256(payload),
            "schema_version": ArtifactSchemaVersion.V1,
            "created_git_commit": "e" * 40,
            "created_environment_sha256": "f" * 64,
            "state": ArtifactState.COMPLETED,
            "completion_manifest_sha256": "0" * 64,
        }
    )


def _completion(manifest: ReusableArtifactManifest) -> CompletionManifest:
    draft = CompletionManifest.model_validate(
        {
            "schema_version": ArtifactSchemaVersion.V1,
            "semantic_experiment_coordinates": COORDINATES,
            "producer_stage": manifest.producer_stage,
            "terminal_state": TerminalState.COMPLETED,
            "dependency_fingerprint_sha256": manifest.dependency_fingerprint_sha256,
            "upstream_artifact_ids": (),
            "mandatory_artifact_paths": (),
            "mandatory_artifact_sha256": "a" * 64,
            "scientific_configuration_sha256": "b" * 64,
            "relevant_code_sha256": "c" * 64,
            "material_runtime_sha256": "d" * 64,
            "upstream_lineage": "{}",
            "completion_validation_state": CompletionValidationState.VALIDATED,
            "completion_written_last": True,
            "completion_manifest_sha256": "0" * 64,
        }
    )
    return draft.model_copy(
        update={"completion_manifest_sha256": completion_manifest_self_hash(draft)}
    )


def _payload(tmp_path: Path, content: bytes = b"payload") -> Path:
    path = tmp_path / "payload.bin"
    path.write_bytes(content)
    return path


def test_completed_artifact_with_matching_completion_validates(tmp_path: Path) -> None:
    path = _payload(tmp_path)
    manifest = _manifest(path)
    completion = _completion(manifest)
    manifest = manifest.model_copy(
        update={"completion_manifest_sha256": completion.completion_manifest_sha256}
    )
    validate_completed_artifact(manifest, completion)


def test_artifact_that_is_not_completed_is_rejected(tmp_path: Path) -> None:
    path = _payload(tmp_path)
    manifest = _manifest(path).model_copy(update={"state": ArtifactState.RUNNING})
    with pytest.raises(ArtifactValidationError):
        validate_reusable_artifact(manifest)


def test_artifact_without_payload_is_rejected(tmp_path: Path) -> None:
    path = _payload(tmp_path)
    manifest = _manifest(path).model_copy(update={"payload_paths": ()})
    with pytest.raises(ArtifactValidationError):
        validate_reusable_artifact(manifest)


def test_missing_payload_file_is_rejected(tmp_path: Path) -> None:
    path = _payload(tmp_path)
    manifest = _manifest(path)
    path.unlink()
    with pytest.raises(ArtifactValidationError):
        validate_reusable_artifact(manifest)


def test_payload_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    path = _payload(tmp_path)
    manifest = _manifest(path)
    path.write_bytes(b"tampered")
    with pytest.raises(ArtifactValidationError):
        validate_reusable_artifact(manifest)


def test_completion_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    path = _payload(tmp_path)
    manifest = _manifest(path)
    completion = _completion(manifest).model_copy(
        update={"completion_manifest_sha256": Sha256Digest("9" * 64)}
    )
    with pytest.raises(ArtifactValidationError):
        validate_completed_artifact(manifest, completion)


def test_completion_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    path = _payload(tmp_path)
    manifest = _manifest(path)
    completion = _completion(manifest).model_copy(
        update={
            "dependency_fingerprint_sha256": Sha256Digest("9" * 64),
            "completion_manifest_sha256": manifest.completion_manifest_sha256,
        }
    )
    with pytest.raises(ArtifactValidationError):
        validate_completed_artifact(manifest, completion)


def test_completion_not_written_last_is_rejected(tmp_path: Path) -> None:
    path = _payload(tmp_path)
    manifest = _manifest(path)
    completion = _completion(manifest).model_copy(update={"completion_written_last": False})
    with pytest.raises(ArtifactValidationError):
        validate_completed_artifact(manifest, completion)


def test_artifact_identity_is_deterministic_and_fingerprint_sensitive() -> None:
    baseline = artifact_id(ArtifactType.PREPARED_SPLIT, COORDINATES, FINGERPRINT)
    assert artifact_id(ArtifactType.PREPARED_SPLIT, COORDINATES, FINGERPRINT) == baseline
    other = artifact_id(ArtifactType.PREPARED_SPLIT, COORDINATES, Sha256Digest("9" * 64))
    assert other != baseline
    assert ArtifactIdentifier(other.value) == other
