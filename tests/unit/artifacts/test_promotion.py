from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.artifacts.artifact_fixtures import (
    artifact_completion,
    artifact_manifest,
    payload_file,
)

from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
    RecoveryBoundary,
    StagedPayload,
)
from fedorbit.infrastructure.manifests import ReusableArtifactManifest
from fedorbit.infrastructure.reuse import ArtifactValidationError, validate_artifact_lineage
from fedorbit.infrastructure.storage import StorageError
from fedorbit.types import (
    ArtifactFingerprint,
    ArtifactStage,
    ArtifactState,
    Sha256Digest,
)


def _root(tmp_path: Path) -> Path:
    return tmp_path / "outputs"


def _published_store(tmp_path: Path) -> tuple[ArtifactStore, Path]:
    store = ArtifactStore(_root(tmp_path))
    payload = payload_file(store.root / "artifacts", "payload.bin", b"payload")
    manifest = artifact_manifest(payload, Sha256Digest("1" * 64), ArtifactStage.RAW)
    store.promote_artifact(manifest, artifact_completion(manifest))
    return store, payload


def _staged(
    store: ArtifactStore,
    fingerprint: Sha256Digest,
    content: bytes,
) -> tuple[ReusableArtifactManifest, StagedPayload, Path]:
    draft_directory = store.staging_dir() / "draft"
    draft = payload_file(draft_directory, "payload.bin", content)
    active = store.root / "artifacts" / f"{fingerprint[:8]}.bin"
    manifest = artifact_manifest(draft, fingerprint, ArtifactStage.RAW).model_copy(
        update={"payload_paths": (str(active),)}
    )
    staged = payload_file(store.staging_area(manifest.artifact_id), "payload.bin", content)
    draft.unlink()
    return manifest, StagedPayload(staged, active), active


def test_staged_payload_is_promoted_then_completion_is_written_last(tmp_path: Path) -> None:
    store = ArtifactStore(_root(tmp_path))
    manifest, entry, active = _staged(store, Sha256Digest("2" * 64), b"staged")
    staged = entry.staged_path
    store.promote_artifact(
        manifest,
        artifact_completion(manifest),
        staged_payloads=(entry,),
    )
    assert active.is_file()
    assert active.read_bytes() == b"staged"
    assert not staged.is_file()
    assert not store.staging_area(manifest.artifact_id).is_dir()
    assert store.is_reusable(manifest.artifact_id) is True
    assert store.resolve(manifest.artifact_id).payload_paths == (str(active),)


def test_corrupt_staged_payload_is_never_promoted_to_a_completed_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(_root(tmp_path))
    manifest, entry, active = _staged(store, Sha256Digest("3" * 64), b"corrupt")
    staged = entry.staged_path
    manifest = manifest.model_copy(update={"payload_sha256": Sha256Digest("9" * 64)})
    with pytest.raises(StorageError, match="checksum mismatch"):
        store.promote_artifact(
            manifest,
            artifact_completion(manifest),
            staged_payloads=(entry,),
        )
    assert not active.is_file()
    assert not store.manifest_path(manifest.artifact_id).is_file()
    assert not store.completion_path(manifest.artifact_id).is_file()
    assert staged.is_file()
    assert store.is_reusable(manifest.artifact_id) is False


def test_torn_promotion_without_completion_is_never_reusable(tmp_path: Path) -> None:
    store = ArtifactStore(_root(tmp_path))
    payload = payload_file(store.root / "artifacts", "torn.bin", b"torn")
    manifest = artifact_manifest(payload, Sha256Digest("4" * 64), ArtifactStage.RAW)
    store.write_reusable(manifest)
    report = store.artifact_state(manifest.artifact_id)
    assert report.state is ArtifactState.INVALID
    assert store.is_reusable(manifest.artifact_id) is False
    assert store.find_by_fingerprint(ArtifactFingerprint("4" * 64)) is None
    with pytest.raises(StorageError, match="no completion manifest"):
        store.resolve(manifest.artifact_id)


def test_interrupted_staging_is_discarded_and_never_reusable(tmp_path: Path) -> None:
    store = ArtifactStore(_root(tmp_path))
    _manifest, entry, _active = _staged(store, Sha256Digest("5" * 64), b"partial")
    staged = entry.staged_path
    assert staged.is_file()
    RecoveryBoundary(store).discard_interrupted_staging()
    assert not staged.is_file()
    assert not store.staging_dir().is_dir()


def test_promotion_persists_the_declared_upstream_lineage(tmp_path: Path) -> None:
    store, _payload = _published_store(tmp_path)
    upstream = store.all_manifests()[0].artifact_id
    payload = payload_file(store.root / "artifacts", "downstream.bin", b"downstream")
    manifest = artifact_manifest(
        payload,
        Sha256Digest("6" * 64),
        ArtifactStage.EVALUATION,
        upstream_artifact_ids=(upstream,),
    )
    completion = artifact_completion(manifest)
    assert completion.upstream_lineage == "{}"
    promoted = store.promote_artifact(manifest, completion)
    recorded = store.read_completion(promoted.artifact_id)
    assert recorded.upstream_lineage != "{}"
    validate_artifact_lineage(store.read_reusable(promoted.artifact_id), recorded)
    assert store.read_reusable(promoted.artifact_id).completion_manifest_sha256 == (
        recorded.completion_manifest_sha256
    )
    assert store.is_reusable(promoted.artifact_id) is True


def test_promotion_fails_closed_when_completion_contradicts_declared_upstream(
    tmp_path: Path,
) -> None:
    store, _payload = _published_store(tmp_path)
    upstream = store.all_manifests()[0].artifact_id
    payload = payload_file(store.root / "artifacts", "contradiction.bin", b"contradiction")
    manifest = artifact_manifest(
        payload,
        Sha256Digest("7" * 64),
        ArtifactStage.EVALUATION,
        upstream_artifact_ids=(upstream,),
    )
    completion = artifact_completion(manifest).model_copy(update={"upstream_artifact_ids": ()})
    with pytest.raises(StorageError, match="disagree on upstream artifact identities"):
        store.promote_artifact(manifest, completion)
    assert not store.manifest_path(manifest.artifact_id).is_file()


def test_corrupt_promoted_payload_is_invalid_and_not_reusable(tmp_path: Path) -> None:
    store, payload = _published_store(tmp_path)
    manifest = store.all_manifests()[0]
    assert store.is_reusable(manifest.artifact_id) is True
    payload.write_bytes(b"payload-corrupted")
    assert store.artifact_state(manifest.artifact_id).state is ArtifactState.INVALID
    assert store.is_reusable(manifest.artifact_id) is False
    assert store.find_by_fingerprint(ArtifactFingerprint("1" * 64)) is None
    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        store.resolve(manifest.artifact_id)


def test_reuse_abstains_when_a_stage_does_not_record_its_upstream_identity(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(_root(tmp_path))
    payload = payload_file(store.root / "preprocessing", "prepared.bin", b"prepared")
    manifest = artifact_manifest(payload, Sha256Digest("8" * 64), ArtifactStage.PREPROCESSING)
    store.promote_artifact(manifest, artifact_completion(manifest))
    assert store.artifact_state(manifest.artifact_id).state is ArtifactState.COMPLETED
    assert store.is_reusable(manifest.artifact_id) is False
    assert store.find_by_fingerprint(ArtifactFingerprint("8" * 64)) is None


def test_stale_artifact_identity_is_not_offered_for_reuse(tmp_path: Path) -> None:
    store, _payload = _published_store(tmp_path)
    manifest = store.all_manifests()[0]
    store.write_reusable(manifest.model_copy(update={"state": ArtifactState.STALE}))
    report = store.artifact_state(manifest.artifact_id)
    assert report.state is ArtifactState.STALE
    assert report.reason != ""
    assert store.is_reusable(manifest.artifact_id) is False
    assert store.find_by_fingerprint(ArtifactFingerprint("1" * 64)) is None
