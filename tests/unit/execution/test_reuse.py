from __future__ import annotations

from pathlib import Path

from fedorbit.artifacts.manifests import (
    ReusableArtifactManifest,
    artifact_id,
    dependency_fingerprint,
    file_sha256,
)
from fedorbit.artifacts.storage import ArtifactStore
from fedorbit.domain.enums import ArtifactState, OverwritePolicy
from fedorbit.domain.records import (
    ArtifactFingerprint,
    ArtifactIdentifier,
    ExecutionCell,
    SemanticCoordinates,
)
from fedorbit.execution.reuse import ExecutionAction, ExecutionReuse

COORDINATES = {"experiment": "Preprocessing", "dataset": "edge_iiotset_network"}


def _cell(coordinates: str, fingerprint: str) -> ExecutionCell:
    return ExecutionCell(
        SemanticCoordinates(coordinates),
        ArtifactIdentifier(fingerprint),
        ArtifactFingerprint(fingerprint),
    )


def _payload(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _manifest(
    payload: Path, fingerprint: str, artifact_type: str = "prepared_split"
) -> ReusableArtifactManifest:
    return ReusableArtifactManifest.model_validate(
        {
            "artifact_id": artifact_id(artifact_type, COORDINATES, fingerprint),
            "artifact_type": artifact_type,
            "semantic_producer_coordinates": "{}",
            "producer_stage": "preprocessing",
            "dependency_fingerprint_sha256": fingerprint,
            "upstream_artifact_ids": (),
            "applicable_configuration_sha256": "c" * 64,
            "relevant_code_sha256": "d" * 64,
            "material_runtime_sha256": "e" * 64,
            "payload_paths": (str(payload),),
            "payload_sha256": file_sha256(payload),
            "schema_version": "1.0",
            "created_git_commit": "a" * 40,
            "created_environment_sha256": "g" * 64,
            "state": ArtifactState.COMPLETED,
            "completion_manifest_sha256": "f" * 64,
        }
    )


def test_identical_fingerprint_is_reused_without_overwrite(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = _payload(tmp_path, "split.parquet", b"payload-v1")
    fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    store.write_reusable(_manifest(payload, fingerprint))
    reuse = ExecutionReuse(store)

    decisions = reuse.decide((_cell("split-cell", fingerprint),), OverwritePolicy.REUSE)
    assert len(decisions) == 1
    assert decisions[0].action == ExecutionAction.REUSE
    assert decisions[0].manifest is not None


def test_missing_artifact_is_executed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    reuse = ExecutionReuse(store)
    fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    decisions = reuse.decide((_cell("split-cell", fingerprint),), OverwritePolicy.REUSE)
    assert decisions[0].action == ExecutionAction.EXECUTE


def test_overwrite_flag_forces_recompute(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = _payload(tmp_path, "split.parquet", b"payload-v1")
    fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    store.write_reusable(_manifest(payload, fingerprint))
    reuse = ExecutionReuse(store)
    decisions = reuse.decide((_cell("split-cell", fingerprint),), OverwritePolicy.REPLACE)
    assert decisions[0].action == ExecutionAction.OVERWRITE


def test_stale_descendant_is_overwritten(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = _payload(tmp_path, "derived.pt", b"payload-v1")
    fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    manifest = _manifest(payload, fingerprint, artifact_type="checkpoint")
    store.write_reusable(manifest)
    reuse = ExecutionReuse(store)
    decisions = reuse.decide(
        (_cell("derived-cell", fingerprint),),
        OverwritePolicy.REUSE,
        stale_artifact_ids=frozenset({ArtifactIdentifier(manifest.artifact_id)}),
    )
    assert decisions[0].action == ExecutionAction.OVERWRITE


def test_corrupted_payload_is_not_reused(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = _payload(tmp_path, "split.parquet", b"payload-v1")
    fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    manifest = _manifest(payload, fingerprint)
    store.write_reusable(manifest)
    payload.write_bytes(b"corrupted")
    reuse = ExecutionReuse(store)
    decisions = reuse.decide((_cell("split-cell", fingerprint),), OverwritePolicy.REUSE)
    assert decisions[0].action == ExecutionAction.EXECUTE


def test_stale_descendants_detected_via_upstream_ids(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = _payload(tmp_path, "checkpoint.pt", b"payload-v1")
    fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    manifest = ReusableArtifactManifest.model_validate(
        {
            **_manifest(payload, fingerprint, artifact_type="checkpoint").model_dump(),
            "upstream_artifact_ids": ("upstream-artifact-1",),
        }
    )
    store.write_reusable(manifest)
    reuse = ExecutionReuse(store)
    stale = reuse.stale_descendants("upstream-artifact-1")
    assert manifest.artifact_id in stale


def test_promote_completed_manifests_validates_before_reuse(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = _payload(tmp_path, "packet.pt", b"payload-v1")
    fingerprint = dependency_fingerprint(COORDINATES, (), "c" * 64, "d" * 64, "e" * 64)
    manifest = _manifest(payload, fingerprint, artifact_type="response_packet")
    reuse = ExecutionReuse(store)
    reuse.promote_completed((manifest,))
    assert (
        store.resolve(ArtifactIdentifier(manifest.artifact_id)).artifact_id == manifest.artifact_id
    )
