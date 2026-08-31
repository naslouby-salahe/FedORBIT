from __future__ import annotations

from pathlib import Path

from fedorbit.artifacts.manifests import ReusableArtifactManifest, artifact_id, file_sha256
from fedorbit.artifacts.storage import ArtifactStore
from fedorbit.domain.enums import ArtifactState
from fedorbit.domain.records import (
    ArtifactFingerprint,
    ArtifactIdentifier,
    ExecutionCell,
    SemanticCoordinates,
)
from fedorbit.execution.recovery import RecoveryBoundary
from fedorbit.execution.reuse import (
    SelectiveInvalidation,
    changed_stage_affects,
    descendants_of_stage,
)

COORDINATES = {"experiment": "Primary Strict Cross-Telemetry Transfer"}


def _payload(tmp_path: Path, name: str, content: bytes = b"payload") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _manifest(
    payload: Path,
    artifact_type: str,
    stage: str,
    fingerprint: str,
    upstream: tuple[str, ...] = (),
) -> ReusableArtifactManifest:
    return ReusableArtifactManifest.model_validate(
        {
            "artifact_id": artifact_id(artifact_type, COORDINATES, fingerprint),
            "artifact_type": artifact_type,
            "semantic_producer_coordinates": "{}",
            "producer_stage": stage,
            "dependency_fingerprint_sha256": fingerprint,
            "upstream_artifact_ids": upstream,
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


def test_descendants_of_stage_follows_dependency_graph() -> None:
    descendants = descendants_of_stage("raw")
    assert "preprocessing" in descendants
    assert "training" in descendants
    assert "statistics" in descendants
    assert "raw" not in descendants


def test_changed_stage_affects_producer() -> None:
    assert changed_stage_affects("training", "raw")
    assert changed_stage_affects("statistics", "training")
    assert changed_stage_affects("reporting", "raw")
    assert not changed_stage_affects("raw", "training")
    assert not changed_stage_affects("preprocessing", "reporting")


def test_invalidation_propagates_only_to_descendants(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    raw_payload = _payload(tmp_path, "raw.bin")
    train_payload = _payload(tmp_path, "train.pt")
    report_payload = _payload(tmp_path, "report.json")
    store.write_reusable(
        _manifest(raw_payload, "prepared_split", "preprocessing", "fp-pre", ("raw-up",))
    )
    store.write_reusable(
        _manifest(train_payload, "checkpoint", "training", "fp-train", ("pre-up",))
    )
    store.write_reusable(_manifest(report_payload, "other", "reporting", "fp-report", ("stat-up",)))
    invalidated = SelectiveInvalidation(store).invalidate_stage("training")
    assert len(invalidated) == 2
    remaining = {path.stem for path in store.manifest_dir().glob("*.json")}
    assert len(remaining) == 1


def test_invalidation_keeps_siblings_and_unrelated(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    training_a = _payload(tmp_path, "a.pt")
    training_b = _payload(tmp_path, "b.pt")
    first = _manifest(training_a, "checkpoint", "training", "fp-a", ("pre-a",))
    second = _manifest(training_b, "checkpoint", "training", "fp-b", ("pre-b",))
    store.write_reusable(first)
    store.write_reusable(second)
    invalidated = SelectiveInvalidation(store).invalidate_descendants("pre-a")
    assert invalidated == (first.artifact_id,)
    assert {path.stem for path in store.manifest_dir().glob("*.json")} == {second.artifact_id}


def test_invalidation_propagates_transitively(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    mid = _payload(tmp_path, "mid.bin")
    leaf = _payload(tmp_path, "leaf.bin")
    mid_manifest = _manifest(mid, "response_packet", "response", "fp-mid", ("target-up",))
    leaf_manifest = _manifest(
        leaf, "confirmation_input", "confirmation", "fp-leaf", (mid_manifest.artifact_id,)
    )
    store.write_reusable(mid_manifest)
    store.write_reusable(leaf_manifest)
    invalidated = SelectiveInvalidation(store).invalidate_descendants("target-up")
    assert set(invalidated) == {mid_manifest.artifact_id, leaf_manifest.artifact_id}
    assert not list(store.manifest_dir().glob("*.json"))


def test_recovery_boundary_finds_first_incomplete_cell(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = _payload(tmp_path, "ok.pt")
    manifest = _manifest(payload, "checkpoint", "training", "fp-ok")
    store.write_reusable(manifest)
    record = RecoveryBoundary(store).next_resume(
        (
            ExecutionCell(
                SemanticCoordinates("cell-1"),
                ArtifactIdentifier(manifest.artifact_id),
                ArtifactFingerprint("fp-ok"),
            ),
            ExecutionCell(
                SemanticCoordinates("cell-2"),
                ArtifactIdentifier("fp-missing"),
                ArtifactFingerprint("fp-missing"),
            ),
            ExecutionCell(
                SemanticCoordinates("cell-3"),
                ArtifactIdentifier("fp-missing"),
                ArtifactFingerprint("fp-missing"),
            ),
        )
    )
    assert record.next_resume_coordinates == SemanticCoordinates("cell-2")
    assert ArtifactIdentifier("fp-missing") not in set(record.valid_artifact_ids)


def test_recovery_boundary_all_valid(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = _payload(tmp_path, "ok.pt")
    manifest = _manifest(payload, "checkpoint", "training", "fp-ok")
    store.write_reusable(manifest)
    record = RecoveryBoundary(store).next_resume(
        (
            ExecutionCell(
                SemanticCoordinates("cell-1"),
                ArtifactIdentifier(manifest.artifact_id),
                ArtifactFingerprint("fp-ok"),
            ),
        )
    )
    assert record.next_resume_coordinates is None


def test_recovery_discards_interrupted_staging(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    staging = store.staging_dir()
    staging.mkdir(parents=True)
    (staging / "partial.bin").write_bytes(b"partial")
    RecoveryBoundary(store).discard_interrupted_staging()
    assert not staging.exists()
