from __future__ import annotations

from pathlib import Path

from fedorbit.artifacts.invalidation import (
    SelectiveInvalidation,
    changed_stage_affects,
    descendants_of_stage,
)
from fedorbit.artifacts.manifests import ReusableArtifactManifest, artifact_id, file_sha256
from fedorbit.artifacts.reuse import ArtifactStore
from fedorbit.domain.enums import ArtifactState
from fedorbit.execution.recovery import RecoveryBoundary

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
    invalidator = SelectiveInvalidation(store)

    invalidated = invalidator.invalidate_stage("training")
    assert len(invalidated) == 2
    assert store.manifest_dir().is_dir()
    remaining = {path.stem for path in store.manifest_dir().glob("*.json")}
    assert len(remaining) == 1


def test_invalidation_keeps_siblings_and_unrelated(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    training_a = _payload(tmp_path, "a.pt")
    training_b = _payload(tmp_path, "b.pt")
    store.write_reusable(_manifest(training_a, "checkpoint", "training", "fp-a", ("pre-a",)))
    store.write_reusable(_manifest(training_b, "checkpoint", "training", "fp-b", ("pre-b",)))
    first_manifest = store.read_reusable(
        _manifest(training_a, "checkpoint", "training", "fp-a", ("pre-a",)).artifact_id
    )
    invalidator = SelectiveInvalidation(store)
    invalidated = invalidator.invalidate_descendants("pre-a")
    assert tuple(invalidated) == (first_manifest.artifact_id,)
    remaining = {path.stem for path in store.manifest_dir().glob("*.json")}
    assert len(remaining) == 1


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
    invalidator = SelectiveInvalidation(store)
    invalidated = invalidator.invalidate_descendants("target-up")
    assert set(invalidated) == {mid_manifest.artifact_id, leaf_manifest.artifact_id}
    assert len(list(store.manifest_dir().glob("*.json"))) == 0


def test_recovery_boundary_finds_first_incomplete_cell(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = _payload(tmp_path, "ok.pt")
    manifest = _manifest(payload, "checkpoint", "training", "fp-ok")
    store.write_reusable(manifest)
    recovery = RecoveryBoundary(store)
    ordered = (
        ("cell-1", manifest.artifact_id),
        ("cell-2", "fp-missing"),
        ("cell-3", "fp-missing"),
    )
    record = recovery.next_resume(ordered)
    assert record.next_resume_coordinates == "cell-2"
    assert "fp-missing" not in set(record.valid_artifact_ids)


def test_recovery_boundary_all_valid(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = _payload(tmp_path, "ok.pt")
    manifest = _manifest(payload, "checkpoint", "training", "fp-ok")
    store.write_reusable(manifest)
    recovery = RecoveryBoundary(store)
    record = recovery.next_resume((("cell-1", manifest.artifact_id),))
    assert record.next_resume_coordinates is None
