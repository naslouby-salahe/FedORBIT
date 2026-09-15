from __future__ import annotations

from pathlib import Path

from tests.unit.artifacts.artifact_fixtures import (
    artifact_completion,
    artifact_manifest,
    payload_file,
)

from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.types import (
    ArtifactFingerprint,
    ArtifactIdentifier,
    ArtifactStage,
    ArtifactState,
    Sha256Digest,
)


def _root() -> Path:
    return Path("outputs")


def _workspace(store: ArtifactStore) -> Path:
    return store.root / "payloads"


def _chain(store: ArtifactStore, workspace: Path) -> tuple[ArtifactIdentifier, ...]:
    parent_payload = payload_file(workspace, "parent.bin", b"parent")
    parent = artifact_manifest(
        parent_payload,
        Sha256Digest("1" * 64),
        ArtifactStage.RAW,
    )
    store.promote_artifact(parent, artifact_completion(parent))
    child_payload = payload_file(workspace, "child.bin", b"child")
    child = artifact_manifest(
        child_payload,
        Sha256Digest("2" * 64),
        ArtifactStage.PREPROCESSING,
        upstream_artifact_ids=(parent.artifact_id,),
    )
    store.promote_artifact(child, artifact_completion(child))
    grandchild_payload = payload_file(workspace, "grandchild.bin", b"grandchild")
    grandchild = artifact_manifest(
        grandchild_payload,
        Sha256Digest("3" * 64),
        ArtifactStage.SCORING,
        upstream_artifact_ids=(child.artifact_id,),
    )
    store.promote_artifact(grandchild, artifact_completion(grandchild))
    return (parent.artifact_id, child.artifact_id, grandchild.artifact_id)


def _sibling(store: ArtifactStore, workspace: Path) -> ArtifactIdentifier:
    sibling_payload = payload_file(workspace, "sibling.bin", b"sibling")
    sibling = artifact_manifest(
        sibling_payload,
        Sha256Digest("4" * 64),
        ArtifactStage.RAW,
    )
    store.promote_artifact(sibling, artifact_completion(sibling))
    return sibling.artifact_id


def test_reverse_dependency_edges_are_derived_from_recorded_upstream_ids(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / _root())
    parent, child, grandchild = _chain(store, _workspace(store))
    assert store.descendants_of(parent) == (child, grandchild)
    graph = store.dependency_graph()
    assert graph.ancestors(grandchild) == (child, parent)


def test_unchanged_upstream_keeps_every_descendant_valid(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / _root())
    parent, child, grandchild = _chain(store, _workspace(store))
    assert store.artifact_state(parent).state is ArtifactState.COMPLETED
    assert store.artifact_state(child).state is ArtifactState.COMPLETED
    assert store.artifact_state(grandchild).state is ArtifactState.COMPLETED
    assert store.is_reusable(parent)
    assert store.is_reusable(child)
    assert store.is_reusable(grandchild)
    assert store.nearest_reusable_ancestor(grandchild) == child
    assert store.artifact_state(grandchild).nearest_reusable_ancestor == child


def test_absent_upstream_identity_marks_descendants_stale(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / _root())
    parent, child, grandchild = _chain(store, _workspace(store))
    store.manifest_path(parent).unlink()
    child_report = store.artifact_state(child)
    assert child_report.state is ArtifactState.STALE
    assert child_report.first_changed_dependency == parent
    assert store.artifact_state(grandchild).state is ArtifactState.STALE
    assert not store.is_reusable(child)
    assert store.find_by_fingerprint(ArtifactFingerprint("2" * 64)) is None


def test_replaced_parent_stales_exactly_its_descendants(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / _root())
    workspace = _workspace(store)
    parent, child, grandchild = _chain(store, _workspace(store))
    sibling = _sibling(store, _workspace(store))
    replacement_payload = payload_file(workspace, "parent-v2.bin", b"parent-rebuilt")
    replacement = artifact_manifest(
        replacement_payload,
        Sha256Digest("9" * 64),
        ArtifactStage.RAW,
    )
    report = store.supersession_report(parent, replacement.artifact_id)
    assert report.identity_preserved is False
    assert report.stale_descendants == (child, grandchild)
    assert report.recompute_boundary == child
    retired = store.retire_superseded_parent(parent, replacement.artifact_id)
    assert retired == (child, grandchild)
    assert store.artifact_state(child).state is ArtifactState.STALE
    assert store.artifact_state(grandchild).state is ArtifactState.STALE
    assert store.is_reusable(child) is False
    assert store.artifact_state(sibling).state is ArtifactState.COMPLETED
    assert store.is_reusable(sibling) is True


def test_same_identity_parent_replacement_preserves_descendants(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / _root())
    parent, child, grandchild = _chain(store, _workspace(store))
    report = store.supersession_report(parent, parent)
    assert report.identity_preserved is True
    assert report.stale_descendants == ()
    assert store.retire_superseded_parent(parent, parent) == ()
    assert store.artifact_state(child).state is ArtifactState.COMPLETED
    assert store.artifact_state(grandchild).state is ArtifactState.COMPLETED
    assert store.is_reusable(grandchild) is True


def test_promotion_with_supersedes_retires_descendants_in_one_step(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / _root())
    workspace = _workspace(store)
    parent, child, grandchild = _chain(store, _workspace(store))
    sibling = _sibling(store, _workspace(store))
    replacement_payload = payload_file(workspace, "parent-rebuilt.bin", b"rebuilt-parent")
    replacement = artifact_manifest(
        replacement_payload,
        Sha256Digest("8" * 64),
        ArtifactStage.RAW,
    )
    promoted = store.promote_artifact(
        replacement,
        artifact_completion(replacement),
        supersedes=parent,
    )
    assert store.is_reusable(promoted.artifact_id) is True
    assert store.artifact_state(child).state is ArtifactState.STALE
    assert store.artifact_state(grandchild).state is ArtifactState.STALE
    assert store.artifact_state(sibling).state is ArtifactState.COMPLETED
    assert store.is_reusable(sibling) is True


def test_stale_descendant_payload_and_completion_leave_the_active_namespace(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / _root())
    workspace = _workspace(store)
    parent, child, _grandchild = _chain(store, workspace)
    child_payload = Path(store.read_reusable(child).payload_paths[0])
    replacement_payload = payload_file(workspace, "parent-again.bin", b"other-parent")
    replacement = artifact_manifest(
        replacement_payload,
        Sha256Digest("7" * 64),
        ArtifactStage.RAW,
    )
    store.retire_superseded_parent(parent, replacement.artifact_id)
    assert not child_payload.is_file()
    assert not store.completion_path(child).is_file()
    assert store.read_reusable(child).state is ArtifactState.STALE
