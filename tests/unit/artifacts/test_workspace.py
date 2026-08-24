from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.artifacts.evidence import EvidenceError, VerifiedEvidenceWriter
from fedorbit.artifacts.manifests import ReusableArtifactManifest, artifact_id, file_sha256
from fedorbit.artifacts.paths import (
    WorkspaceError,
    build_layout,
    enforce_workspace_boundary,
    experiment_workspace,
    leaf_path,
    results_workspace,
)
from fedorbit.artifacts.provenance import provenance_record
from fedorbit.artifacts.reuse import ArtifactStore
from fedorbit.artifacts.serialization import atomic_write_bytes, atomic_write_json
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.domain.enums import ArtifactState, ExperimentName

COORDINATES = {"experiment": "Primary Strict Cross-Telemetry Transfer"}


def _manifest(
    payload: Path, fingerprint: str, state: ArtifactState = ArtifactState.COMPLETED
) -> ReusableArtifactManifest:
    return ReusableArtifactManifest.model_validate(
        {
            "artifact_id": artifact_id("response_packet", COORDINATES, fingerprint),
            "artifact_type": "response_packet",
            "semantic_producer_coordinates": "{}",
            "producer_stage": "response",
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
            "state": state,
            "completion_manifest_sha256": "f" * 64,
        }
    )


def test_layout_matches_roadmap_roots(tmp_path: Path) -> None:
    config = load_fedorbit_config()
    layout = build_layout(config, root=tmp_path)
    assert layout.execution_root.name == "outputs"
    assert layout.manuscript_root.name == "results"
    assert layout.preprocessing == layout.execution_root / "preprocessing"
    assert layout.artifacts == layout.execution_root / "artifacts"
    assert layout.experiments == layout.execution_root / "experiments"
    assert layout.cache == layout.execution_root / "cache"
    assert layout.staging == layout.execution_root / "cache" / "staging"
    assert layout.results_experiments == layout.manuscript_root / "experiments"
    assert layout.project_summary == layout.manuscript_root / "project_summary"


def test_experiment_workspaces(tmp_path: Path) -> None:
    config = load_fedorbit_config()
    layout = build_layout(config, root=tmp_path)
    workspace = experiment_workspace(layout, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    assert workspace == layout.experiments / "primary-strict-cross-telemetry-transfer"
    results = results_workspace(layout, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    assert results == layout.results_experiments / "primary-strict-cross-telemetry-transfer"


def test_leaf_path_carries_coordinates_and_fingerprint(tmp_path: Path) -> None:
    config = load_fedorbit_config()
    layout = build_layout(config, root=tmp_path)
    path = leaf_path(
        layout,
        layout.artifacts,
        "primary-transfer.support=2",
        "a" * 64,
        ".parquet",
    )
    assert path.name.startswith("primary-transfer-support-2.")
    assert "aaaaaaaaaaaaaaaa" in path.name
    assert path.parent == layout.artifacts


def test_workspace_boundary_rejects_outside_paths(tmp_path: Path) -> None:
    config = load_fedorbit_config()
    layout = build_layout(config, root=tmp_path)
    with pytest.raises(WorkspaceError):
        enforce_workspace_boundary(layout, tmp_path / "outside.bin")
    with pytest.raises(WorkspaceError):
        enforce_workspace_boundary(layout, layout.execution_root)
    enforce_workspace_boundary(layout, layout.artifacts / "payload.bin")


def test_atomic_write_is_deterministic_and_complete(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "manifest.json"
    atomic_write_json(target, {"z": 1, "a": [2, 3]})
    content = target.read_text(encoding="utf-8")
    assert content == '{"a":[2,3],"z":1}\n'
    assert not list((tmp_path / "nested").glob(".tmp-*"))
    atomic_write_bytes(tmp_path / "payload.bin", b"bytes")
    assert (tmp_path / "payload.bin").read_bytes() == b"bytes"


def test_provenance_record_captures_all_components() -> None:
    config = load_fedorbit_config()
    payload = Path("/tmp/probe-payload.bin")
    payload.write_bytes(b"x")
    manifest = _manifest(payload, "fp-probe")
    record = provenance_record(config, manifest)
    assert record.artifact_id == manifest.artifact_id
    assert len(record.created_git_commit) == 40
    assert record.operating_system
    assert record.hardware
    assert record.driver
    assert len(record.environment_sha256) == 64


def test_evidence_writer_requires_verified_completed_artifact(tmp_path: Path) -> None:
    config = load_fedorbit_config()
    layout = build_layout(config, root=tmp_path)
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "packet.pt"
    payload.write_bytes(b"payload")
    manifest = _manifest(payload, "fp-evidence")
    store.write_reusable(manifest)
    writer = VerifiedEvidenceWriter(store, layout)

    destination = writer.write(
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        manifest.artifact_id,
        {"evidence": 1},
    )
    assert destination.exists()
    assert "primary-strict-cross-telemetry-transfer" in str(destination)


def test_evidence_writer_rejects_unverified_artifact(tmp_path: Path) -> None:
    config = load_fedorbit_config()
    layout = build_layout(config, root=tmp_path)
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "packet.pt"
    payload.write_bytes(b"payload")
    manifest = _manifest(payload, "fp-bad", state=ArtifactState.FAILED)
    store.write_reusable(manifest)
    writer = VerifiedEvidenceWriter(store, layout)
    with pytest.raises(EvidenceError):
        writer.write(
            ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
            manifest.artifact_id,
            {"evidence": 1},
        )


def test_evidence_writer_rejects_missing_payload(tmp_path: Path) -> None:
    config = load_fedorbit_config()
    layout = build_layout(config, root=tmp_path)
    store = ArtifactStore(tmp_path)
    missing = tmp_path / "missing.pt"
    missing.write_bytes(b"payload")
    manifest = _manifest(missing, "fp-missing")
    missing.unlink()
    store.write_reusable(manifest)
    writer = VerifiedEvidenceWriter(store, layout)
    with pytest.raises(EvidenceError):
        writer.write(
            ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
            manifest.artifact_id,
            {"evidence": 1},
        )
