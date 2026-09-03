from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.analysis.records import MetricDirection, MetricRecord
from fedorbit.infrastructure.execution import ArtifactStore, atomic_write_bytes, atomic_write_json
from fedorbit.infrastructure.manifests import ReusableArtifactManifest, artifact_id, file_sha256
from fedorbit.infrastructure.provenance import provenance_record
from fedorbit.infrastructure.workspace import (
    WorkspaceError,
    build_layout,
    enforce_workspace_boundary,
    experiment_workspace,
    leaf_path,
    results_workspace,
)
from fedorbit.reporting import EvidenceExportError, VerifiedEvidenceWriter
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactState,
    ExperimentName,
    MetricId,
    TransferMethod,
)

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


def test_layout_matches_workspace_roots(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
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
    layout = build_layout(root=tmp_path)
    workspace = experiment_workspace(layout, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    assert workspace == layout.experiments / "primary-strict-cross-telemetry-transfer"
    results = results_workspace(layout, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    assert results == layout.results_experiments / "primary-strict-cross-telemetry-transfer"


def test_leaf_path_carries_coordinates_and_fingerprint(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
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
    layout = build_layout(root=tmp_path)
    with pytest.raises(WorkspaceError):
        enforce_workspace_boundary(layout, tmp_path / "outside.bin")
    with pytest.raises(WorkspaceError):
        enforce_workspace_boundary(layout, layout.execution_root)
    enforce_workspace_boundary(layout, layout.artifacts / "payload.bin")


def test_atomic_write_is_deterministic_and_complete(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "manifest.json"
    atomic_write_json(target, {"z": 1, "a": [2, 3]})
    assert target.read_text(encoding="utf-8") == '{"a":[2,3],"z":1}\n'
    assert not list((tmp_path / "nested").glob(".tmp-*"))
    atomic_write_bytes(tmp_path / "payload.bin", b"bytes")
    assert (tmp_path / "payload.bin").read_bytes() == b"bytes"


def test_provenance_record_captures_all_components(tmp_path: Path) -> None:
    payload = tmp_path / "probe-payload.bin"
    payload.write_bytes(b"x")
    manifest = _manifest(payload, "fp-probe")
    record = provenance_record(manifest)
    assert record.artifact_id == manifest.artifact_id
    assert len(record.created_git_commit) == 40
    assert record.operating_system
    assert record.hardware
    assert record.driver
    assert len(record.environment_sha256) == 64


def test_evidence_writer_requires_verified_completed_artifact(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "packet.pt"
    payload.write_bytes(b"payload")
    manifest = _manifest(payload, "fp-evidence")
    store.write_reusable(manifest)
    destination = VerifiedEvidenceWriter(store, layout).write(
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        ArtifactIdentifier(manifest.artifact_id),
        {"evidence": 1},
    )
    assert destination.exists()
    assert "primary-strict-cross-telemetry-transfer" in str(destination)


def test_evidence_writer_reuses_matching_export_without_overwriting(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "packet.pt"
    payload.write_bytes(b"payload")
    manifest = _manifest(payload, "fp-idempotent")
    store.write_reusable(manifest)
    writer = VerifiedEvidenceWriter(store, layout)
    destination = writer.write(
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        ArtifactIdentifier(manifest.artifact_id),
        {"evidence": 1},
    )
    assert (
        writer.write(
            ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
            ArtifactIdentifier(manifest.artifact_id),
            {"evidence": 1},
        )
        == destination
    )


def test_evidence_writer_exports_validated_metric_record(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(tmp_path)
    metric = MetricRecord(
        experiment=ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION,
        pair="synthetic",
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        condition="generated",
        seed=1103,
        metric_name=MetricId.ACTIVE_IMAGE_CANDIDATES,
        metric_value=2.0,
        metric_unit="count",
        direction=MetricDirection.DESCRIPTIVE,
        evaluation_class_set_sha256="a" * 64,
        input_artifact_ids=("synthetic-generator",),
        dependency_fingerprint_sha256="b" * 64,
        valid=True,
        invalid_reason=None,
    )
    payload = tmp_path / "metric.json"
    atomic_write_json(payload, {"metric_record": metric.model_dump(mode="json")})
    manifest = _manifest(payload, "fp-metric")
    store.write_reusable(manifest)

    paths = VerifiedEvidenceWriter(store, layout).write_metric_exports(
        ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION,
        ArtifactIdentifier(manifest.artifact_id),
    )

    assert tuple(path.name for path in paths) == (
        "summary.json",
        "metric_records.csv",
        "metric_records.tex",
        "metric_value.svg",
        "metric_value.pdf",
    )
    assert paths[0].read_text(encoding="utf-8").startswith("{")
    assert "Active-Image Candidates" in paths[1].read_text(encoding="utf-8")
    assert "Active-Image Candidates" in paths[3].read_text(encoding="utf-8")
    assert paths[4].read_bytes().startswith(b"%PDF-1.4")

    summary_paths = VerifiedEvidenceWriter(store, layout).write_project_summary(
        (manifest,),
        (metric,),
    )

    assert tuple(path.name for path in summary_paths) == (
        "experiments.csv",
        "evidence_summary.csv",
        "summary.json",
        "scientific_configuration.json",
        "execution.json",
    )
    assert "Active-Image Candidates" in summary_paths[1].read_text(encoding="utf-8")


def test_evidence_writer_requires_explicit_overwrite_for_changed_export(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "packet.pt"
    payload.write_bytes(b"payload")
    manifest = _manifest(payload, "fp-overwrite")
    store.write_reusable(manifest)
    writer = VerifiedEvidenceWriter(store, layout)
    writer.write(
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        ArtifactIdentifier(manifest.artifact_id),
        {"evidence": 1},
    )
    with pytest.raises(EvidenceExportError, match="--overwrite"):
        writer.write(
            ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
            ArtifactIdentifier(manifest.artifact_id),
            {"evidence": 2},
        )
    destination = writer.write(
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        ArtifactIdentifier(manifest.artifact_id),
        {"evidence": 2},
        overwrite=True,
    )
    assert destination.read_text(encoding="utf-8") == '{"evidence":2}\n'


def test_evidence_writer_rejects_unverified_artifact(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "packet.pt"
    payload.write_bytes(b"payload")
    manifest = _manifest(payload, "fp-bad", state=ArtifactState.FAILED)
    store.write_reusable(manifest)
    with pytest.raises(EvidenceExportError):
        VerifiedEvidenceWriter(store, layout).write(
            ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
            ArtifactIdentifier(manifest.artifact_id),
            {"evidence": 1},
        )


def test_evidence_writer_rejects_missing_payload(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(tmp_path)
    missing = tmp_path / "missing.pt"
    missing.write_bytes(b"payload")
    manifest = _manifest(missing, "fp-missing")
    missing.unlink()
    store.write_reusable(manifest)
    with pytest.raises(EvidenceExportError):
        VerifiedEvidenceWriter(store, layout).write(
            ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
            ArtifactIdentifier(manifest.artifact_id),
            {"evidence": 1},
        )
