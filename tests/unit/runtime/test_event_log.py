from __future__ import annotations

import json
from pathlib import Path

from structlog.testing import capture_logs
from tests.unit.artifacts.artifact_fixtures import (
    artifact_completion,
    artifact_manifest,
    payload_file,
)

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.runtime import (
    ExecutionEventLog,
    InfrastructureEventName,
    ResourcePhaseObservation,
    execution_logger,
)
from fedorbit.infrastructure.workspace import (
    ExperimentLogDirectory,
    PreprocessingDirectorySegment,
    PreprocessingPayloadFileName,
    experiment_event_log_path,
)
from fedorbit.types import (
    ArtifactFingerprint,
    ArtifactIdentifier,
    ArtifactStage,
    ExperimentName,
    FieldDescription,
    SemanticCoordinates,
    Sha256Digest,
)


def test_execution_event_log_appends_canonical_json_lines(tmp_path: Path) -> None:
    log = ExecutionEventLog(tmp_path / "logs" / "execution" / "events.jsonl")
    log.append(
        InfrastructureEventName.ARTIFACT_PROMOTED,
        {"artifact_id": "a" * 64, "elapsed_seconds": 0.5},
    )
    log.append(InfrastructureEventName.ARTIFACT_STALE, {"artifact_id": "b" * 64})
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "artifact_promoted"
    assert first["artifact_id"] == "a" * 64
    assert list(first) == sorted(first)
    assert json.loads(lines[1])["event"] == "artifact_stale"


def test_artifact_promotion_appends_to_the_experiment_execution_log(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "outputs")
    payload = payload_file(store.root / "artifacts", "payload.bin", b"payload")
    manifest = artifact_manifest(payload, Sha256Digest("1" * 64), ArtifactStage.RAW)
    store.promote_artifact(manifest, artifact_completion(manifest))
    log_path = experiment_event_log_path(
        store.root,
        ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION,
        ExperimentLogDirectory.EXECUTION,
    )
    assert log_path == (
        tmp_path
        / "outputs"
        / "experiments"
        / "mathematical-primitive-validation"
        / "logs"
        / "execution"
        / "events.jsonl"
    )
    recorded = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    promoted = [entry for entry in recorded if entry["event"] == "artifact_promoted"]
    assert len(promoted) == 1
    assert promoted[0]["artifact_id"] == manifest.artifact_id.value
    assert promoted[0]["stage"] == ArtifactStage.RAW.value
    assert promoted[0]["terminal_state"] == "Completed"
    assert promoted[0]["upstream_artifact_ids"] == []


def test_promotion_emits_resource_phase_telemetry(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "outputs")
    payload = payload_file(store.root / "artifacts", "payload.bin", b"payload")
    manifest = artifact_manifest(payload, Sha256Digest("2" * 64), ArtifactStage.RAW)
    with capture_logs() as logs:
        store.promote_artifact(manifest, artifact_completion(manifest))
    phases = [entry for entry in logs if entry["event"] == "resource_phase"]
    assert len(phases) == 1
    assert phases[0]["phase"] == f"artifact-promotion:{ArtifactStage.RAW.value}"
    assert phases[0]["elapsed_seconds"] >= 0.0
    assert phases[0]["peak_host_rss_mib"] > 0.0
    assert phases[0]["peak_cuda_allocated_bytes"] == 0


def test_reuse_abstention_telemetry_is_emitted_for_unrecorded_lineage(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "outputs")
    payload = payload_file(store.root / "preprocessing", "prepared.bin", b"prepared")
    manifest = artifact_manifest(payload, Sha256Digest("3" * 64), ArtifactStage.PREPROCESSING)
    store.promote_artifact(manifest, artifact_completion(manifest))
    with capture_logs() as logs:
        assert store.find_by_fingerprint(ArtifactFingerprint("3" * 64)) is None
    abstentions = [entry for entry in logs if entry["event"] == "abstention"]
    assert len(abstentions) == 1
    assert "reuse abstained" in abstentions[0]["reason"]
    assert abstentions[0]["cell_coordinates"] == manifest.semantic_producer_coordinates


def test_stale_retirement_emits_a_warning_and_a_retirement_event(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "outputs")
    workspace = store.root / "artifacts"
    parent_payload = payload_file(workspace, "parent.bin", b"parent")
    parent = artifact_manifest(parent_payload, Sha256Digest("4" * 64), ArtifactStage.RAW)
    store.promote_artifact(parent, artifact_completion(parent))
    child_payload = payload_file(workspace, "child.bin", b"child")
    child = artifact_manifest(
        child_payload,
        Sha256Digest("5" * 64),
        ArtifactStage.PREPROCESSING,
        upstream_artifact_ids=(parent.artifact_id,),
    )
    store.promote_artifact(child, artifact_completion(child))
    with capture_logs() as logs:
        retired = store.retire_superseded_parent(parent.artifact_id, ArtifactIdentifier("6" * 64))
    assert retired == (child.artifact_id,)
    warnings = [entry for entry in logs if entry["event"] == "lineage_warning"]
    events = [entry for entry in logs if entry["event"] == "artifact_retired"]
    assert len(warnings) == 1
    assert len(events) == 1
    assert events[0]["stale_descendants"] == [child.artifact_id.value]
    assert events[0]["superseded"] == parent.artifact_id.value


def test_structured_telemetry_helpers_record_actionable_fields() -> None:
    logger = execution_logger()
    with capture_logs() as logs:
        logger.resource_phase(
            ResourcePhaseObservation(
                phase=FieldDescription("final-source-response-band"),
                elapsed_seconds=12.5,
                peak_host_rss_mib=1024.0,
                peak_cuda_allocated_bytes=2048,
            )
        )
        logger.abstention(
            SemanticCoordinates("cell"),
            FieldDescription("insufficient evidence"),
        )
        logger.warning(FieldDescription("stale descendants retired"))
    names = [entry["event"] for entry in logs]
    assert names == ["resource_phase", "abstention", "lineage_warning"]
    assert logs[0]["elapsed_seconds"] == 12.5
    assert logs[1]["cell_coordinates"] == "cell"
    assert logs[2]["message"] == "stale descendants retired"


def test_path_segments_match_the_configuration_authority() -> None:
    layout = active_config().runtime.artifact_layout
    assert ExperimentLogDirectory.EXECUTION.value in layout.experiment_subdirectories.logs
    assert ExperimentLogDirectory.FAILURES.value in layout.experiment_subdirectories.logs
    assert PreprocessingDirectorySegment.SPLITS.value in layout.preprocessing_subdirectories
    assert PreprocessingPayloadFileName.DATASET_MANIFEST.value == "data.json"
    assert PreprocessingPayloadFileName.MATERIALIZED_CLIENT.value == "client.pt"
    assert PreprocessingPayloadFileName.SPLIT_PAYLOAD.value == "data.parquet"
    assert layout.staging_directory in layout.cache_subdirectories


def test_store_staging_root_follows_the_configured_layout(tmp_path: Path) -> None:
    from fedorbit.infrastructure.workspace import build_layout

    layout = build_layout(tmp_path)
    store = ArtifactStore(layout.execution_root, layout.staging)
    assert store.staging_dir() == layout.staging
    assert store.staging_dir() == tmp_path / "outputs" / "cache" / "staging"
    detached = ArtifactStore(tmp_path / "outputs")
    assert detached.staging_dir() == tmp_path / "outputs" / "staging"
