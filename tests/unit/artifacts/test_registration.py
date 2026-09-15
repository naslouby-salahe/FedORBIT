from __future__ import annotations

from pathlib import Path

import pytest
from structlog.testing import capture_logs
from tests.unit.artifacts.artifact_fixtures import artifact_completion, artifact_manifest

from fedorbit.infrastructure.artifacts import (
    ArtifactAccessContext,
    ArtifactPayloadRequest,
    ArtifactStore,
    unconsumed_artifact_ids,
    verify_registered_artifact,
)
from fedorbit.infrastructure.environment import observed_hardware
from fedorbit.infrastructure.manifests import (
    ArtifactProvenance,
    ProvenanceFamilyEvidence,
)
from fedorbit.infrastructure.provenance import (
    ARTIFACT_FAMILY_REGISTRATION,
    SUBSYSTEM_REGISTRATION,
    artifact_family_registration,
    subsystem_stage,
    validate_artifact_family_registry,
)
from fedorbit.infrastructure.storage import PayloadIntegrityError, serialized_payload_bytes
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactPath,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    ConfigurationSection,
    OverwritePolicy,
    ProvenanceAccessRole,
    ProvenanceArtifactFamily,
    ScientificSubsystem,
    SemanticCoordinateText,
    SerializedPacket,
    Sha256Digest,
)

PACKET_COORDINATES = SemanticCoordinateText(
    '{"experiment":"Final Source-Response Band Validation"}'
)
PACKET_SERIALIZED = SerializedPacket('{"packet_integrity_sha256":"' + "a" * 64 + '"}')


def _packet_request(
    payload_path: Path,
    upstream_artifact_ids: ArtifactIdentifiers = (),
) -> ArtifactPayloadRequest:
    return ArtifactPayloadRequest(
        artifact_type=ArtifactType.RESPONSE_PACKET,
        coordinates={"packet": "final"},
        semantic_coordinates=PACKET_COORDINATES,
        stage=ArtifactStage.RESPONSE,
        payload_path=ArtifactPath(payload_path),
        serialized=PACKET_SERIALIZED,
        upstream_artifact_ids=upstream_artifact_ids,
        access=ArtifactAccessContext(subsystem=ScientificSubsystem.SOURCE_RESPONSE),
    )


def _published_upstream(store: ArtifactStore, fingerprint: str = "9" * 64) -> ArtifactIdentifier:
    payload = store.root / "upstream" / "source.bin"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"source")
    manifest = artifact_manifest(payload, Sha256Digest(fingerprint), ArtifactStage.RAW)
    store.promote_artifact(manifest, artifact_completion(manifest))
    return manifest.artifact_id


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "outputs")


def test_every_artifact_family_and_subsystem_is_registered() -> None:
    validate_artifact_family_registry()
    assert set(ARTIFACT_FAMILY_REGISTRATION) == set(ArtifactType)
    assert set(SUBSYSTEM_REGISTRATION) == set(ScientificSubsystem)
    for family in (
        ArtifactType.RESPONSE_PACKET,
        ArtifactType.TARGET_IMPORTANCE,
        ArtifactType.SOLVER_RESULT,
        ArtifactType.CONFIRMATION_INPUT,
    ):
        registration = artifact_family_registration(family)
        assert registration.stage is not None
        assert registration.cache_key_material
    assert (
        artifact_family_registration(ArtifactType.RESPONSE_PACKET).stage is ArtifactStage.RESPONSE
    )
    assert artifact_family_registration(
        ArtifactType.CONFIRMATION_INPUT
    ).cache_key_material == frozenset(
        {ConfigurationSection.CONFIRMATION, ConfigurationSection.MODELS}
    )
    assert subsystem_stage(ScientificSubsystem.TARGET_IMPORTANCE) is ArtifactStage.TARGET_IMPORTANCE
    assert subsystem_stage(ScientificSubsystem.SOURCE_RESPONSE) is ArtifactStage.RESPONSE


def test_registered_payload_write_and_verify_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    upstream = _published_upstream(store)
    payload_path = store.root / "artifacts" / "packets" / "packet.json"
    manifest = store.register_artifact_payload(
        _packet_request(payload_path, (upstream,)), OverwritePolicy.REPLACE
    )
    assert manifest.artifact_type is ArtifactType.RESPONSE_PACKET
    assert manifest.producer_stage is ArtifactStage.RESPONSE
    assert payload_path.is_file()
    assert payload_path.read_bytes() == serialized_payload_bytes(PACKET_SERIALIZED)
    assert not store.staging_area(manifest.artifact_id).is_dir()
    assert store.is_reusable(manifest.artifact_id) is True
    assert (
        verify_registered_artifact(store, manifest.artifact_id).artifact_id == manifest.artifact_id
    )
    reused = store.register_artifact_payload(
        _packet_request(payload_path, (upstream,)), OverwritePolicy.REUSE
    )
    assert reused.artifact_id == manifest.artifact_id
    recorded = store.read_completion(manifest.artifact_id)
    assert recorded.upstream_artifact_ids == (upstream,)


def test_registration_rejects_a_stage_that_disagrees_with_the_registered_family(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    request = _packet_request(store.root / "artifacts" / "packet.json")
    mismatched = ArtifactPayloadRequest(
        artifact_type=request.artifact_type,
        coordinates=request.coordinates,
        semantic_coordinates=request.semantic_coordinates,
        stage=ArtifactStage.SCORING,
        payload_path=request.payload_path,
        serialized=request.serialized,
    )
    with pytest.raises(ValueError, match="registered at stage response"):
        store.register_artifact_payload(mismatched, OverwritePolicy.REPLACE)


def test_tampered_registered_payload_is_rejected_on_read(tmp_path: Path) -> None:
    store = _store(tmp_path)
    upstream = _published_upstream(store)
    payload_path = store.root / "artifacts" / "packets" / "packet.json"
    manifest = store.register_artifact_payload(
        _packet_request(payload_path, (upstream,)), OverwritePolicy.REPLACE
    )
    payload_path.write_bytes(b'{"tampered":true}\n')
    with pytest.raises(PayloadIntegrityError, match="checksum mismatch"):
        store.register_artifact_payload(
            _packet_request(payload_path, (upstream,)), OverwritePolicy.REPLACE
        )
    assert store.artifact_state(manifest.artifact_id).state is ArtifactState.INVALID
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_registered_artifact(store, manifest.artifact_id)


def test_registered_write_records_access_and_unconsumed_telemetry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload_path = store.root / "artifacts" / "packets" / "packet.json"
    upstream = _published_upstream(store)
    with capture_logs() as logs:
        manifest = store.register_artifact_payload(
            _packet_request(payload_path, (upstream,)), OverwritePolicy.REPLACE
        )
    execution_events = [entry for entry in logs if entry["event"] == "execution_event"]
    assert len(execution_events) == 1
    assert execution_events[0]["subsystem"] == ScientificSubsystem.SOURCE_RESPONSE.value
    assert execution_events[0]["unconsumed_artifact_count"] == 1
    assert store.unconsumed_artifacts() == (manifest.artifact_id,)
    assert execution_events[0]["upstream_artifact_ids"] == [upstream.value]
    events_path = (
        store.root
        / "experiments"
        / "final-source-response-band-validation"
        / "logs"
        / "execution"
        / "events.jsonl"
    )
    assert events_path.is_file()


def test_unconsumed_artifact_ids_reports_producers_without_consumers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = store.root / "payloads"
    producer_payload = workspace / "producer.bin"
    producer_payload.parent.mkdir(parents=True, exist_ok=True)
    producer_payload.write_bytes(b"producer")
    producer = artifact_manifest(producer_payload, Sha256Digest("1" * 64), ArtifactStage.RAW)
    store.promote_artifact(producer, artifact_completion(producer))
    consumer_payload = workspace / "consumer.bin"
    consumer_payload.write_bytes(b"consumer")
    consumer = artifact_manifest(
        consumer_payload,
        Sha256Digest("2" * 64),
        ArtifactStage.PREPROCESSING,
        upstream_artifact_ids=(producer.artifact_id,),
    )
    assert unconsumed_artifact_ids((producer,)) == (producer.artifact_id,)
    assert unconsumed_artifact_ids((producer, consumer)) == (consumer.artifact_id,)
    store.promote_artifact(consumer, artifact_completion(consumer))
    assert store.unconsumed_artifacts() == (consumer.artifact_id,)


def test_recorded_provenance_keeps_family_evidence_hardware_and_access_trace(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    payload_path = store.root / "artifacts" / "packets" / "packet.json"
    checkpoint_sha = Sha256Digest("c" * 64)
    provenance = ArtifactProvenance(
        family_evidence=(
            ProvenanceFamilyEvidence(
                family=ProvenanceArtifactFamily.BASE_CHECKPOINT, sha256=checkpoint_sha
            ),
            ProvenanceFamilyEvidence(
                family=ProvenanceArtifactFamily.ELIGIBILITY, sha256=Sha256Digest("e" * 64)
            ),
        ),
        access_trace=(),
        hardware=observed_hardware(),
    )
    request = _packet_request(payload_path)
    manifest = store.register_artifact_payload(
        ArtifactPayloadRequest(
            artifact_type=request.artifact_type,
            coordinates=request.coordinates,
            semantic_coordinates=request.semantic_coordinates,
            stage=request.stage,
            payload_path=request.payload_path,
            serialized=request.serialized,
            upstream_artifact_ids=request.upstream_artifact_ids,
            access=ArtifactAccessContext(
                subsystem=ScientificSubsystem.SOURCE_RESPONSE,
                artifacts_read=(),
                provenance=provenance,
            ),
        ),
        OverwritePolicy.REPLACE,
    )
    completion = store.read_completion(manifest.artifact_id)
    from fedorbit.infrastructure.reuse import recorded_lineage

    recorded = recorded_lineage(completion)
    assert recorded.provenance is not None
    recorded_evidence = {
        evidence.family: evidence.sha256 for evidence in recorded.provenance.family_evidence
    }
    assert recorded_evidence[ProvenanceArtifactFamily.BASE_CHECKPOINT] == checkpoint_sha
    assert recorded.provenance.hardware is not None
    assert ProvenanceArtifactFamily.TARGET_IMPORTANCE not in recorded_evidence
    assert ProvenanceAccessRole.OUTPUT.value == "output"


def test_registered_write_promotes_from_staging_and_leaves_no_staged_payload(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    upstream = _published_upstream(store)
    payload_path = store.root / "artifacts" / "packets" / "packet.json"
    assert not payload_path.exists()
    manifest = store.register_artifact_payload(
        _packet_request(payload_path, (upstream,)), OverwritePolicy.REPLACE
    )
    assert payload_path.is_file()
    assert not store.staging_area(manifest.artifact_id).is_dir()
    assert not store.completion_path(manifest.artifact_id).with_suffix(".staged").exists()
    assert store.manifest_path(manifest.artifact_id).is_file()


def test_failed_registration_leaves_no_active_payload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload_path = store.root / "artifacts" / "packets" / "packet.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(b'{"preexisting":true}\n')
    with pytest.raises(PayloadIntegrityError):
        store.register_artifact_payload(_packet_request(payload_path), OverwritePolicy.REPLACE)
    assert store.all_manifests() == ()
