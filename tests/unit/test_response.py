from __future__ import annotations

import pytest

from fedorbit.response.packet import SourcePacket, build_source_packet
from fedorbit.response.uncertainty import FinalResponseEntry, FinalResponseEstimate


def _estimate() -> FinalResponseEstimate:
    return FinalResponseEstimate(
        entries=(
            FinalResponseEntry(0, 0, 0.3, 0.05, 0.1, 0.5, True),
            FinalResponseEntry(1, 0, -0.2, 0.04, -0.4, -0.01, True),
        ),
        critical_value=4.0,
        useful_intervention_columns=1,
        median_band_width_ratio=1.3,
        stability_rule_passed=True,
    )


def _packet(timestamp: str = "2026-08-22T00:00:00Z") -> SourcePacket:
    return build_source_packet(
        _estimate(),
        anonymous_fine_node_ids=("node-0001", "node-0002"),
        exposed_coarse_group_id="Disruption",
        per_node_train_support=(120, 90),
        per_node_meta_support=(30, 20),
        per_node_effective_replicate_count=(24, 24),
        source_checkpoint_sha256="a" * 64,
        response_configuration_sha256="b" * 64,
        creation_timestamp=timestamp,
    )


def test_packet_uses_exact_anonymous_identifiers_and_integrity() -> None:
    packet = _packet()
    packet.validate()
    assert packet.anonymous_fine_node_ids == ("node-0001", "node-0002")
    assert packet.packet_integrity_sha256 == packet.compute_integrity_sha256()
    assert '"dtype":"float64"' in packet.integrity_payload()
    assert '"order":"C"' in packet.integrity_payload()


def test_timestamp_does_not_change_scientific_integrity() -> None:
    first = _packet("2026-08-22T00:00:00Z")
    second = _packet("2026-08-23T00:00:00Z")
    assert first.packet_integrity_sha256 == second.packet_integrity_sha256
    assert first.payload_sha256() != second.payload_sha256()


def test_packet_rejects_semantic_or_nonstable_node_ids() -> None:
    packet = _packet()
    invalid = SourcePacket(
        anonymous_fine_node_ids=("ddos", "ransomware"),
        exposed_coarse_group_id=packet.exposed_coarse_group_id,
        L=packet.L,
        U=packet.U,
        per_node_train_support=packet.per_node_train_support,
        per_node_meta_support=packet.per_node_meta_support,
        per_node_effective_replicate_count=packet.per_node_effective_replicate_count,
        packet_schema_metadata=packet.packet_schema_metadata,
        source_checkpoint_sha256=packet.source_checkpoint_sha256,
        response_configuration_sha256=packet.response_configuration_sha256,
        packet_integrity_sha256=packet.packet_integrity_sha256,
        packet_validity_state=packet.packet_validity_state,
        technical_creation_timestamp=packet.technical_creation_timestamp,
    )
    with pytest.raises(PermissionError):
        invalid.validate()


def test_packet_rejects_invalid_timestamp_and_hashes() -> None:
    packet = _packet()
    invalid = SourcePacket(
        anonymous_fine_node_ids=packet.anonymous_fine_node_ids,
        exposed_coarse_group_id=packet.exposed_coarse_group_id,
        L=packet.L,
        U=packet.U,
        per_node_train_support=packet.per_node_train_support,
        per_node_meta_support=packet.per_node_meta_support,
        per_node_effective_replicate_count=packet.per_node_effective_replicate_count,
        packet_schema_metadata=packet.packet_schema_metadata,
        source_checkpoint_sha256="not-a-digest",
        response_configuration_sha256=packet.response_configuration_sha256,
        packet_integrity_sha256=packet.packet_integrity_sha256,
        packet_validity_state=packet.packet_validity_state,
        technical_creation_timestamp="2026-08-22 00:00:00",
    )
    with pytest.raises(PermissionError):
        invalid.validate()
