from __future__ import annotations

import pytest

from fedorbit.domain.enums import ClientRole
from fedorbit.strict_interface.packet import (
    PACKET_PERMITTED_FIELDS,
    PacketError,
    SourcePacket,
)
from fedorbit.strict_interface.resources import (
    SOURCE_LOCAL_WHITELIST,
    TARGET_LOCAL_WHITELIST,
    ResourceKind,
    StrictResourcePolicy,
    StrictResourceViolationError,
)
from fedorbit.strict_interface.trace import AccessLogger

POLICY = StrictResourcePolicy()


def test_source_local_whitelist_membership() -> None:
    for resource in (
        ResourceKind.TRAIN,
        ResourceKind.META,
        ResourceKind.VALID,
        ResourceKind.LOCAL_FINE_LABELS,
        ResourceKind.LOCAL_COARSE_GROUP_MEMBERSHIP,
        ResourceKind.LOCAL_PREPROCESSING,
        ResourceKind.LOCAL_CLASSIFIER,
        ResourceKind.LOCAL_OPTIMIZER_CHECKPOINT,
        ResourceKind.LOCAL_INTERVENTION_EXPERIMENTS,
    ):
        assert resource in SOURCE_LOCAL_WHITELIST
        POLICY.assert_source_allowed(resource)


def test_source_local_rejects_off_whitelist() -> None:
    for resource in (
        ResourceKind.CONFIRM,
        ResourceKind.TEST,
        ResourceKind.ANONYMOUS_SOURCE_PACKET,
        ResourceKind.LOCAL_IMPORTANCE_VECTOR,
    ):
        assert resource not in SOURCE_LOCAL_WHITELIST
        with pytest.raises(StrictResourceViolationError):
            POLICY.assert_source_allowed(resource)


def test_target_local_whitelist_membership() -> None:
    for resource in (
        ResourceKind.TRAIN,
        ResourceKind.META,
        ResourceKind.VALID,
        ResourceKind.CONFIRM,
        ResourceKind.LOCAL_FINE_LABELS,
        ResourceKind.SHARED_COARSE_GROUPS,
        ResourceKind.ANONYMOUS_SOURCE_PACKET,
        ResourceKind.LOCAL_MODEL_STATE,
        ResourceKind.LOCAL_IMPORTANCE_VECTOR,
    ):
        assert resource in TARGET_LOCAL_WHITELIST
        POLICY.assert_target_allowed(resource, transfer_finalized=True)


def test_target_test_requires_finalized_transfer() -> None:
    with pytest.raises(StrictResourceViolationError):
        POLICY.assert_target_allowed(ResourceKind.TEST, transfer_finalized=False)
    POLICY.assert_target_allowed(ResourceKind.TEST, transfer_finalized=True)


def test_target_rejects_off_whitelist() -> None:
    with pytest.raises(StrictResourceViolationError):
        POLICY.assert_target_allowed(ResourceKind.LOCAL_PREPROCESSING, transfer_finalized=True)
    with pytest.raises(StrictResourceViolationError):
        POLICY.assert_target_allowed(ResourceKind.LOCAL_CLASSIFIER, transfer_finalized=True)


def test_role_allowed_dispatch() -> None:
    POLICY.assert_role_allowed(ClientRole.SOURCE, ResourceKind.TRAIN)
    POLICY.assert_role_allowed(ClientRole.TARGET, ResourceKind.CONFIRM)
    with pytest.raises(StrictResourceViolationError):
        POLICY.assert_role_allowed(ClientRole.SOURCE, ResourceKind.CONFIRM)


def test_access_logger_records_permitted_access() -> None:
    logger = AccessLogger()
    logger.record(ClientRole.SOURCE, ResourceKind.TRAIN)
    logger.record(ClientRole.TARGET, ResourceKind.ANONYMOUS_SOURCE_PACKET)
    logger.record(ClientRole.TARGET, ResourceKind.TEST, transfer_finalized=True)
    trace = logger.trace()
    assert len(trace.events) == 3
    logger.validate()


def test_access_logger_rejects_violation() -> None:
    logger = AccessLogger()
    with pytest.raises(StrictResourceViolationError):
        logger.record(ClientRole.SOURCE, ResourceKind.CONFIRM)
    with pytest.raises(StrictResourceViolationError):
        logger.record(ClientRole.TARGET, ResourceKind.TEST, transfer_finalized=False)
    assert len(logger.trace().events) == 0


def _packet(integrity_sha256: str = "") -> SourcePacket:
    return SourcePacket(
        anonymous_fine_node_ids=("node-1", "node-2"),
        exposed_coarse_group_id="coarse-1",
        L=0.1,
        U=0.9,
        per_node_train_support=(1.0, 2.0),
        per_node_meta_support=(0.5, 0.5),
        per_node_effective_replicate_count=(1, 1),
        packet_schema_metadata="schema-v1",
        source_checkpoint_sha256="a" * 64,
        response_configuration_sha256="b" * 64,
        packet_integrity_sha256=integrity_sha256,
        packet_validity_state="valid",
    )


def test_packet_permitted_fields_exact() -> None:
    assert (
        frozenset(
            {
                "anonymous_fine_node_ids",
                "exposed_coarse_group_id",
                "L",
                "U",
                "per_node_train_support",
                "per_node_meta_support",
                "per_node_effective_replicate_count",
                "packet_schema_metadata",
                "source_checkpoint_sha256",
                "response_configuration_sha256",
                "packet_integrity_sha256",
                "packet_validity_state",
                "technical_creation_timestamp",
            }
        )
        == PACKET_PERMITTED_FIELDS
    )


def test_packet_integrity_excludes_self_and_timestamp() -> None:
    packet = _packet()
    payload = packet.integrity_payload()
    assert "packet_integrity_sha256" not in payload
    assert "technical_creation_timestamp" not in payload
    expected = __import__("hashlib").sha256(payload.encode("utf-8")).hexdigest()
    assert packet.compute_integrity_sha256() == expected


def test_packet_integrity_is_stable_and_field_sensitive() -> None:
    first = _packet().compute_integrity_sha256()
    second = _packet().compute_integrity_sha256()
    assert first == second
    base = _packet()
    altered = SourcePacket(
        anonymous_fine_node_ids=("node-1", "node-3"),
        exposed_coarse_group_id=base.exposed_coarse_group_id,
        L=base.L,
        U=base.U,
        per_node_train_support=base.per_node_train_support,
        per_node_meta_support=base.per_node_meta_support,
        per_node_effective_replicate_count=base.per_node_effective_replicate_count,
        packet_schema_metadata=base.packet_schema_metadata,
        source_checkpoint_sha256=base.source_checkpoint_sha256,
        response_configuration_sha256=base.response_configuration_sha256,
        packet_integrity_sha256=base.packet_integrity_sha256,
        packet_validity_state=base.packet_validity_state,
    )
    assert altered.compute_integrity_sha256() != first


def test_packet_validates_integrity() -> None:
    packet = _packet()
    valid = SourcePacket(
        anonymous_fine_node_ids=packet.anonymous_fine_node_ids,
        exposed_coarse_group_id=packet.exposed_coarse_group_id,
        L=packet.L,
        U=packet.U,
        per_node_train_support=packet.per_node_train_support,
        per_node_meta_support=packet.per_node_meta_support,
        per_node_effective_replicate_count=packet.per_node_effective_replicate_count,
        packet_schema_metadata=packet.packet_schema_metadata,
        source_checkpoint_sha256=packet.source_checkpoint_sha256,
        response_configuration_sha256=packet.response_configuration_sha256,
        packet_integrity_sha256=packet.compute_integrity_sha256(),
        packet_validity_state="valid",
    )
    valid.validate()
    corrupted = SourcePacket(
        anonymous_fine_node_ids=packet.anonymous_fine_node_ids,
        exposed_coarse_group_id=packet.exposed_coarse_group_id,
        L=0.999,
        U=packet.U,
        per_node_train_support=packet.per_node_train_support,
        per_node_meta_support=packet.per_node_meta_support,
        per_node_effective_replicate_count=packet.per_node_effective_replicate_count,
        packet_schema_metadata=packet.packet_schema_metadata,
        source_checkpoint_sha256=packet.source_checkpoint_sha256,
        response_configuration_sha256=packet.response_configuration_sha256,
        packet_integrity_sha256=packet.compute_integrity_sha256(),
        packet_validity_state="valid",
    )
    with pytest.raises(PacketError):
        corrupted.validate()


def test_packet_rejects_forbidden_content() -> None:
    packet = _packet()
    forbidden = SourcePacket(
        anonymous_fine_node_ids=packet.anonymous_fine_node_ids,
        exposed_coarse_group_id=packet.exposed_coarse_group_id,
        L=packet.L,
        U=packet.U,
        per_node_train_support=packet.per_node_train_support,
        per_node_meta_support=packet.per_node_meta_support,
        per_node_effective_replicate_count=packet.per_node_effective_replicate_count,
        packet_schema_metadata=packet.packet_schema_metadata,
        source_checkpoint_sha256=packet.source_checkpoint_sha256,
        response_configuration_sha256=packet.response_configuration_sha256,
        packet_integrity_sha256=packet.compute_integrity_sha256(),
        packet_validity_state="valid",
        forbidden_content=("raw source samples",),
    )
    with pytest.raises(StrictResourceViolationError):
        forbidden.validate()
