from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Mapping

from fedorbit.domain.canonical import canonical_json
from fedorbit.strict_interface.resources import StrictResourceViolationError

PACKET_PERMITTED_FIELDS = frozenset(
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

FORBIDDEN_PACKET_CONTENT = frozenset(
    {
        "fine semantic names",
        "raw source samples",
        "source feature names as alignment bridge",
        "model parameters",
        "gradients",
        "shared embeddings",
        "prototypes",
    }
)


class PacketError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourcePacket:
    anonymous_fine_node_ids: tuple[str, ...]
    exposed_coarse_group_id: str
    L: tuple[float, ...]
    U: tuple[float, ...]
    per_node_train_support: tuple[float, ...]
    per_node_meta_support: tuple[float, ...]
    per_node_effective_replicate_count: tuple[int, ...]
    packet_schema_metadata: str
    source_checkpoint_sha256: str
    response_configuration_sha256: str
    packet_integrity_sha256: str
    packet_validity_state: str
    technical_creation_timestamp: str = ""

    def integrity_payload(self) -> str:
        values: dict[str, object] = {
            "anonymous_fine_node_ids": list(self.anonymous_fine_node_ids),
            "exposed_coarse_group_id": self.exposed_coarse_group_id,
            "L": list(self.L),
            "U": list(self.U),
            "per_node_train_support": list(self.per_node_train_support),
            "per_node_meta_support": list(self.per_node_meta_support),
            "per_node_effective_replicate_count": list(self.per_node_effective_replicate_count),
            "packet_schema_metadata": self.packet_schema_metadata,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "response_configuration_sha256": self.response_configuration_sha256,
            "packet_validity_state": self.packet_validity_state,
        }
        return canonical_json(values)

    def compute_integrity_sha256(self) -> str:
        return hashlib.sha256(self.integrity_payload().encode("utf-8")).hexdigest()

    def validate(self) -> None:
        actual_fields = frozenset(field.name for field in fields(self))
        if actual_fields != PACKET_PERMITTED_FIELDS:
            unexpected = sorted(actual_fields - PACKET_PERMITTED_FIELDS)
            missing = sorted(PACKET_PERMITTED_FIELDS - actual_fields)
            raise PacketError(f"invalid packet schema; unexpected={unexpected}, missing={missing}")
        validate_packet_mapping({field.name: getattr(self, field.name) for field in fields(self)})
        if self.packet_integrity_sha256 != self.compute_integrity_sha256():
            raise PacketError("packet integrity sha256 mismatch")


def validate_packet_mapping(payload: Mapping[str, object]) -> None:
    keys = frozenset(payload)
    unexpected = keys - PACKET_PERMITTED_FIELDS
    missing = PACKET_PERMITTED_FIELDS - keys
    if unexpected or missing:
        raise StrictResourceViolationError(
            f"packet fields violate strict interface; unexpected={sorted(unexpected)}, missing={sorted(missing)}"
        )

    lowered_keys = {key.casefold().replace("_", " ") for key in keys}
    forbidden = sorted(FORBIDDEN_PACKET_CONTENT & lowered_keys)
    if forbidden:
        raise StrictResourceViolationError(f"forbidden packet content present: {', '.join(forbidden)}")
