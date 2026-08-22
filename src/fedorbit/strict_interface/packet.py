from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

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
    preprocessing_state_sha256: str = ""
    transfer_node_manifest_sha256: str = ""
    response_seed: int = 0
    technical_creation_timestamp: str = ""
    forbidden_content: tuple[str, ...] = field(default_factory=tuple)

    def integrity_payload(self) -> str:
        values: dict[
            str,
            str | int | float | tuple[float, ...] | list[str] | list[float] | list[int] | None,
        ] = {
            "anonymous_fine_node_ids": list(self.anonymous_fine_node_ids),
            "exposed_coarse_group_id": self.exposed_coarse_group_id,
            "L": self.L,
            "U": self.U,
            "per_node_train_support": list(self.per_node_train_support),
            "per_node_meta_support": list(self.per_node_meta_support),
            "per_node_effective_replicate_count": list(self.per_node_effective_replicate_count),
            "packet_schema_metadata": self.packet_schema_metadata,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "response_configuration_sha256": self.response_configuration_sha256,
            "packet_validity_state": self.packet_validity_state,
            "preprocessing_state_sha256": self.preprocessing_state_sha256,
            "transfer_node_manifest_sha256": self.transfer_node_manifest_sha256,
            "response_seed": self.response_seed,
        }
        return canonical_json(values)

    def compute_integrity_sha256(self) -> str:
        payload = self.integrity_payload()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        for field_name in PACKET_PERMITTED_FIELDS:
            if not hasattr(self, field_name):
                raise PacketError(f"missing permitted packet field: {field_name}")
        if self.packet_integrity_sha256 != self.compute_integrity_sha256():
            raise PacketError("packet integrity sha256 mismatch")
        if self.forbidden_content:
            raise StrictResourceViolationError(
                f"forbidden packet content present: {', '.join(self.forbidden_content)}"
            )
