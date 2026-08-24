from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fedorbit.domain.enums import ClientRole
from fedorbit.strict_interface.resources import (
    ResourceKind,
    StrictResourcePolicy,
    StrictResourceViolationError,
)

_ANONYMOUS_NODE_PATTERN = re.compile(r"^node-[0-9]{4,}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class AccessEvent:
    role: ClientRole
    resource: ResourceKind
    transfer_finalized: bool


@dataclass(frozen=True, slots=True)
class AccessTrace:
    events: tuple[AccessEvent, ...] = field(default_factory=tuple)

    def resources(self) -> tuple[ResourceKind, ...]:
        return tuple(event.resource for event in self.events)


class AccessLogger:
    def __init__(self, policy: StrictResourcePolicy | None = None) -> None:
        self._policy = policy if policy is not None else StrictResourcePolicy()
        self._events: list[AccessEvent] = []

    def record(
        self,
        role: ClientRole,
        resource: ResourceKind,
        transfer_finalized: bool = False,
    ) -> None:
        self._policy.assert_role_allowed(role, resource, transfer_finalized=transfer_finalized)
        self._events.append(AccessEvent(role, resource, transfer_finalized))

    def trace(self) -> AccessTrace:
        return AccessTrace(tuple(self._events))

    def validate(self) -> None:
        for event in self._events:
            self._policy.assert_role_allowed(
                event.role,
                event.resource,
                transfer_finalized=event.transfer_finalized,
            )


def validate_exact_fields(
    payload: Mapping[str, object],
    permitted_fields: frozenset[str],
) -> None:
    keys = frozenset(payload)
    unexpected = keys - permitted_fields
    missing = permitted_fields - keys
    if unexpected or missing:
        raise StrictResourceViolationError(
            f"strict interface fields differ from contract; unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )


def validate_anonymous_node_ids(node_ids: tuple[str, ...]) -> None:
    if not node_ids:
        raise StrictResourceViolationError("anonymous node identifier list is empty")
    expected = tuple(f"node-{index:04d}" for index in range(1, len(node_ids) + 1))
    if node_ids != expected:
        raise StrictResourceViolationError(
            "anonymous node identifiers are not canonical sequential IDs"
        )
    if any(_ANONYMOUS_NODE_PATTERN.fullmatch(node_id) is None for node_id in node_ids):
        raise StrictResourceViolationError("invalid anonymous node identifier")


def validate_sha256(value: str, field_name: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise StrictResourceViolationError(f"{field_name} is not a lowercase SHA-256 digest")


def validate_rfc3339_utc(value: str) -> None:
    if not value.endswith("Z"):
        raise StrictResourceViolationError("technical creation timestamp must be RFC 3339 UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise StrictResourceViolationError(
            "technical creation timestamp must be RFC 3339 UTC"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise StrictResourceViolationError("technical creation timestamp must be UTC")
