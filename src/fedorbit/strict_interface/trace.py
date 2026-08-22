from __future__ import annotations

from dataclasses import dataclass, field

from fedorbit.domain.enums import ClientRole
from fedorbit.strict_interface.resources import (
    ResourceKind,
    StrictResourcePolicy,
    StrictResourceViolationError,
)


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
        self, role: ClientRole, resource: ResourceKind, transfer_finalized: bool = True
    ) -> None:
        if role == ClientRole.SOURCE:
            self._policy.assert_source_allowed(resource)
        elif role == ClientRole.TARGET:
            self._policy.assert_target_allowed(resource, transfer_finalized)
        else:
            raise StrictResourceViolationError(f"unknown client role: {role.value}")
        self._events.append(AccessEvent(role, resource, transfer_finalized))

    def trace(self) -> AccessTrace:
        return AccessTrace(tuple(self._events))

    def validate(self) -> None:
        for event in self._events:
            if event.role == ClientRole.SOURCE:
                self._policy.assert_source_allowed(event.resource)
            elif event.role == ClientRole.TARGET:
                self._policy.assert_target_allowed(event.resource, event.transfer_finalized)
