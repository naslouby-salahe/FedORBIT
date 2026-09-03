from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import cast

import torch

from fedorbit.infrastructure.runtime import RandomSeed, SeedDerivationRequest, derive_seed32
from fedorbit.types import ClientRole, CoarseGroup, RngNamespace, StableJsonPayload


class AnonymityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AnonymityCoordinateEntry:
    name: str
    value: str

    def __post_init__(self) -> None:
        if not self.name or not self.value:
            raise AnonymityError("anonymity coordinate entries must be non-empty")


@dataclass(frozen=True, slots=True)
class AnonymityCoordinate:
    entries: tuple[AnonymityCoordinateEntry, ...]

    def __post_init__(self) -> None:
        names = tuple(entry.name for entry in self.entries)
        if not names or len(set(names)) != len(names):
            raise AnonymityError("anonymity coordinates must have unique non-empty names")


@dataclass(frozen=True, slots=True)
class AnonymousNodeOrder:
    permutation: tuple[int, ...]
    display_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = tuple(range(len(self.permutation)))
        if tuple(sorted(self.permutation)) != expected:
            raise AnonymityError("anonymous node permutation is not a bijection")
        expected_ids = tuple(f"node-{index:04d}" for index in range(1, len(self.permutation) + 1))
        if self.display_ids != expected_ids:
            raise AnonymityError("anonymous node identifiers are not stable sequential IDs")

    def reorder[T](self, values: tuple[T, ...]) -> tuple[T, ...]:
        if len(values) != len(self.permutation):
            raise AnonymityError("value count does not match anonymous node count")
        return tuple(values[index] for index in self.permutation)


def anonymous_node_order(
    node_count: int,
    base_seed: int,
    endpoint: ClientRole,
    coarse_group: CoarseGroup,
    coordinate: AnonymityCoordinate,
) -> AnonymousNodeOrder:
    if node_count <= 0:
        raise AnonymityError("anonymous node order requires at least one node")
    if endpoint not in (ClientRole.SOURCE, ClientRole.TARGET):
        raise AnonymityError("anonymous ordering endpoint must be source or target")
    seed = derive_seed32(
        SeedDerivationRequest(
            RandomSeed(base_seed),
            RngNamespace.ANONYMOUS_NODE_ORDER,
            cast(
                StableJsonPayload,
                OrderedDict(
                    endpoint=endpoint.value,
                    coarse_group=coarse_group.value,
                    coordinate=coordinate,
                ),
            ),
        )
    ).value
    generator = torch.Generator().manual_seed(seed)
    permutation = tuple(int(index) for index in torch.randperm(node_count, generator=generator))
    display_ids = tuple(f"node-{index:04d}" for index in range(1, node_count + 1))
    return AnonymousNodeOrder(permutation, display_ids)


class ResourceKind(StrEnum):
    TRAIN = "TRAIN"
    META = "META"
    VALID = "VALID"
    CONFIRM = "CONFIRM"
    TEST = "TEST"
    LOCAL_FINE_LABELS = "local fine labels"
    LOCAL_COARSE_GROUP_MEMBERSHIP = "local coarse-group membership"
    LOCAL_PREPROCESSING = "local preprocessing"
    LOCAL_CLASSIFIER = "local classifier"
    LOCAL_OPTIMIZER_CHECKPOINT = "local optimizer/checkpoint"
    LOCAL_INTERVENTION_EXPERIMENTS = "local intervention experiments"
    SHARED_COARSE_GROUPS = "shared coarse groups"
    ANONYMOUS_SOURCE_PACKET = "anonymous source packet"
    LOCAL_MODEL_STATE = "local model state"
    LOCAL_IMPORTANCE_VECTOR = "local importance vector"


SOURCE_LOCAL_WHITELIST = frozenset(
    {
        ResourceKind.TRAIN,
        ResourceKind.META,
        ResourceKind.VALID,
        ResourceKind.LOCAL_FINE_LABELS,
        ResourceKind.LOCAL_COARSE_GROUP_MEMBERSHIP,
        ResourceKind.LOCAL_PREPROCESSING,
        ResourceKind.LOCAL_CLASSIFIER,
        ResourceKind.LOCAL_OPTIMIZER_CHECKPOINT,
        ResourceKind.LOCAL_INTERVENTION_EXPERIMENTS,
    }
)

TARGET_LOCAL_WHITELIST = frozenset(
    {
        ResourceKind.TRAIN,
        ResourceKind.META,
        ResourceKind.VALID,
        ResourceKind.CONFIRM,
        ResourceKind.LOCAL_FINE_LABELS,
        ResourceKind.SHARED_COARSE_GROUPS,
        ResourceKind.ANONYMOUS_SOURCE_PACKET,
        ResourceKind.LOCAL_MODEL_STATE,
        ResourceKind.LOCAL_IMPORTANCE_VECTOR,
    }
)


class StrictResourceViolationError(PermissionError):
    pass


class StrictResourcePolicy:
    def assert_source_allowed(self, resource: ResourceKind) -> None:
        if resource not in SOURCE_LOCAL_WHITELIST:
            raise StrictResourceViolationError(
                f"source-local resource outside the whitelist: {resource.value}"
            )

    def assert_target_allowed(self, resource: ResourceKind, transfer_finalized: bool) -> None:
        if resource == ResourceKind.TEST:
            if not transfer_finalized:
                raise StrictResourceViolationError(
                    "TEST is permitted only after the transfer decision is fully finalized"
                )
            return
        if resource not in TARGET_LOCAL_WHITELIST:
            raise StrictResourceViolationError(
                f"target-local resource outside the whitelist: {resource.value}"
            )

    def assert_role_allowed(
        self,
        role: ClientRole,
        resource: ResourceKind,
        transfer_finalized: bool = False,
    ) -> None:
        if role == ClientRole.SOURCE:
            self.assert_source_allowed(resource)
            return
        if role == ClientRole.TARGET:
            self.assert_target_allowed(resource, transfer_finalized)
            return
        raise StrictResourceViolationError(f"unknown client role: {role.value}")


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
    field_names: frozenset[str],
    permitted_fields: frozenset[str],
) -> None:
    unexpected = field_names - permitted_fields
    missing = permitted_fields - field_names
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
            "anonymous node identifiers are not stable sequential IDs"
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
