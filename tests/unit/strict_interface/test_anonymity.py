from __future__ import annotations

from fedorbit.interface import (
    AnonymityCoordinate,
    AnonymityCoordinateEntry,
    anonymous_node_order,
)
from fedorbit.types import ClientRole, CoarseGroup


def _coordinate(name: str, value: str) -> AnonymityCoordinate:
    return AnonymityCoordinate((AnonymityCoordinateEntry(name, value),))


def test_anonymous_node_order_is_deterministic_bijection() -> None:
    coordinate = _coordinate("packet", "a")
    first = anonymous_node_order(
        5,
        17,
        ClientRole.SOURCE,
        CoarseGroup.DISRUPTION,
        coordinate,
    )
    second = anonymous_node_order(
        5,
        17,
        ClientRole.SOURCE,
        CoarseGroup.DISRUPTION,
        coordinate,
    )
    assert first == second
    assert tuple(sorted(first.permutation)) == tuple(range(5))
    assert first.display_ids == tuple(f"node-{index:04d}" for index in range(1, 6))


def test_source_and_target_orders_use_independent_streams() -> None:
    coordinate = _coordinate("cell", "x")
    source = anonymous_node_order(
        12,
        17,
        ClientRole.SOURCE,
        CoarseGroup.EXPLOITATION,
        coordinate,
    )
    target = anonymous_node_order(
        12,
        17,
        ClientRole.TARGET,
        CoarseGroup.EXPLOITATION,
        coordinate,
    )
    assert source.permutation != target.permutation
    assert source.display_ids == target.display_ids


def test_reorder_never_emits_semantic_names_as_ids() -> None:
    semantic_names = ("DDoS", "Ransomware", "Backdoor")
    order = anonymous_node_order(
        len(semantic_names),
        7,
        ClientRole.SOURCE,
        CoarseGroup.DISRUPTION,
        _coordinate("fixture", "reorder"),
    )
    reordered = order.reorder(semantic_names)
    assert set(reordered) == set(semantic_names)
    assert not set(order.display_ids) & set(semantic_names)
