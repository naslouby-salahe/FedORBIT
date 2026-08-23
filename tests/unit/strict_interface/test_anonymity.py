from __future__ import annotations

from fedorbit.domain.enums import ClientRole, CoarseGroup
from fedorbit.strict_interface.anonymity import anonymous_node_order


def test_anonymous_node_order_is_deterministic_bijection() -> None:
    first = anonymous_node_order(
        5,
        17,
        ClientRole.SOURCE,
        CoarseGroup.DISRUPTION,
        {"packet": "a"},
    )
    second = anonymous_node_order(
        5,
        17,
        ClientRole.SOURCE,
        CoarseGroup.DISRUPTION,
        {"packet": "a"},
    )
    assert first == second
    assert tuple(sorted(first.permutation)) == tuple(range(5))
    assert first.display_ids == tuple(f"node-{index:04d}" for index in range(1, 6))


def test_source_and_target_orders_use_independent_streams() -> None:
    source = anonymous_node_order(
        12,
        17,
        ClientRole.SOURCE,
        CoarseGroup.EXPLOITATION,
        {"cell": "x"},
    )
    target = anonymous_node_order(
        12,
        17,
        ClientRole.TARGET,
        CoarseGroup.EXPLOITATION,
        {"cell": "x"},
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
        "fixture",
    )
    reordered = order.reorder(semantic_names)
    assert set(reordered) == set(semantic_names)
    assert not set(order.display_ids) & set(semantic_names)
