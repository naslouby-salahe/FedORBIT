from __future__ import annotations

import numpy as np

from fedorbit.runtime.seeds import RandomSeed
from fedorbit.synthetic.mechanisms import (
    UnresolvedMapWorldKind,
    UnresolvedMapWorldRequest,
    generate_unresolved_map_world,
)


def test_common_action_world_is_block_constant_and_weight_normalized() -> None:
    world = generate_unresolved_map_world(
        UnresolvedMapWorldRequest(UnresolvedMapWorldKind.COMMON_ACTION, RandomSeed(101))
    )
    response = world.lower_response_matrix
    assert world.block_pattern == (2, 2)
    assert np.all(response[:2, :2] == response[0, 0])
    assert np.all(response[:2, 2:] == response[0, 2])
    assert np.all(response[2:, :2] == response[2, 0])
    assert np.all(response[2:, 2:] == response[2, 2])
    assert np.isclose(world.target_importance.sum(), 1.0)


def test_unresolved_map_worlds_are_seed_deterministic() -> None:
    request = UnresolvedMapWorldRequest(UnresolvedMapWorldKind.MAP_DEPENDENT, RandomSeed(202))
    first = generate_unresolved_map_world(request)
    second = generate_unresolved_map_world(request)
    assert np.array_equal(first.lower_response_matrix, second.lower_response_matrix)
    assert np.array_equal(first.target_importance, second.target_importance)
