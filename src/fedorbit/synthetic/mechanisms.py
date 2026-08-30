from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from fedorbit.config.context import active_config
from fedorbit.runtime.seeds import RandomSeed
from fedorbit.synthetic.generators import SyntheticRandomRequest, create_float64_random_stream


class MechanismGenerationError(ValueError):
    pass


class UnresolvedMapWorldKind(StrEnum):
    COMMON_ACTION = "common_action"
    ROBUST_COMPROMISE = "robust_compromise"
    MAP_DEPENDENT = "map_dependent"


@dataclass(frozen=True, slots=True)
class UnresolvedMapWorldRequest:
    world_kind: UnresolvedMapWorldKind
    seed: RandomSeed


@dataclass(frozen=True, slots=True)
class UnresolvedMapWorld:
    block_pattern: tuple[int, int]
    lower_response_matrix: np.ndarray
    target_importance: np.ndarray
    generation_seed: RandomSeed
    world_kind: UnresolvedMapWorldKind


def generate_unresolved_map_world(request: UnresolvedMapWorldRequest) -> UnresolvedMapWorld:
    random = create_float64_random_stream(SyntheticRandomRequest(request.seed)).generator
    if request.world_kind == UnresolvedMapWorldKind.COMMON_ACTION:
        pattern = active_config().generators.common_action_unresolved_map.block_pattern
        response = _common_action_response(random, (pattern[0], pattern[1]))
    elif request.world_kind == UnresolvedMapWorldKind.ROBUST_COMPROMISE:
        configuration = active_config().generators.robust_compromise_unresolved_map
        pattern = configuration.block_pattern
        response = _independent_response(
            random, (pattern[0], pattern[1]), configuration.response_uniform
        )
    else:
        configuration = active_config().generators.map_dependent
        pattern = configuration.block_pattern
        response = _independent_response(
            random, (pattern[0], pattern[1]), configuration.response_uniform
        )
    return UnresolvedMapWorld(
        block_pattern=(pattern[0], pattern[1]),
        lower_response_matrix=response,
        target_importance=_gamma_normalized_importance(random, sum(pattern)),
        generation_seed=request.seed,
        world_kind=request.world_kind,
    )


def _common_action_response(
    random: np.random.Generator,
    pattern: tuple[int, int],
) -> np.ndarray:
    lower, upper = (
        active_config().generators.common_action_unresolved_map.block_pair_response_uniform
    )
    block_values = random.uniform(lower, upper, size=(2, 2)).astype(np.float64)
    first, second = pattern
    response = np.empty((first + second, first + second), dtype=np.float64)
    response[:first, :first] = block_values[0, 0]
    response[:first, first:] = block_values[0, 1]
    response[first:, :first] = block_values[1, 0]
    response[first:, first:] = block_values[1, 1]
    return response


def _independent_response(
    random: np.random.Generator,
    pattern: tuple[int, int],
    bounds: tuple[float, float],
) -> np.ndarray:
    size = sum(pattern)
    return random.uniform(bounds[0], bounds[1], size=(size, size)).astype(np.float64)


def _gamma_normalized_importance(random: np.random.Generator, size: int) -> np.ndarray:
    gamma = active_config().generators.exact_separator_theorem.target_importance_gamma
    weights = random.gamma(gamma.shape, gamma.scale, size=size).astype(np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise MechanismGenerationError("gamma target-importance draw has nonpositive total")
    return weights / total
