from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.runtime import RandomSeed
from fedorbit.types import ScalabilityBlockPattern


@dataclass(frozen=True, slots=True)
class SyntheticRandomRequest:
    seed: RandomSeed


@dataclass(frozen=True, slots=True)
class SyntheticRandomStream:
    generator: np.random.Generator


def create_float64_random_stream(request: SyntheticRandomRequest) -> SyntheticRandomStream:
    return SyntheticRandomStream(np.random.Generator(np.random.PCG64(request.seed.value)))


class SyntheticGenerationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExactSeparatorInstanceRequest:
    block_pattern: tuple[int, ...]
    seed: RandomSeed

    def __post_init__(self) -> None:
        if not self.block_pattern or any(size < 1 for size in self.block_pattern):
            raise SyntheticGenerationError("synthetic block pattern must contain positive blocks")


@dataclass(frozen=True, slots=True)
class ExactSeparatorInstance:
    block_pattern: tuple[int, ...]
    lower_response_matrix: np.ndarray
    upper_response_matrix: np.ndarray
    target_importance: np.ndarray
    active_action: np.ndarray
    generation_seed: RandomSeed


def generate_exact_separator_instance(
    request: ExactSeparatorInstanceRequest,
) -> ExactSeparatorInstance:
    generator_config = active_config().generators.exact_separator_theorem
    random = create_float64_random_stream(SyntheticRandomRequest(request.seed)).generator
    size = sum(request.block_pattern)
    response_lower, response_upper = generator_config.response_uniform
    lower = random.uniform(response_lower, response_upper, size=(size, size)).astype(np.float64)
    increment_lower, increment_upper = generator_config.serialization_upper_band_increment_uniform
    upper = lower + random.uniform(increment_lower, increment_upper, size=(size, size))
    gamma = generator_config.target_importance_gamma
    importance = random.gamma(gamma.shape, gamma.scale, size=size).astype(np.float64)
    action_lower, action_upper = generator_config.active_action_uniform
    active_action = random.uniform(action_lower, action_upper, size=size).astype(np.float64)
    return ExactSeparatorInstance(
        block_pattern=request.block_pattern,
        lower_response_matrix=lower,
        upper_response_matrix=upper,
        target_importance=importance,
        active_action=active_action,
        generation_seed=request.seed,
    )


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


class ScalabilityGenerationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ScalabilityInstanceRequest:
    node_count: int
    block_pattern: ScalabilityBlockPattern
    support_size: int
    seed: RandomSeed

    def __post_init__(self) -> None:
        if self.node_count < 2:
            raise ScalabilityGenerationError("scalability node count must be at least two")
        if self.support_size < 1:
            raise ScalabilityGenerationError("scalability support size must be positive")


@dataclass(frozen=True, slots=True)
class ScalabilityInstance:
    block_pattern: tuple[int, int]
    lower_response_matrix: np.ndarray
    target_importance: np.ndarray
    fixed_action: np.ndarray
    generation_seed: RandomSeed


def generate_scalability_instance(request: ScalabilityInstanceRequest) -> ScalabilityInstance:
    block_pattern = _block_pattern(request)
    largest_block_size = max(block_pattern)
    if request.support_size > largest_block_size:
        raise ScalabilityGenerationError("support size exceeds the largest coarse block")
    lower, upper = active_config().generators.scalability.response_uniform
    random = create_float64_random_stream(SyntheticRandomRequest(request.seed)).generator
    response = random.uniform(lower, upper, size=(request.node_count, request.node_count)).astype(
        np.float64
    )
    target_importance = np.full(request.node_count, 1.0 / request.node_count, dtype=np.float64)
    fixed_action = np.zeros(request.node_count, dtype=np.float64)
    action = active_config().scientific.action
    action_value = min(action.coordinate_cap, action.total_curriculum_budget / request.support_size)
    fixed_action[: request.support_size] = action_value
    return ScalabilityInstance(
        block_pattern=block_pattern,
        lower_response_matrix=response,
        target_importance=target_importance,
        fixed_action=fixed_action,
        generation_seed=request.seed,
    )


def _block_pattern(request: ScalabilityInstanceRequest) -> tuple[int, int]:
    if request.block_pattern == ScalabilityBlockPattern.BALANCED:
        lower = request.node_count // 2
        return (lower, request.node_count - lower)
    return (request.node_count - 1, 1)
