from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from fedorbit.config.context import active_config
from fedorbit.runtime.seeds import RandomSeed
from fedorbit.synthetic.generators import SyntheticRandomRequest, create_float64_random_stream


class ScalabilityGenerationError(ValueError):
    pass


class ScalabilityBlockPattern(StrEnum):
    BALANCED = "balanced"
    MAXIMALLY_SKEWED_TWO_BLOCK = "maximally_skewed_two-block"


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
