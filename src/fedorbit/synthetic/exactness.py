from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedorbit.config.context import active_config
from fedorbit.synthetic.generators import SyntheticRandomRequest, create_float64_random_stream


class SyntheticGenerationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExactSeparatorInstanceRequest:
    block_pattern: tuple[int, ...]
    seed: int

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
    generation_seed: int


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
