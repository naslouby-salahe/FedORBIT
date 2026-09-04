from __future__ import annotations

import numpy as np

from fedorbit.config.loading import active_config
from fedorbit.experiments.synthetic import (
    ScalabilityInstanceRequest,
    generate_scalability_instance,
)
from fedorbit.types import ScalabilityBlockPattern


def test_balanced_scalability_instance_uses_roadmap_support_and_weights() -> None:
    instance = generate_scalability_instance(
        ScalabilityInstanceRequest(7, ScalabilityBlockPattern.BALANCED, 2, 101)
    )
    action = active_config().scientific.action
    assert instance.block_pattern == (3, 4)
    assert instance.lower_response_matrix.shape == (7, 7)
    assert np.all(instance.target_importance == 1.0 / 7.0)
    assert np.all(
        instance.fixed_action[:2] == min(action.coordinate_cap, action.total_curriculum_budget / 2)
    )
    assert np.all(instance.fixed_action[2:] == 0.0)


def test_scalability_generation_is_seed_deterministic() -> None:
    request = ScalabilityInstanceRequest(6, ScalabilityBlockPattern.MAXIMALLY_SKEWED, 3, 202)
    first = generate_scalability_instance(request)
    second = generate_scalability_instance(request)
    assert first.block_pattern == (5, 1)
    assert np.array_equal(first.lower_response_matrix, second.lower_response_matrix)
