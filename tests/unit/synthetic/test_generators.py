from __future__ import annotations

import numpy as np

from fedorbit.config.models import FedorbitConfig
from fedorbit.experiments.synthetic import (
    ExactSeparatorInstanceRequest,
    generate_exact_separator_instance,
)


def test_exact_separator_generation_is_deterministic() -> None:
    request = ExactSeparatorInstanceRequest((2, 2), 101)
    first = generate_exact_separator_instance(request)
    second = generate_exact_separator_instance(request)

    assert np.array_equal(first.lower_response_matrix, second.lower_response_matrix)
    assert np.array_equal(first.upper_response_matrix, second.upper_response_matrix)
    assert np.array_equal(first.target_importance, second.target_importance)


def test_exact_separator_generation_respects_response_bands(
    fedorbit_config: FedorbitConfig,
) -> None:
    instance = generate_exact_separator_instance(ExactSeparatorInstanceRequest((2, 3), 202))
    lower, upper = fedorbit_config.generators.exact_separator_theorem.response_uniform

    assert np.all(instance.lower_response_matrix >= lower)
    assert np.all(instance.lower_response_matrix <= upper)
    assert np.all(instance.upper_response_matrix >= instance.lower_response_matrix)
    assert instance.active_action.shape == (sum(instance.block_pattern),)


def test_exact_separator_action_is_restricted_to_the_requested_support(
    fedorbit_config: FedorbitConfig,
) -> None:
    request = ExactSeparatorInstanceRequest((3, 3), 303, active_support_size=2)
    instance = generate_exact_separator_instance(request)

    nonzero = np.flatnonzero(instance.active_action)
    assert nonzero.shape == (2,)
    budget = fedorbit_config.scientific.action.total_curriculum_budget
    assert float(instance.active_action.sum()) <= budget + 1e-12


def test_exact_separator_action_support_size_one_matches_a_single_draw(
    fedorbit_config: FedorbitConfig,
) -> None:
    instance = generate_exact_separator_instance(
        ExactSeparatorInstanceRequest((2, 2), 101, active_support_size=1)
    )
    nonzero = np.flatnonzero(instance.active_action)
    assert nonzero.shape == (1,)
    lower, upper = fedorbit_config.generators.exact_separator_theorem.active_action_uniform
    assert lower <= float(instance.active_action[nonzero[0]]) <= upper
