from __future__ import annotations

import numpy as np

from fedorbit.config.models import FedorbitConfig
from fedorbit.synthetic.generators import (
    ExactSeparatorInstanceRequest,
    generate_exact_separator_instance,
)


def test_exact_separator_generation_is_deterministic(fedorbit_config: FedorbitConfig) -> None:
    request = ExactSeparatorInstanceRequest(fedorbit_config, (2, 2), 101)
    first = generate_exact_separator_instance(request)
    second = generate_exact_separator_instance(request)

    assert np.array_equal(first.lower_response_matrix, second.lower_response_matrix)
    assert np.array_equal(first.upper_response_matrix, second.upper_response_matrix)
    assert np.array_equal(first.target_importance, second.target_importance)


def test_exact_separator_generation_respects_response_bands(
    fedorbit_config: FedorbitConfig,
) -> None:
    instance = generate_exact_separator_instance(
        ExactSeparatorInstanceRequest(fedorbit_config, (2, 3), 202)
    )
    lower, upper = fedorbit_config.generators.exact_separator_theorem.response_uniform

    assert np.all(instance.lower_response_matrix >= lower)
    assert np.all(instance.lower_response_matrix <= upper)
    assert np.all(instance.upper_response_matrix >= instance.lower_response_matrix)
    assert instance.active_action.shape == (sum(instance.block_pattern),)
