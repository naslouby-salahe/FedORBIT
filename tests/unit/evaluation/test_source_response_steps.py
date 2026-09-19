from __future__ import annotations

import pytest

from fedorbit.response.estimation import (
    SHADOW_DIRECTIONS_PER_PAIR,
    ResponseEstimationError,
    source_response_optimizer_steps,
)


def test_source_response_optimizer_steps_match_shadow_pair_loops() -> None:
    horizon = 25
    interventions = 3
    replicates = 8
    assert SHADOW_DIRECTIONS_PER_PAIR == 2
    assert source_response_optimizer_steps(horizon, interventions, replicates) == (
        horizon * 2 * interventions * replicates
    )


def test_source_response_optimizer_steps_reject_invalid_counts() -> None:
    with pytest.raises(ResponseEstimationError):
        source_response_optimizer_steps(0, 1, 2)
    with pytest.raises(ResponseEstimationError):
        source_response_optimizer_steps(25, 0, 2)
    with pytest.raises(ResponseEstimationError):
        source_response_optimizer_steps(25, 1, 1)
