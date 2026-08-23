from __future__ import annotations

import math

import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.response.pilot import (
    CandidateResult,
    ResponseCandidate,
    ResponsePilotError,
    select_response_configuration,
    sign_agreement,
    standard_error,
)


def _result(
    candidate: ResponseCandidate,
    score: float,
    eligible: bool = True,
) -> CandidateResult:
    return CandidateResult(candidate, (), eligible, (), score)


def test_selection_prefers_score_then_smaller_horizon_then_magnitude() -> None:
    results = (
        _result(ResponseCandidate(0.2, 100), 5.0),
        _result(ResponseCandidate(0.1, 50), 7.0),
        _result(ResponseCandidate(0.05, 25), 7.0),
        _result(ResponseCandidate(0.05, 50), 7.0),
    )
    assert select_response_configuration(results) == ResponseCandidate(0.05, 25)


def test_selection_raises_without_eligible_candidate() -> None:
    with pytest.raises(ResponsePilotError):
        select_response_configuration((_result(ResponseCandidate(0.1, 25), 3.0, False),))


def test_sign_agreement_treats_zero_as_disagreement() -> None:
    assert sign_agreement((1.0, 1.0, -1.0, -1.0, -1.0)) == pytest.approx(0.6)
    assert sign_agreement((0.0, 1.0, 1.0, -1.0)) == pytest.approx(0.5)
    assert sign_agreement((0.0, 0.0, 0.0)) == 0.0


def test_standard_error_uses_ddof_one() -> None:
    values = (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
    assert standard_error(values) == pytest.approx(math.sqrt(32.0 / 7.0) / math.sqrt(8.0))
    assert math.isnan(standard_error((1.0,)))


def test_pilot_grid_is_configuration_owned() -> None:
    config = load_fedorbit_config()
    pilot = config.scientific.source_response_pilot
    assert tuple(pilot.intervention_magnitudes) == (0.05, 0.10, 0.20)
    assert tuple(pilot.optimizer_step_horizons) == (25, 50, 100)
    assert pilot.paired_schedules_per_candidate == 8
