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
from fedorbit.response.risk import equal_native_class_risk, native_class_cross_entropy
from fedorbit.response.shadows import (
    ShadowError,
    paired_shadow_derivative,
    shadow_batch_schedule,
)


def test_paired_shadow_derivative_formula() -> None:
    value = paired_shadow_derivative(
        positive_risk=0.9, negative_risk=1.1, baseline_risk=1.0, epsilon=0.1, denominator_floor=1e-8
    )
    expected = (1.1 - 0.9) / (2 * 0.1 * max(1.0, 1e-8))
    assert value == pytest.approx(expected)


def test_paired_shadow_derivative_uses_baseline_floor() -> None:
    value = paired_shadow_derivative(
        positive_risk=0.5,
        negative_risk=0.6,
        baseline_risk=0.0,
        epsilon=0.05,
        denominator_floor=1e-8,
    )
    assert value == pytest.approx((0.6 - 0.5) / (2 * 0.05 * 1e-8))


def test_paired_shadow_derivative_rejects_nonpositive_epsilon() -> None:
    with pytest.raises(ShadowError):
        paired_shadow_derivative(0.9, 1.1, 1.0, 0.0, 1e-8)


def test_equal_native_class_risk_hand_solvable() -> None:
    import torch

    logits = torch.tensor([[3.0, 1.0], [1.0, 3.0], [0.5, 2.5]])
    targets = torch.tensor([0, 1, 0])
    floor = 1e-12
    class_zero = native_class_cross_entropy(logits, targets, 0, floor)
    assert class_zero == pytest.approx((0.126928 + 2.126928) / 2, abs=1e-4)
    class_one = native_class_cross_entropy(logits, targets, 1, floor)
    assert class_one == pytest.approx(0.126928, abs=1e-4)
    risk = equal_native_class_risk(logits, targets, (0, 1), floor)
    assert risk == pytest.approx((class_zero + class_one) / 2, abs=1e-4)


def test_equal_native_class_risk_empty_class_is_nan() -> None:
    import torch

    logits = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    targets = torch.tensor([0, 0])
    assert math.isnan(native_class_cross_entropy(logits, targets, 1, 1e-12))
    assert math.isnan(equal_native_class_risk(logits, targets, (0, 1), 1e-12))


def test_shadow_schedule_is_infinite_and_retains_partial_batch() -> None:
    import torch

    rng = torch.Generator().manual_seed(7)
    schedule = shadow_batch_schedule(10, 4, rng)
    first_pass = tuple(next(schedule) for _ in range(3))
    assert [int(batch.shape[0]) for batch in first_pass] == [4, 4, 2]
    assert tuple(sorted(int(value) for batch in first_pass for value in batch)) == tuple(range(10))


def test_shadow_schedule_continues_passes() -> None:
    import torch

    rng = torch.Generator().manual_seed(7)
    schedule = shadow_batch_schedule(10, 4, rng)
    for _ in range(3):
        next(schedule)
    second = next(schedule)
    assert int(second.shape[0]) == 4
    assert {int(value) for value in second} <= set(range(10))


def test_shadow_schedule_is_deterministic() -> None:
    import torch

    first = tuple(
        int(value)
        for _ in range(5)
        for value in next(shadow_batch_schedule(10, 4, torch.Generator().manual_seed(7)))
    )
    second = tuple(
        int(value)
        for _ in range(5)
        for value in next(shadow_batch_schedule(10, 4, torch.Generator().manual_seed(7)))
    )
    assert first == second


def test_shadow_schedule_rejects_empty_train() -> None:
    import torch

    schedule = shadow_batch_schedule(0, 4, torch.Generator())
    with pytest.raises(ShadowError):
        next(schedule)


def test_sign_agreement_larger_fraction_and_zero_disagreement() -> None:
    assert sign_agreement((1.0, 1.0, -1.0, -1.0, -1.0)) == pytest.approx(0.6)
    assert sign_agreement((0.0, 1.0, 1.0, -1.0)) == pytest.approx(0.5)
    assert sign_agreement((0.0, 0.0, 0.0)) == pytest.approx(0.0)


def test_standard_error_ddof1() -> None:
    assert standard_error((2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)) == pytest.approx(
        math.sqrt(32 / 7) / math.sqrt(8)
    )
    assert math.isnan(standard_error((1.0,)))


def test_selection_prefers_higher_score_then_smaller_horizon_then_smaller_magnitude() -> None:
    results = (
        _result(ResponseCandidate(0.2, 100), 5.0),
        _result(ResponseCandidate(0.1, 50), 7.0),
        _result(ResponseCandidate(0.05, 25), 7.0),
        _result(ResponseCandidate(0.05, 50), 7.0),
    )
    assert select_response_configuration(results) == ResponseCandidate(0.05, 25)


def test_selection_raises_without_eligible_candidates() -> None:
    results = (_result(ResponseCandidate(0.1, 25), 3.0, eligible=False),)
    with pytest.raises(ResponsePilotError):
        select_response_configuration(results)


def test_selection_orders_by_higher_q() -> None:
    results = (
        _result(ResponseCandidate(0.1, 25), 4.0),
        _result(ResponseCandidate(0.2, 25), 9.0),
    )
    assert select_response_configuration(results) == ResponseCandidate(0.2, 25)


def _result(
    candidate: ResponseCandidate,
    score: float,
    eligible: bool = True,
) -> CandidateResult:
    return CandidateResult(
        candidate=candidate,
        entries=(),
        eligible=eligible,
        ineligibility_reasons=(),
        pilot_score=score,
    )


def test_pilot_grid_uses_registered_magnitudes_and_horizons() -> None:
    config = load_fedorbit_config()
    assert tuple(config.scientific.source_response_pilot.intervention_magnitudes) == (
        0.05,
        0.10,
        0.20,
    )
    assert tuple(config.scientific.source_response_pilot.optimizer_step_horizons) == (25, 50, 100)
    assert config.scientific.source_response_pilot.paired_schedules_per_candidate == 8
