from __future__ import annotations

import math

import pytest
import torch

from fedorbit.response.estimation import (
    ResponseEstimationError,
    equal_native_class_risk,
    native_class_cross_entropy,
    paired_shadow_derivative,
    shadow_batch_schedule,
)


def test_paired_shadow_derivative_formula() -> None:
    value = paired_shadow_derivative(0.9, 1.1, 1.0, 0.1, 1e-8)
    assert value == pytest.approx((1.1 - 0.9) / (2.0 * 0.1))


def test_paired_shadow_derivative_rejects_invalid_scales() -> None:
    with pytest.raises(ResponseEstimationError):
        paired_shadow_derivative(0.9, 1.1, 1.0, 0.0, 1e-8)
    with pytest.raises(ResponseEstimationError):
        paired_shadow_derivative(0.9, 1.1, 1.0, 0.1, 0.0)


def test_equal_native_class_risk_is_equal_class_average() -> None:
    logits = torch.tensor([[3.0, 1.0], [1.0, 3.0], [0.5, 2.5]])
    targets = torch.tensor([0, 1, 0])
    class_zero = native_class_cross_entropy(logits, targets, 0, 1e-12)
    class_one = native_class_cross_entropy(logits, targets, 1, 1e-12)
    assert class_zero == pytest.approx((0.126928 + 2.126928) / 2.0, abs=1e-4)
    assert class_one == pytest.approx(0.126928, abs=1e-4)
    assert equal_native_class_risk(logits, targets, (0, 1), 1e-12) == pytest.approx(
        (class_zero + class_one) / 2.0,
        abs=1e-4,
    )


def test_missing_native_class_is_invalid_risk() -> None:
    logits = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    targets = torch.tensor([0, 0])
    assert math.isnan(equal_native_class_risk(logits, targets, (0, 1), 1e-12))


def test_shadow_batch_schedule_retains_partial_batch_and_replays() -> None:
    first_rng = torch.Generator().manual_seed(7)
    second_rng = torch.Generator().manual_seed(7)
    first = shadow_batch_schedule(10, 4, first_rng)
    second = shadow_batch_schedule(10, 4, second_rng)
    first_pass = tuple(next(first) for _ in range(3))
    second_pass = tuple(next(second) for _ in range(3))
    assert [int(batch.shape[0]) for batch in first_pass] == [4, 4, 2]
    assert tuple(sorted(int(value) for batch in first_pass for value in batch)) == tuple(range(10))
    assert all(torch.equal(left, right) for left, right in zip(first_pass, second_pass, strict=True))


def test_shadow_batch_schedule_rejects_invalid_sizes() -> None:
    with pytest.raises(ResponseEstimationError):
        next(shadow_batch_schedule(0, 4, torch.Generator()))
    with pytest.raises(ResponseEstimationError):
        next(shadow_batch_schedule(10, 0, torch.Generator()))
