from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from fedorbit.analysis.statistics import (
    InsufficientDevicesError,
    average_ranks,
    cluster_bootstrap_interval,
    exact_sign_test,
    exact_signed_rank_test,
    holm_adjust,
    spearman,
)
from fedorbit.detection.metrics import (
    UndefinedMetricError,
    auroc,
    operating_point,
    threshold_at_rate,
    worst_case_auroc_standard_error,
)
from fedorbit.types import ContrastDirection, RandomSeed


def brute_force_auroc(benign: list[float], attack: list[float]) -> float:
    wins = 0.0
    for benign_score, attack_score in product(benign, attack):
        wins += 1.0 if attack_score > benign_score else 0.5 if attack_score == benign_score else 0.0
    return wins / (len(benign) * len(attack))


def test_auroc_matches_pairwise_definition_with_ties() -> None:
    benign = [0.1, 0.4, 0.4, 0.9, 0.2]
    attack = [0.4, 0.8, 0.95, 0.1]
    assert auroc(np.array(benign), np.array(attack)) == pytest.approx(
        brute_force_auroc(benign, attack)
    )


def test_auroc_extremes() -> None:
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert auroc(np.array([1.0]), np.array([1.0])) == 0.5


def test_auroc_rejects_empty_classes() -> None:
    with pytest.raises(UndefinedMetricError):
        auroc(np.array([]), np.array([1.0]))


def test_operating_point_and_threshold() -> None:
    scores = np.arange(100, dtype=np.float64)
    threshold = threshold_at_rate(scores, 0.05)
    assert threshold == pytest.approx(np.quantile(scores, 0.95))
    metrics = operating_point(50.0, np.array([10.0, 60.0, 70.0]), np.array([40.0, 90.0]))
    assert metrics.false_positive_rate == pytest.approx(2.0 / 3.0)
    assert metrics.true_positive_rate == pytest.approx(0.5)


def test_exact_sign_test_known_values() -> None:
    all_positive = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert exact_sign_test(all_positive, ContrastDirection.GREATER) == pytest.approx(1 / 32)
    assert exact_sign_test(all_positive, ContrastDirection.TWO_SIDED) == pytest.approx(2 / 32)
    assert exact_sign_test(all_positive, ContrastDirection.LESS) == pytest.approx(1.0)
    assert exact_sign_test(np.zeros(4), ContrastDirection.GREATER) is None


def test_signed_rank_matches_full_enumeration() -> None:
    deltas = np.array([0.3, -0.1, 0.5, 0.2, -0.4, 0.05])
    ranks = average_ranks(np.abs(deltas))
    observed = float(ranks[deltas > 0].sum())
    null = [
        float(np.dot(np.array(signs), ranks)) for signs in product((0.0, 1.0), repeat=len(deltas))
    ]
    expected = float(np.mean([value >= observed - 1e-12 for value in null]))
    assert exact_signed_rank_test(deltas, ContrastDirection.GREATER) == pytest.approx(expected)


def test_signed_rank_smallest_attainable_p_with_nine_devices() -> None:
    deltas = np.linspace(0.01, 0.09, 9)
    assert exact_signed_rank_test(deltas, ContrastDirection.GREATER) == pytest.approx(1 / 512)
    assert exact_signed_rank_test(deltas, ContrastDirection.TWO_SIDED) == pytest.approx(2 / 512)


def test_signed_rank_drops_zero_differences() -> None:
    with_zero = np.array([0.0, 0.2, 0.3])
    without = np.array([0.2, 0.3])
    assert exact_signed_rank_test(with_zero, ContrastDirection.GREATER) == pytest.approx(
        exact_signed_rank_test(without, ContrastDirection.GREATER) or 0.0
    )


def test_tests_reject_too_many_devices() -> None:
    with pytest.raises(InsufficientDevicesError):
        exact_signed_rank_test(np.ones(25), ContrastDirection.GREATER)


def test_average_ranks_handles_ties() -> None:
    assert list(average_ranks(np.array([3.0, 1.0, 3.0, 2.0]))) == [3.5, 1.0, 3.5, 2.0]


def test_holm_adjustment_is_monotone_and_capped() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.6, 0.7]) == pytest.approx([1.0, 1.0])


def test_cluster_bootstrap_is_deterministic_and_contains_mean() -> None:
    values = np.array([0.1, 0.2, 0.15, 0.3, 0.05])
    first = cluster_bootstrap_interval(values, 500, 0.95, RandomSeed(7))
    second = cluster_bootstrap_interval(values, 500, 0.95, RandomSeed(7))
    assert first == second
    assert first[0] <= float(values.mean()) <= first[1]


def test_spearman_perfect_and_undefined() -> None:
    assert spearman(np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 40.0])) == 1.0
    assert spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) is None
    assert spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None


def test_worst_case_standard_error_matches_hanley_bound() -> None:
    assert worst_case_auroc_standard_error(100, 100) == pytest.approx(
        np.sqrt(201.0 / (12.0 * 100 * 100))
    )
    assert worst_case_auroc_standard_error(3, 2000) > worst_case_auroc_standard_error(3000, 2000)
    with pytest.raises(UndefinedMetricError):
        worst_case_auroc_standard_error(0, 5)
