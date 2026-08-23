from __future__ import annotations

import math
import statistics

import numpy as np
import pytest

from fedorbit.analysis.statistics import (
    BcaInterval,
    exact_sign_flip_test,
    holm_step_down,
    mcnemar_asymptotic_continuity_corrected_p,
    mcnemar_exact_p,
    mcnemar_test,
    minimum_valid_seeds_met,
    nominal_alpha,
    one_sided_sign_flip_p_value,
    paired_bca_interval,
    sign_flip_p_value,
    statistical_bootstrap_seed,
    tost_equivalence,
)
from fedorbit.config.loading import load_fedorbit_config


@pytest.fixture
def config():
    return load_fedorbit_config()


def test_nominal_alpha_is_derived_not_configured(config) -> None:
    level = config.scientific.statistics.confidence_level
    assert "alpha" not in set(dir(config.scientific.statistics))
    assert nominal_alpha(config) == pytest.approx(1.0 - level)


def test_sign_flip_all_zero_differences_returns_one() -> None:
    p = sign_flip_p_value((0.0, 0.0, 0.0), comparison_tolerance=1e-15)
    assert p == 1.0


def test_sign_flip_two_sided_matches_hand_enumeration() -> None:
    differences = (1.0, -1.0)
    observed_mean = 0.0
    extremes = 0
    for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        mean = (signs[0] * 1.0 + signs[1] * -1.0) / 2
        if abs(mean) >= abs(observed_mean) - 1e-15:
            extremes += 1
    expected = extremes / 4
    assert sign_flip_p_value(differences, 1e-15) == expected


def test_exact_sign_flip_test_reports_point_summaries(config) -> None:
    method = (2.0, 3.0, 2.5, 3.5)
    reference = (1.0, 1.5, 1.2, 1.1)
    result = exact_sign_flip_test(config, method, reference)
    differences = [m - r for m, r in zip(method, reference, strict=True)]
    assert result.mean_difference == pytest.approx(sum(differences) / len(differences))
    assert result.median_difference == pytest.approx(statistics.median(differences))
    assert result.nonzero_difference_count == 4


def test_one_sided_greater_is_half_of_two_sided_when_strict() -> None:
    differences = (0.4, 0.6, 0.5, 0.7)
    two_sided = sign_flip_p_value(differences, 1e-15)
    greater = one_sided_sign_flip_p_value(differences, "greater", 1e-15)
    assert greater <= two_sided + 1e-12
    assert greater >= two_sided / 2 - 1e-12


def test_bca_interval_contains_point_estimate_for_spread_data(
    config,
) -> None:
    rng = np.random.default_rng(42)
    reference = tuple(float(v) for v in rng.uniform(1.0, 2.0, size=10))
    method = tuple(value + 0.3 for value in reference)
    seed = statistical_bootstrap_seed(config, "contrast", "family", "pair", "metric", "ci")
    interval = paired_bca_interval(config, method, reference, seed)
    assert isinstance(interval, BcaInterval)
    assert not interval.degenerate
    assert interval.lower <= interval.point_estimate <= interval.upper


def test_identical_differences_return_point_interval_without_degenerate_flag(
    config,
) -> None:
    reference = (1.0, 2.0, 3.0, 4.0, 5.0)
    method = (value + 0.25 for value in reference)
    method_tuple = tuple(method)
    interval = paired_bca_interval(config, method_tuple, reference, 7)
    assert not interval.degenerate
    assert interval.lower == pytest.approx(interval.point_estimate)
    assert interval.upper == pytest.approx(interval.point_estimate)


def test_statistical_seed_depends_on_contrast_and_pair(config) -> None:
    first = statistical_bootstrap_seed(config, "c1", "fam", "pair-a", "m", "ci")
    second = statistical_bootstrap_seed(config, "c2", "fam", "pair-a", "m", "ci")
    third = statistical_bootstrap_seed(config, "c1", "fam", "pair-b", "m", "ci")
    assert len({first, second, third}) == 3


def test_holm_step_down_monotone_with_lexicographic_tie_break() -> None:
    raw = {"b": 0.03, "a": 0.03, "c": 0.01}
    adjusted = holm_step_down(raw)
    assert adjusted["c"] == pytest.approx(min(1.0, 0.01 * 3))
    assert adjusted["a"] == pytest.approx(min(1.0, 0.03 * 2))
    assert adjusted["b"] == pytest.approx(max(adjusted["a"], min(1.0, 0.03 * 1)))
    values = list(adjusted.values())
    assert values == sorted(values)


def test_mcnemar_exact_small_discordant_counts() -> None:
    assert mcnemar_exact_p(0, 0) == 1.0
    p_11 = mcnemar_exact_p(1, 1)
    assert p_11 == pytest.approx(1.0)
    p_30 = mcnemar_exact_p(3, 0)
    assert p_30 == pytest.approx(2 * (1 / 8))


def test_mcnemar_switch_by_configured_discordant_count(config) -> None:
    switch = config.scientific.statistics.mcnemar_exact_to_asymptotic_discordant_pair_switch
    kind_exact, _ = mcnemar_test(config, switch // 2, switch // 2)
    assert kind_exact == "exact"
    kind_asymptotic, p_asymptotic = mcnemar_test(config, switch + 10, 0)
    assert kind_asymptotic == "asymptotic"
    assert 0.0 <= p_asymptotic <= 1.0
    assert math.isfinite(mcnemar_asymptotic_continuity_corrected_p(switch + 10, 0))


def test_tost_equivalence_detects_within_margin_differences(config) -> None:
    reference = (1.0, 1.2, 0.9, 1.1, 1.05, 0.95)
    method = tuple(value + 0.005 for value in reference)
    tost = tost_equivalence(config, method, reference)
    alpha = config.scientific.statistics.tost_alpha_per_one_sided_test
    assert max(tost.p_lower, tost.p_upper) < alpha or tost.p_equiv < 0.25


def test_tost_rejects_outside_margin_differences(config) -> None:
    reference = (1.0,) * 8
    method = tuple(value + 0.05 for value in reference)
    tost = tost_equivalence(config, method, reference)
    alpha = config.scientific.statistics.tost_alpha_per_one_sided_test
    assert tost.p_equiv >= alpha


def test_minimum_seeds_gate(config) -> None:
    required = config.scientific.statistics.minimum_valid_paired_seeds
    assert minimum_valid_seeds_met(config, required)
    assert not minimum_valid_seeds_met(config, required - 1)
