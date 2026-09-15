from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pytest
from numpy.random import PCG64, Generator
from scipy import stats as scipy_stats

from fedorbit.analysis.statistics import (
    NamedPValue,
    PValueSet,
    StatisticsError,
    exact_sign_flip_test,
    holm_step_down,
    minimum_valid_seeds_met,
    paired_bca_interval,
    sign_flip_p_value,
    statistical_bootstrap_seed,
    tost_equivalence,
)
from fedorbit.config.models import FedorbitConfig
from fedorbit.types import (
    BootstrapPurpose,
    ContrastName,
    DirectedPairName,
    MetricId,
    MultiplicityFamily,
    PValueName,
)


def _reference_two_sided_sign_flip_p_value(
    differences: tuple[float, ...],
    comparison_tolerance: float,
) -> float:
    nonzero = tuple(value for value in differences if value != 0.0)
    if not nonzero:
        return 1.0
    observed = sum(nonzero) / len(nonzero)
    extreme = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(nonzero)):
        permuted = sum(sign * value for sign, value in zip(signs, nonzero, strict=True)) / len(
            nonzero
        )
        extreme += int(abs(permuted) >= abs(observed) - comparison_tolerance)
    return extreme / 2 ** len(nonzero)


def _reference_one_sided_sign_flip_p_value(
    differences: tuple[float, ...],
    margin: float,
    greater: bool,
    comparison_tolerance: float,
) -> float:
    shifted = tuple(value - margin for value in differences)
    nonzero = tuple(value for value in shifted if value != 0.0)
    if not nonzero:
        return 1.0
    observed = sum(nonzero) / len(nonzero)
    extreme = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(nonzero)):
        permuted = sum(sign * value for sign, value in zip(signs, nonzero, strict=True)) / len(
            nonzero
        )
        if greater:
            extreme += int(permuted >= observed - comparison_tolerance)
        else:
            extreme += int(permuted <= observed + comparison_tolerance)
    return extreme / 2 ** len(nonzero)


def _bootstrap_statistic(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.asarray(np.mean(x - y, axis=axis), dtype=np.float64)


def _unpaired_bootstrap_statistic(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.asarray(np.mean(x, axis=axis) - np.mean(y, axis=axis), dtype=np.float64)


def test_exact_sign_flip_p_value_matches_a_hand_computed_enumeration(
    fedorbit_config: FedorbitConfig,
) -> None:
    tolerance = fedorbit_config.scientific.statistics.exact_sign_flip_comparison_tolerance
    assert sign_flip_p_value((1.0,), tolerance) == pytest.approx(1.0)
    assert sign_flip_p_value((1.0, 1.0), tolerance) == pytest.approx(0.5)
    assert sign_flip_p_value((1.0, -1.0), tolerance) == pytest.approx(1.0)
    assert sign_flip_p_value((1.0, 2.0), tolerance) == pytest.approx(0.5)
    assert sign_flip_p_value((1.0, 2.0, 3.0), tolerance) == pytest.approx(0.25)
    assert sign_flip_p_value((0.0, 0.0), tolerance) == pytest.approx(1.0)


def test_exact_sign_flip_p_value_matches_an_independent_enumeration(
    fedorbit_config: FedorbitConfig,
) -> None:
    tolerance = fedorbit_config.scientific.statistics.exact_sign_flip_comparison_tolerance
    cases = (
        (0.4, 0.9, 0.15, 0.7),
        (-0.2, 0.05, 0.3, -0.1, 0.25),
        (0.02, 0.0, -0.04, 0.06, 0.0, 0.01),
    )
    for case in cases:
        expected = _reference_two_sided_sign_flip_p_value(case, tolerance)
        assert sign_flip_p_value(case, tolerance) == pytest.approx(expected)


def test_sign_flip_tolerance_boundary_counts_near_identical_patterns(
    fedorbit_config: FedorbitConfig,
) -> None:
    tolerance = fedorbit_config.scientific.statistics.exact_sign_flip_comparison_tolerance
    differences = (1.0, tolerance / 10.0)
    assert sign_flip_p_value(differences, tolerance) == pytest.approx(1.0)
    assert sign_flip_p_value(differences, 0.0) == pytest.approx(0.5)


def test_exact_sign_flip_test_reports_point_summaries_and_zero_count(
    fedorbit_config: FedorbitConfig,
) -> None:
    del fedorbit_config
    identical = exact_sign_flip_test((0.5, 0.5), (0.5, 0.5))
    assert identical.p_value == pytest.approx(1.0)
    assert identical.nonzero_difference_count == 0
    assert identical.mean_difference == pytest.approx(0.0)
    assert identical.median_difference == pytest.approx(0.0)
    shifted = exact_sign_flip_test((1.0, 2.0, 10.0), (0.0,) * 3)
    assert shifted.mean_difference == pytest.approx(13.0 / 3.0)
    assert shifted.median_difference == pytest.approx(2.0)
    assert shifted.nonzero_difference_count == 3


def test_exact_enumeration_bound_comes_from_configuration(
    fedorbit_config: FedorbitConfig,
) -> None:
    statistics_config = fedorbit_config.scientific.statistics
    maximum = statistics_config.exact_sign_flip_max_nonzero_differences_for_enumeration
    within_bound = exact_sign_flip_test((1.0,) * 3, (0.0,) * 3)
    assert within_bound.nonzero_difference_count == 3
    beyond_bound = (1.0,) * (maximum + 1)
    with pytest.raises(StatisticsError):
        exact_sign_flip_test(beyond_bound, (0.0,) * (maximum + 1))


def test_paired_bca_interval_reproduces_the_pinned_scipy_paired_call(
    fedorbit_config: FedorbitConfig,
) -> None:
    statistics_config = fedorbit_config.scientific.statistics
    method_values = (
        0.30,
        0.52,
        0.11,
        0.47,
        0.62,
        0.25,
        0.39,
        0.58,
    )
    reference_values = (
        0.20,
        0.31,
        0.09,
        0.35,
        0.41,
        0.18,
        0.27,
        0.44,
    )
    derived_seed = statistical_bootstrap_seed(
        ContrastName("FedORBIT Exact-Sparse Solver vs Local-Only"),
        MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
        DirectedPairName("client-a -> client-b"),
        MetricId.MACRO_CROSS_ENTROPY,
        BootstrapPurpose.PRIMARY_TRANSFER_GAIN,
    )
    interval = paired_bca_interval(method_values, reference_values, derived_seed)
    method_array = np.asarray(method_values, dtype=np.float64)
    reference_array = np.asarray(reference_values, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paired_reference = scipy_stats.bootstrap(
            (method_array, reference_array),
            _bootstrap_statistic,
            paired=True,
            vectorized=False,
            method="BCa",
            alternative="two-sided",
            confidence_level=statistics_config.confidence_level,
            n_resamples=statistics_config.ci_bootstrap_repetitions,
            rng=Generator(PCG64(derived_seed)),
        )
        unpaired_reference = scipy_stats.bootstrap(
            (method_array, reference_array),
            _unpaired_bootstrap_statistic,
            paired=False,
            vectorized=False,
            method="BCa",
            alternative="two-sided",
            confidence_level=statistics_config.confidence_level,
            n_resamples=statistics_config.ci_bootstrap_repetitions,
            rng=Generator(PCG64(derived_seed)),
        )
    paired_low = float(paired_reference.confidence_interval.low)
    paired_high = float(paired_reference.confidence_interval.high)
    assert interval.lower == pytest.approx(paired_low)
    assert interval.upper == pytest.approx(paired_high)
    assert interval.degenerate is False
    expected_point = sum(
        method - reference
        for method, reference in zip(method_values, reference_values, strict=True)
    ) / len(method_values)
    assert interval.point_estimate == pytest.approx(expected_point)
    assert not math.isclose(float(unpaired_reference.confidence_interval.low), paired_low)


def test_statistical_bootstrap_seed_is_deterministic_and_purpose_specific(
    fedorbit_config: FedorbitConfig,
) -> None:
    del fedorbit_config
    first = statistical_bootstrap_seed(
        ContrastName("contrast"),
        MultiplicityFamily.COUPLING_MECHANISM,
        DirectedPairName("client-a -> client-b"),
        MetricId.ROBUST_COUPLING_VALUE_GAP,
        BootstrapPurpose.COUPLING_MECHANISM_GAP,
    )
    repeated = statistical_bootstrap_seed(
        ContrastName("contrast"),
        MultiplicityFamily.COUPLING_MECHANISM,
        DirectedPairName("client-a -> client-b"),
        MetricId.ROBUST_COUPLING_VALUE_GAP,
        BootstrapPurpose.COUPLING_MECHANISM_GAP,
    )
    other_purpose = statistical_bootstrap_seed(
        ContrastName("contrast"),
        MultiplicityFamily.COUPLING_MECHANISM,
        DirectedPairName("client-a -> client-b"),
        MetricId.ROBUST_COUPLING_VALUE_GAP,
        BootstrapPurpose.MECHANISM_ABLATIONS_DIFFERENCE,
    )
    assert isinstance(first, int)
    assert first == repeated
    assert first != other_purpose


def test_tost_equivalence_matches_independent_one_sided_enumeration(
    fedorbit_config: FedorbitConfig,
) -> None:
    margins = fedorbit_config.scientific.materiality.equivalence_relative_macro_ce
    tolerance = fedorbit_config.scientific.statistics.exact_sign_flip_comparison_tolerance
    method_values = (0.05, 0.08, -0.01, 0.04, 0.02)
    reference_values = (0.0,) * 5
    differences = tuple(
        method - reference
        for method, reference in zip(method_values, reference_values, strict=True)
    )
    result = tost_equivalence(method_values, reference_values)
    expected_lower = _reference_one_sided_sign_flip_p_value(
        differences, margins.lower, True, tolerance
    )
    expected_upper = _reference_one_sided_sign_flip_p_value(
        differences, margins.upper, False, tolerance
    )
    assert result.p_lower == pytest.approx(expected_lower)
    assert result.p_upper == pytest.approx(expected_upper)
    assert result.p_equiv == pytest.approx(max(result.p_lower, result.p_upper))


def test_tost_equivalence_rejects_large_differences_in_both_directions(
    fedorbit_config: FedorbitConfig,
) -> None:
    del fedorbit_config
    identical = tost_equivalence((0.0,), (0.0,))
    assert identical.p_lower == pytest.approx(0.5)
    assert identical.p_upper == pytest.approx(0.5)
    assert identical.p_equiv == pytest.approx(0.5)
    worse = tost_equivalence((0.5,), (0.0,))
    assert worse.p_lower == pytest.approx(0.5)
    assert worse.p_upper == pytest.approx(1.0)
    assert worse.p_equiv == pytest.approx(1.0)
    better = tost_equivalence((-0.5,), (0.0,))
    assert better.p_lower == pytest.approx(1.0)
    assert better.p_upper == pytest.approx(0.5)
    assert better.p_equiv == pytest.approx(1.0)


def test_holm_step_down_orders_ties_by_raw_p_then_lexicographic_contrast(
    fedorbit_config: FedorbitConfig,
) -> None:
    del fedorbit_config
    raw = PValueSet(
        (
            NamedPValue(PValueName("zeta"), 0.01),
            NamedPValue(PValueName("alpha"), 0.01),
            NamedPValue(PValueName("beta"), 0.5),
        )
    )
    adjusted = holm_step_down(raw)
    assert tuple(entry.name for entry in adjusted.entries) == ("alpha", "zeta", "beta")
    assert adjusted.value_of(PValueName("alpha")) == pytest.approx(0.03)
    assert adjusted.value_of(PValueName("zeta")) == pytest.approx(0.03)
    assert adjusted.value_of(PValueName("beta")) == pytest.approx(0.5)
    reordered = holm_step_down(
        PValueSet(
            (
                NamedPValue(PValueName("beta"), 0.5),
                NamedPValue(PValueName("alpha"), 0.01),
                NamedPValue(PValueName("zeta"), 0.01),
            )
        )
    )
    assert tuple(entry.name for entry in reordered.entries) == ("alpha", "zeta", "beta")
    assert reordered.value_of(PValueName("alpha")) == pytest.approx(0.03)
    assert reordered.value_of(PValueName("beta")) == pytest.approx(0.5)


def test_minimum_valid_paired_seed_boundary_comes_from_configuration(
    fedorbit_config: FedorbitConfig,
) -> None:
    minimum = fedorbit_config.scientific.statistics.minimum_valid_paired_seeds
    assert minimum_valid_seeds_met(minimum)
    assert not minimum_valid_seeds_met(minimum - 1)
