from __future__ import annotations

import itertools
import math
import statistics
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.random import PCG64, Generator
from scipy import stats as scipy_stats

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.canonical import canonical_json
from fedorbit.domain.enums import RngNamespace
from fedorbit.runtime.seeds import derive_seed32


class StatisticsError(ValueError):
    pass


def nominal_alpha(config: FedorbitConfig) -> float:
    return 1.0 - config.scientific.statistics.confidence_level


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def sign_flip_p_value(
    differences: tuple[float, ...],
    comparison_tolerance: float,
) -> float:
    nonzero = [value for value in differences if value != 0.0]
    effective_count = len(nonzero)
    if effective_count == 0:
        return 1.0
    observed_mean = _mean(tuple(nonzero))
    extremes = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=effective_count):
        permuted_mean = math.fsum(s * v for s, v in zip(signs, nonzero, strict=True)) / (
            effective_count
        )
        if abs(permuted_mean) >= abs(observed_mean) - comparison_tolerance:
            extremes += 1
        total += 1
    return extremes / total


def one_sided_sign_flip_p_value(
    differences: tuple[float, ...],
    alternative: str,
    comparison_tolerance: float,
) -> float:
    if alternative not in {"greater", "less"}:
        raise StatisticsError(f"unsupported one-sided alternative: {alternative}")
    nonzero = [value for value in differences if value != 0.0]
    effective_count = len(nonzero)
    if effective_count == 0:
        return 1.0
    observed_mean = _mean(tuple(nonzero))
    extremes = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=effective_count):
        permuted_mean = math.fsum(s * v for s, v in zip(signs, nonzero, strict=True)) / (
            effective_count
        )
        if alternative == "greater":
            if permuted_mean >= observed_mean - comparison_tolerance:
                extremes += 1
        else:
            if permuted_mean <= observed_mean + comparison_tolerance:
                extremes += 1
        total += 1
    return extremes / total


@dataclass(frozen=True, slots=True)
class SignFlipResult:
    p_value: float
    mean_difference: float
    median_difference: float
    nonzero_difference_count: int


def exact_sign_flip_test(
    config: FedorbitConfig,
    method_values: tuple[float, ...],
    reference_values: tuple[float, ...],
) -> SignFlipResult:
    if len(method_values) != len(reference_values):
        raise StatisticsError("paired sample sizes differ")
    differences = tuple(
        method - reference
        for method, reference in zip(method_values, reference_values, strict=True)
    )
    tolerance = config.scientific.statistics.exact_sign_flip_comparison_tolerance
    p_value = sign_flip_p_value(differences, tolerance)
    return SignFlipResult(
        p_value=p_value,
        mean_difference=_mean(differences),
        median_difference=statistics.median(differences),
        nonzero_difference_count=sum(1 for d in differences if d != 0.0),
    )


def statistical_bootstrap_seed(
    config: FedorbitConfig,
    contrast_name: str,
    family: str,
    directed_pair: str,
    metric: str,
    purpose: str,
) -> int:
    base_seed = config.scientific.randomness.statistical_seed
    coordinates = {
        "contrast": contrast_name,
        "family": family,
        "metric": metric,
        "pair": directed_pair,
        "purpose": purpose,
    }
    return derive_seed32(base_seed, RngNamespace.STATISTICAL_BOOTSTRAP, canonical_json(coordinates))


@dataclass(frozen=True, slots=True)
class BcaInterval:
    lower: float | None
    upper: float | None
    point_estimate: float
    degenerate: bool


def paired_bca_interval(
    config: FedorbitConfig,
    method_values: tuple[float, ...],
    reference_values: tuple[float, ...],
    bootstrap_seed: int,
) -> BcaInterval:
    if len(method_values) != len(reference_values):
        raise StatisticsError("paired sample sizes differ")
    differences = tuple(
        method - reference
        for method, reference in zip(method_values, reference_values, strict=True)
    )
    point_estimate = _mean(differences)
    identical_tolerance = config.scientific.statistics.identical_difference_tolerance
    all_identical = all(abs(value - differences[0]) <= identical_tolerance for value in differences)
    if all_identical:
        return BcaInterval(point_estimate, point_estimate, point_estimate, False)
    method_array = np.asarray(method_values, dtype=np.float64)
    reference_array = np.asarray(reference_values, dtype=np.float64)

    def statistic(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.mean(x - y, axis=axis)

    rng = Generator(PCG64(bootstrap_seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scipy_stats.bootstrap(
            (method_array, reference_array),
            statistic,
            paired=True,
            vectorized=False,
            method="BCa",
            alternative="two-sided",
            confidence_level=config.scientific.statistics.confidence_level,
            n_resamples=config.scientific.statistics.ci_bootstrap_repetitions,
            rng=rng,
        )
    lower = float(result.confidence_interval.low)
    upper = float(result.confidence_interval.high)
    if not math.isfinite(lower) or not math.isfinite(upper):
        return BcaInterval(None, None, point_estimate, True)
    return BcaInterval(lower, upper, point_estimate, False)


@dataclass(frozen=True, slots=True)
class TostResult:
    p_lower: float
    p_upper: float
    p_equiv: float


def tost_equivalence(
    config: FedorbitConfig,
    method_values: tuple[float, ...],
    reference_values: tuple[float, ...],
) -> TostResult:
    margins = config.scientific.materiality.equivalence_relative_macro_ce
    tolerance = config.scientific.statistics.exact_sign_flip_comparison_tolerance
    differences = tuple(
        method - reference
        for method, reference in zip(method_values, reference_values, strict=True)
    )
    shifted_lower = tuple(d - margins.lower for d in differences)
    shifted_upper = tuple(d - margins.upper for d in differences)
    p_lower = one_sided_sign_flip_p_value(shifted_lower, "greater", tolerance)
    p_upper = one_sided_sign_flip_p_value(shifted_upper, "less", tolerance)
    return TostResult(p_lower=p_lower, p_upper=p_upper, p_equiv=max(p_lower, p_upper))


def holm_step_down(
    raw_p_values: dict[str, float],
) -> dict[str, float]:
    ordered = sorted(raw_p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    family_size = len(ordered)
    for index, (name, raw_p) in enumerate(ordered):
        scaled = min(1.0, raw_p * (family_size - index))
        running_max = max(running_max, scaled)
        adjusted[name] = running_max
    return adjusted


def mcnemar_exact_p(b01: int, b10: int) -> float:
    discordant = b01 + b10
    if discordant == 0:
        return 1.0
    count = min(b01, b10)
    tail = sum(math.comb(discordant, k) for k in range(0, count + 1))
    return min(1.0, 2.0 * tail / 2**discordant)


def mcnemar_asymptotic_continuity_corrected_p(b01: int, b10: int) -> float:
    discordant = b01 + b10
    if discordant == 0:
        return 1.0
    numerator = (abs(b01 - b10) - 1.0) ** 2
    denominator = b01 + b10
    chi_square = numerator / denominator
    survival = 1.0 - float(scipy_stats.chi2.cdf(chi_square, df=1))
    return max(0.0, min(1.0, survival))


def mcnemar_test(
    config: FedorbitConfig,
    b01: int,
    b10: int,
) -> tuple[str, float]:
    switch = config.scientific.statistics.mcnemar_exact_to_asymptotic_discordant_pair_switch
    discordant = b01 + b10
    if discordant <= switch:
        return "exact", mcnemar_exact_p(b01, b10)
    return "asymptotic", mcnemar_asymptotic_continuity_corrected_p(b01, b10)


def minimum_valid_seeds_met(config: FedorbitConfig, seed_count: int) -> bool:
    return seed_count >= config.scientific.statistics.minimum_valid_paired_seeds
