from __future__ import annotations

import itertools
import math
import statistics
import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

import numpy as np
from numpy.random import PCG64, Generator
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from fedorbit.config.context import active_config
from fedorbit.domain.enums import RngNamespace
from fedorbit.runtime.seeds import derive_seed32

FloatArray = NDArray[np.float64]


class StatisticsError(ValueError):
    pass


class McNemarMode(StrEnum):
    EXACT = "exact"
    ASYMPTOTIC = "asymptotic"


@dataclass(frozen=True, slots=True)
class NamedPValue:
    name: str
    p_value: float

    def __post_init__(self) -> None:
        if not self.name:
            raise StatisticsError("p-value name must be non-empty")
        if not math.isfinite(self.p_value) or not 0.0 <= self.p_value <= 1.0:
            raise StatisticsError("p-value must be finite and lie in [0,1]")


@dataclass(frozen=True, slots=True)
class PValueSet:
    entries: tuple[NamedPValue, ...]

    def __post_init__(self) -> None:
        names = tuple(entry.name for entry in self.entries)
        if len(set(names)) != len(names):
            raise StatisticsError("p-value set contains duplicate names")

    def value_of(self, name: str) -> float | None:
        for entry in self.entries:
            if entry.name == name:
                return entry.p_value
        return None


@dataclass(frozen=True, slots=True)
class McNemarResult:
    mode: McNemarMode
    p_value: float


class _ConfidenceIntervalLike(Protocol):
    low: float
    high: float


class _BootstrapResultLike(Protocol):
    confidence_interval: _ConfidenceIntervalLike


class _ChiSquareDistribution(Protocol):
    cdf: Callable[..., float]


class _ScipyStats(Protocol):
    bootstrap: Callable[..., _BootstrapResultLike]
    chi2: _ChiSquareDistribution


_TYPED_SCIPY_STATS = cast(_ScipyStats, scipy_stats)
_BOOTSTRAP = _TYPED_SCIPY_STATS.bootstrap
_CHI_SQUARE_CDF = _TYPED_SCIPY_STATS.chi2.cdf


def nominal_alpha() -> float:
    return 1.0 - active_config().scientific.statistics.confidence_level


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        raise StatisticsError("statistical mean requires at least one value")
    return math.fsum(values) / len(values)


def sign_flip_p_value(
    differences: tuple[float, ...],
    comparison_tolerance: float,
) -> float:
    nonzero = tuple(value for value in differences if value != 0.0)
    if not nonzero:
        return 1.0
    observed_mean = _mean(nonzero)
    extremes = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(nonzero)):
        permuted_mean = math.fsum(
            sign * value for sign, value in zip(signs, nonzero, strict=True)
        ) / len(nonzero)
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
    nonzero = tuple(value for value in differences if value != 0.0)
    if not nonzero:
        return 1.0
    observed_mean = _mean(nonzero)
    extremes = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(nonzero)):
        permuted_mean = math.fsum(
            sign * value for sign, value in zip(signs, nonzero, strict=True)
        ) / len(nonzero)
        if alternative == "greater":
            extreme = permuted_mean >= observed_mean - comparison_tolerance
        else:
            extreme = permuted_mean <= observed_mean + comparison_tolerance
        extremes += int(extreme)
        total += 1
    return extremes / total


@dataclass(frozen=True, slots=True)
class SignFlipResult:
    p_value: float
    mean_difference: float
    median_difference: float
    nonzero_difference_count: int


def exact_sign_flip_test(
    method_values: tuple[float, ...],
    reference_values: tuple[float, ...],
) -> SignFlipResult:
    if len(method_values) != len(reference_values):
        raise StatisticsError("paired sample sizes differ")
    if not method_values:
        raise StatisticsError("paired sign-flip test requires at least one pair")
    differences = tuple(
        method - reference
        for method, reference in zip(method_values, reference_values, strict=True)
    )
    nonzero_count = sum(value != 0.0 for value in differences)
    statistics_config = active_config().scientific.statistics
    maximum = statistics_config.exact_sign_flip_max_nonzero_differences_for_enumeration
    if nonzero_count > maximum:
        raise StatisticsError(
            f"exact sign-flip enumeration has {nonzero_count} nonzero differences; "
            f"maximum is {maximum}"
        )
    tolerance = statistics_config.exact_sign_flip_comparison_tolerance
    return SignFlipResult(
        p_value=sign_flip_p_value(differences, tolerance),
        mean_difference=_mean(differences),
        median_difference=statistics.median(differences),
        nonzero_difference_count=nonzero_count,
    )


def statistical_bootstrap_seed(
    contrast_name: str,
    family: str,
    directed_pair: str,
    metric: str,
    purpose: str,
) -> int:
    coordinates: OrderedDict[str, str] = OrderedDict(
        contrast=contrast_name,
        family=family,
        metric=metric,
        pair=directed_pair,
        purpose=purpose,
    )
    return derive_seed32(
        active_config().scientific.randomness.statistical_seed,
        RngNamespace.STATISTICAL_BOOTSTRAP,
        coordinates,
    )


@dataclass(frozen=True, slots=True)
class BcaInterval:
    lower: float | None
    upper: float | None
    point_estimate: float
    degenerate: bool


def paired_bca_interval(
    method_values: tuple[float, ...],
    reference_values: tuple[float, ...],
    bootstrap_seed: int,
) -> BcaInterval:
    if len(method_values) != len(reference_values):
        raise StatisticsError("paired sample sizes differ")
    if not method_values:
        raise StatisticsError("paired BCa interval requires at least one pair")
    differences = tuple(
        method - reference
        for method, reference in zip(method_values, reference_values, strict=True)
    )
    if any(not math.isfinite(value) for value in differences):
        return BcaInterval(None, None, math.nan, True)
    point_estimate = _mean(differences)
    statistics_config = active_config().scientific.statistics
    identical_tolerance = statistics_config.identical_difference_tolerance
    if all(abs(value - differences[0]) <= identical_tolerance for value in differences):
        return BcaInterval(point_estimate, point_estimate, point_estimate, False)
    method_array = np.asarray(method_values, dtype=np.float64)
    reference_array = np.asarray(reference_values, dtype=np.float64)

    def statistic(x: FloatArray, y: FloatArray, axis: int = -1) -> FloatArray:
        return np.asarray(np.mean(x - y, axis=axis), dtype=np.float64)

    rng = Generator(PCG64(bootstrap_seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _BOOTSTRAP(
            (method_array, reference_array),
            statistic,
            paired=True,
            vectorized=False,
            method="BCa",
            alternative="two-sided",
            confidence_level=statistics_config.confidence_level,
            n_resamples=statistics_config.ci_bootstrap_repetitions,
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
    method_values: tuple[float, ...],
    reference_values: tuple[float, ...],
) -> TostResult:
    if len(method_values) != len(reference_values):
        raise StatisticsError("paired sample sizes differ")
    config = active_config()
    margins = config.scientific.materiality.equivalence_relative_macro_ce
    tolerance = config.scientific.statistics.exact_sign_flip_comparison_tolerance
    differences = tuple(
        method - reference
        for method, reference in zip(method_values, reference_values, strict=True)
    )
    shifted_lower = tuple(difference - margins.lower for difference in differences)
    shifted_upper = tuple(difference - margins.upper for difference in differences)
    p_lower = one_sided_sign_flip_p_value(shifted_lower, "greater", tolerance)
    p_upper = one_sided_sign_flip_p_value(shifted_upper, "less", tolerance)
    return TostResult(p_lower, p_upper, max(p_lower, p_upper))


def holm_step_down(raw_p_values: PValueSet) -> PValueSet:
    ordered = sorted(raw_p_values.entries, key=lambda entry: (entry.p_value, entry.name))
    adjusted: list[NamedPValue] = []
    running_max = 0.0
    family_size = len(ordered)
    for index, entry in enumerate(ordered):
        scaled = min(1.0, entry.p_value * (family_size - index))
        running_max = max(running_max, scaled)
        adjusted.append(NamedPValue(entry.name, running_max))
    return PValueSet(tuple(adjusted))


def mcnemar_exact_p(b01: int, b10: int) -> float:
    if b01 < 0 or b10 < 0:
        raise StatisticsError("McNemar discordant counts must be nonnegative")
    discordant = b01 + b10
    if discordant == 0:
        return 1.0
    count = min(b01, b10)
    tail = sum(math.comb(discordant, k) for k in range(count + 1))
    return min(1.0, 2.0 * tail / 2**discordant)


def mcnemar_asymptotic_continuity_corrected_p(b01: int, b10: int) -> float:
    if b01 < 0 or b10 < 0:
        raise StatisticsError("McNemar discordant counts must be nonnegative")
    discordant = b01 + b10
    if discordant == 0:
        return 1.0
    chi_square = (abs(b01 - b10) - 1.0) ** 2 / discordant
    survival = 1.0 - _CHI_SQUARE_CDF(chi_square, df=1)
    return max(0.0, min(1.0, survival))


def mcnemar_test(
    b01: int,
    b10: int,
) -> McNemarResult:
    switch = (
        active_config().scientific.statistics.mcnemar_exact_to_asymptotic_discordant_pair_switch
    )
    if b01 + b10 <= switch:
        return McNemarResult(McNemarMode.EXACT, mcnemar_exact_p(b01, b10))
    return McNemarResult(
        McNemarMode.ASYMPTOTIC,
        mcnemar_asymptotic_continuity_corrected_p(b01, b10),
    )


def minimum_valid_seeds_met(seed_count: int) -> bool:
    return seed_count >= active_config().scientific.statistics.minimum_valid_paired_seeds
