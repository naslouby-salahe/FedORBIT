from __future__ import annotations

import math
import statistics
from collections import OrderedDict
from dataclasses import dataclass

from fedorbit.config.loading import active_config
from fedorbit.types import MetricId


class MetricComputationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Probability:
    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise MetricComputationError(f"class probability outside [0,1]: {self.value}")


@dataclass(frozen=True, slots=True)
class TrueClassProbabilities:
    values: tuple[Probability, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise MetricComputationError(
                "evaluation class with zero examples makes the cell Invalid Data"
            )


@dataclass(frozen=True, slots=True)
class CrossEntropy:
    value: float


@dataclass(frozen=True, slots=True)
class ClassEntropySet:
    values: tuple[CrossEntropy, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise MetricComputationError(
                "fixed evaluation class set is empty; the cell is Invalid Data"
            )


@dataclass(frozen=True, slots=True)
class ClassF1:
    value: float


@dataclass(frozen=True, slots=True)
class ClassF1Set:
    values: tuple[ClassF1, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise MetricComputationError("macro-F1 over an empty evaluation class set")


@dataclass(frozen=True, slots=True)
class ClassRecall:
    value: float


@dataclass(frozen=True, slots=True)
class ClassRecallSet:
    values: tuple[ClassRecall, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise MetricComputationError("balanced accuracy over an empty evaluation class set")


@dataclass(frozen=True, slots=True)
class RelativeMacroCeGain:
    value: float | None
    absolute_difference: float
    is_na: bool


def class_conditional_cross_entropy(
    true_class_probabilities: TrueClassProbabilities,
) -> CrossEntropy:
    log_floor = active_config().scientific.metrics.probability_log_floor
    total = 0.0
    for probability in true_class_probabilities.values:
        total += -math.log(max(probability.value, log_floor))
    return CrossEntropy(total / len(true_class_probabilities.values))


def macro_cross_entropy(class_entropies: ClassEntropySet) -> CrossEntropy:
    return CrossEntropy(statistics.fmean(entry.value for entry in class_entropies.values))


def relative_macro_ce_gain(
    reference_macro_ce: float,
    method_macro_ce: float,
) -> RelativeMacroCeGain:
    floor = active_config().scientific.metrics.relative_macro_ce_denominator_floor
    absolute_difference = reference_macro_ce - method_macro_ce
    if reference_macro_ce < floor:
        return RelativeMacroCeGain(
            value=None,
            absolute_difference=absolute_difference,
            is_na=True,
        )
    return RelativeMacroCeGain(
        value=absolute_difference / max(reference_macro_ce, floor),
        absolute_difference=absolute_difference,
        is_na=False,
    )


def precision_from_counts(true_positives: int, false_positives: int) -> float:
    denominator = true_positives + false_positives
    if denominator == 0:
        return 0.0
    return true_positives / denominator


def recall_from_counts(true_positives: int, false_negatives: int) -> float:
    denominator = true_positives + false_negatives
    if denominator == 0:
        return 0.0
    return true_positives / denominator


def f1_from_counts(true_positives: int, false_positives: int, false_negatives: int) -> float:
    precision_value = precision_from_counts(true_positives, false_positives)
    recall_value = recall_from_counts(true_positives, false_negatives)
    denominator = precision_value + recall_value
    if denominator == 0:
        return 0.0
    return 2 * precision_value * recall_value / denominator


def macro_f1(per_class_f1: ClassF1Set) -> ClassF1:
    return ClassF1(statistics.fmean(value.value for value in per_class_f1.values))


def balanced_accuracy(per_class_recall: ClassRecallSet) -> ClassRecall:
    return ClassRecall(statistics.fmean(value.value for value in per_class_recall.values))


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    true_positives: int
    false_positives: int
    false_negatives: int


def confusion_counts(
    predicted_labels: tuple[int, ...],
    true_labels: tuple[int, ...],
    positive_class: int,
) -> ConfusionCounts:
    if len(predicted_labels) != len(true_labels):
        raise MetricComputationError("prediction and label counts differ")
    true_positives = sum(
        1
        for predicted, actual in zip(predicted_labels, true_labels, strict=True)
        if predicted == actual == positive_class
    )
    false_positives = sum(
        1
        for predicted, actual in zip(predicted_labels, true_labels, strict=True)
        if predicted == positive_class and actual != positive_class
    )
    false_negatives = sum(
        1
        for predicted, actual in zip(predicted_labels, true_labels, strict=True)
        if actual == positive_class and predicted != positive_class
    )
    return ConfusionCounts(true_positives, false_positives, false_negatives)


def certified_robust_predicted_value(certified_objective: float) -> float:
    return certified_objective


def fixed_action_rectangularization_gap_metric(gap: float) -> float:
    return gap


def robust_coupling_value_gap_metric(gap: float) -> float:
    return gap


def coupling_upper_bound_diagnostic_metric(value: float) -> float:
    return value


def exact_map_action_value_metric(delta_map: float) -> float:
    return delta_map


def absolute_objective_error(objective_value: float, truth_value: float) -> float:
    return abs(objective_value - truth_value)


def relative_objective_error(
    objective_value: float,
    truth_value: float,
) -> float:
    floor = active_config().scientific.metrics.relative_solver_error_denominator_floor
    return abs(objective_value - truth_value) / max(abs(truth_value), floor)


@dataclass(frozen=True, slots=True)
class ProposalOutcomeTally:
    proposed: int
    accepted: int
    harmful_accepted: int
    useful_accepted: int

    def __post_init__(self) -> None:
        for name, value in (
            ("proposed", self.proposed),
            ("accepted", self.accepted),
            ("harmful_accepted", self.harmful_accepted),
            ("useful_accepted", self.useful_accepted),
        ):
            if value < 0:
                raise MetricComputationError(f"negative {name} count")
        if self.accepted > self.proposed or self.harmful_accepted > self.accepted:
            raise MetricComputationError("proposal outcome counts are inconsistent")


@dataclass(frozen=True, slots=True)
class ProposalRates:
    acceptance_rate: float | None
    harmful_accepted_rate: float | None
    useful_accepted_rate: float | None

    @property
    def all_na(self) -> bool:
        return (
            self.acceptance_rate is None
            and self.harmful_accepted_rate is None
            and self.useful_accepted_rate is None
        )


def proposal_rates(tally: ProposalOutcomeTally) -> ProposalRates:
    if tally.proposed == 0:
        return ProposalRates(None, None, None)
    return ProposalRates(
        acceptance_rate=tally.accepted / tally.proposed,
        harmful_accepted_rate=tally.harmful_accepted / tally.proposed,
        useful_accepted_rate=tally.useful_accepted / tally.proposed,
    )


def confirmation_coverage(live_transfer_decisions: int, eligible_decisions: int) -> float | None:
    if eligible_decisions == 0:
        return None
    return live_transfer_decisions / eligible_decisions


def no_confirmation_coverage(eligible_decisions: int) -> float | None:
    if eligible_decisions == 0:
        return None
    return 1.0


def coverage_loss(
    coverage_no_confirm: float | None, coverage_confirm: float | None
) -> float | None:
    if coverage_no_confirm is None or coverage_confirm is None:
        return None
    return coverage_no_confirm - coverage_confirm


def harm_indicator(test_gain: float, harmful_threshold: float) -> bool:
    return test_gain <= harmful_threshold


def seed_harm_rate(decision_gains: tuple[float, ...], harmful_threshold: float) -> float | None:
    if not decision_gains:
        return None
    indicators = [harm_indicator(gain, harmful_threshold) for gain in decision_gains]
    return sum(1 for indicator in indicators if indicator) / len(indicators)


def absolute_risk_reduction(
    harm_rate_no_confirm: float | None, harm_rate_confirm: float | None
) -> float | None:
    if harm_rate_no_confirm is None or harm_rate_confirm is None:
        return None
    return harm_rate_no_confirm - harm_rate_confirm


def relative_risk_reduction(
    harm_rate_no_confirm: float | None,
    risk_reduction: float | None,
) -> float | None:
    if harm_rate_no_confirm is None or risk_reduction is None or harm_rate_no_confirm <= 0.0:
        return None
    return risk_reduction / harm_rate_no_confirm


def beneficial_rejected_rate(
    rejected_with_counterfactual_gain: int,
    proposed: int,
) -> float | None:
    if proposed == 0:
        return None
    return rejected_with_counterfactual_gain / proposed


def pair_mean(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def equal_pair_mean(pair_means_values: tuple[float | None, ...]) -> float | None:
    present = [value for value in pair_means_values if value is not None]
    if not present:
        return None
    return statistics.fmean(present)


def equal_pair_absolute_risk_reduction(
    equal_pair_harm_no_confirm: float | None,
    equal_pair_harm_confirm: float | None,
) -> float | None:
    return absolute_risk_reduction(equal_pair_harm_no_confirm, equal_pair_harm_confirm)


def equal_pair_relative_risk_reduction(
    equal_pair_harm_no_confirm: float | None,
    equal_pair_risk_reduction: float | None,
) -> float | None:
    return relative_risk_reduction(equal_pair_harm_no_confirm, equal_pair_risk_reduction)


METRIC_NAMES: OrderedDict[str, MetricId] = OrderedDict(
    certified_value=MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE,
    rectangularization_gap=MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP,
    coupling_gap=MetricId.ROBUST_COUPLING_VALUE_GAP,
    upper_bound=MetricId.COUPLING_UPPER_BOUND_DIAGNOSTIC,
    map_value=MetricId.EXACT_MAP_ACTION_VALUE,
)


class EfficiencyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EfficiencyRecord:
    wall_time_seconds: float
    peak_host_rss_mib: float
    peak_cuda_allocated_bytes: int
    packet_serialized_byte_count: int
    source_response_optimizer_steps: int
    target_confirmation_optimizer_steps: int
    live_assimilation_optimizer_steps: int
    timeout_indicator: bool
    resource_limit_indicator: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds < 0.0:
            raise EfficiencyError("wall time must be finite and nonnegative")
        if not math.isfinite(self.peak_host_rss_mib) or self.peak_host_rss_mib < 0.0:
            raise EfficiencyError("peak host RSS must be finite and nonnegative")
        for name, value in (
            ("CUDA bytes", self.peak_cuda_allocated_bytes),
            ("packet bytes", self.packet_serialized_byte_count),
            ("source response steps", self.source_response_optimizer_steps),
            ("confirmation steps", self.target_confirmation_optimizer_steps),
            ("assimilation steps", self.live_assimilation_optimizer_steps),
        ):
            if value < 0:
                raise EfficiencyError(f"{name} must be nonnegative")
