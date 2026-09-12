from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from fedorbit.config.loading import active_config
from fedorbit.types import (
    ByteCount,
    ClassIndex,
    EfficiencyMetricName,
    ElapsedSeconds,
    Estimate,
    Fraction,
    Index,
    MemoryMib,
    RelativeGain,
    SampleCount,
    Score,
    StepCount,
    Threshold,
)


class MetricComputationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Probability:
    value: Fraction

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
    value: Estimate


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
    value: Fraction


@dataclass(frozen=True, slots=True)
class ClassF1Set:
    values: tuple[ClassF1, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise MetricComputationError("macro-F1 over an empty evaluation class set")


@dataclass(frozen=True, slots=True)
class ClassRecall:
    value: Fraction


@dataclass(frozen=True, slots=True)
class ClassRecallSet:
    values: tuple[ClassRecall, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise MetricComputationError("balanced accuracy over an empty evaluation class set")


@dataclass(frozen=True, slots=True)
class RelativeMacroCeGain:
    value: RelativeGain | None
    absolute_difference: RelativeGain
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
    reference_macro_ce: Score,
    method_macro_ce: Score,
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


def precision_from_counts(true_positives: Index, false_positives: Index) -> Fraction:
    denominator = true_positives + false_positives
    if denominator == 0:
        zero: Fraction = 0.0
        return zero
    precision: Fraction = true_positives / denominator
    return precision


def recall_from_counts(true_positives: Index, false_negatives: Index) -> Fraction:
    denominator = true_positives + false_negatives
    if denominator == 0:
        zero: Fraction = 0.0
        return zero
    recall: Fraction = true_positives / denominator
    return recall


def f1_from_counts(
    true_positives: Index,
    false_positives: Index,
    false_negatives: Index,
) -> Fraction:
    precision_value = precision_from_counts(true_positives, false_positives)
    recall_value = recall_from_counts(true_positives, false_negatives)
    denominator = precision_value + recall_value
    if denominator == 0:
        zero: Fraction = 0.0
        return zero
    f1: Fraction = 2 * precision_value * recall_value / denominator
    return f1


def macro_f1(per_class_f1: ClassF1Set) -> ClassF1:
    return ClassF1(statistics.fmean(value.value for value in per_class_f1.values))


def balanced_accuracy(per_class_recall: ClassRecallSet) -> ClassRecall:
    return ClassRecall(statistics.fmean(value.value for value in per_class_recall.values))


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    true_positives: Index
    false_positives: Index
    false_negatives: Index


def confusion_counts(
    predicted_labels: tuple[ClassIndex, ...],
    true_labels: tuple[ClassIndex, ...],
    positive_class: ClassIndex,
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
    true_positive_count: Index = true_positives
    false_positive_count: Index = false_positives
    false_negative_count: Index = false_negatives
    return ConfusionCounts(
        true_positive_count,
        false_positive_count,
        false_negative_count,
    )


def certified_robust_predicted_value(certified_objective: Score) -> Score: #TODO: what's the point of this? Does it need more code or better wiring? Analyze properly
    return certified_objective


def fixed_action_rectangularization_gap_metric(gap: Score) -> Score :#TODO: what's the point of this? Does it need more code or better wiring? Analyze properly
    return gap


def robust_coupling_value_gap_metric(gap: Score) -> Score: #TODO: what's the point of this? Does it need more code or better wiring? Analyze properly
    return gap


def coupling_upper_bound_diagnostic_metric(value: Score) -> Score: #TODO: what's the point of this? Does it need more code or better wiring? Analyze properly
    return value


def exact_map_action_value_metric(delta_map: Score) -> Score: #TODO: what's the point of this? Does it need more code or better wiring? Analyze properly
    return delta_map


def absolute_objective_error(objective_value: Score, truth_value: Score) -> Score: #TODO: what's the point of this? Does it need more code or better wiring? Analyze properly. Or should it be inlined
    return abs(objective_value - truth_value)


def relative_objective_error(
    objective_value: Score,
    truth_value: Score,
) -> RelativeGain:
    floor = active_config().scientific.metrics.relative_solver_error_denominator_floor
    error: RelativeGain = abs(objective_value - truth_value) / max(abs(truth_value), floor)
    return error


@dataclass(frozen=True, slots=True)
class ProposalOutcomeTally:
    proposed: SampleCount
    accepted: SampleCount
    harmful_accepted: SampleCount
    useful_accepted: SampleCount

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
    acceptance_rate: Fraction | None
    harmful_accepted_rate: Fraction | None
    useful_accepted_rate: Fraction | None

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


def confirmation_coverage(
    live_transfer_decisions: Index,
    eligible_decisions: Index,
) -> Fraction | None:
    if eligible_decisions == 0:
        return None
    coverage: Fraction = live_transfer_decisions / eligible_decisions
    return coverage


def no_confirmation_coverage(eligible_decisions: Index) -> Fraction | None: #TODO: what's the point of this? Does it need more code or better wiring? Analyze properly. Or should it be inlined
    if eligible_decisions == 0:
        return None
    full_coverage: Fraction = 1.0
    return full_coverage


def coverage_loss(
    coverage_no_confirm: Fraction | None,
    coverage_confirm: Fraction | None,
) -> RelativeGain | None:
    if coverage_no_confirm is None or coverage_confirm is None:
        return None
    loss: RelativeGain = coverage_no_confirm - coverage_confirm
    return loss


def harm_indicator(test_gain: RelativeGain, harmful_threshold: Threshold) -> bool:
    return test_gain <= harmful_threshold


def seed_harm_rate(
    decision_gains: tuple[RelativeGain, ...], harmful_threshold: Threshold
) -> Fraction | None:
    if not decision_gains:
        return None
    indicators = [harm_indicator(gain, harmful_threshold) for gain in decision_gains]
    rate: Fraction = sum(1 for indicator in indicators if indicator) / len(indicators)
    return rate


def absolute_risk_reduction(
    harm_rate_no_confirm: Fraction | None,
    harm_rate_confirm: Fraction | None,
) -> RelativeGain | None:
    if harm_rate_no_confirm is None or harm_rate_confirm is None:
        return None
    reduction: RelativeGain = harm_rate_no_confirm - harm_rate_confirm
    return reduction


def relative_risk_reduction(
    harm_rate_no_confirm: Fraction | None,
    risk_reduction: RelativeGain | None,
) -> RelativeGain | None:
    if harm_rate_no_confirm is None or risk_reduction is None or harm_rate_no_confirm <= 0.0:
        return None
    reduction: RelativeGain = risk_reduction / harm_rate_no_confirm
    return reduction


def beneficial_rejected_rate(
    rejected_with_counterfactual_gain: Index,
    proposed: SampleCount,
) -> Fraction | None:
    if proposed == 0:
        return None
    rate: Fraction = rejected_with_counterfactual_gain / proposed
    return rate


def pair_mean(values: tuple[RelativeGain, ...]) -> RelativeGain | None:
    if not values:
        return None
    mean: RelativeGain = statistics.fmean(values)
    return mean


def equal_pair_mean(
    pair_means_values: tuple[RelativeGain | None, ...],
) -> RelativeGain | None:
    present = [value for value in pair_means_values if value is not None]
    if not present:
        return None
    mean: RelativeGain = statistics.fmean(present)
    return mean


def equal_pair_absolute_risk_reduction( #TODO: what's the point of this? Does it need more code or better wiring? Analyze properly. Or should it be inlined
    equal_pair_harm_no_confirm: Fraction | None,
    equal_pair_harm_confirm: Fraction | None,
) -> RelativeGain | None:
    return absolute_risk_reduction(equal_pair_harm_no_confirm, equal_pair_harm_confirm)


def equal_pair_relative_risk_reduction( #TODO: what's the point of this? Does it need more code or better wiring? Analyze properly. Or should it be inlined
    equal_pair_harm_no_confirm: Fraction | None,
    equal_pair_risk_reduction: RelativeGain | None,
) -> RelativeGain | None:
    return relative_risk_reduction(equal_pair_harm_no_confirm, equal_pair_risk_reduction)


class EfficiencyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EfficiencyRecord:
    wall_time_seconds: ElapsedSeconds
    peak_host_rss_mib: MemoryMib
    peak_cuda_allocated_bytes: ByteCount
    packet_serialized_byte_count: ByteCount
    source_response_optimizer_steps: StepCount
    target_confirmation_optimizer_steps: StepCount
    live_assimilation_optimizer_steps: StepCount
    timeout_indicator: bool
    resource_limit_indicator: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds < 0.0:
            raise EfficiencyError("wall time must be finite and nonnegative")
        if not math.isfinite(self.peak_host_rss_mib) or self.peak_host_rss_mib < 0.0:
            raise EfficiencyError("peak host RSS must be finite and nonnegative")
        for name, value in (
            (EfficiencyMetricName.CUDA_BYTES, self.peak_cuda_allocated_bytes),
            (EfficiencyMetricName.PACKET_BYTES, self.packet_serialized_byte_count),
            (EfficiencyMetricName.SOURCE_RESPONSE_STEPS, self.source_response_optimizer_steps),
            (EfficiencyMetricName.CONFIRMATION_STEPS, self.target_confirmation_optimizer_steps),
            (EfficiencyMetricName.ASSIMILATION_STEPS, self.live_assimilation_optimizer_steps),
        ):
            if value < 0:
                raise EfficiencyError(f"{name.value} must be nonnegative")
