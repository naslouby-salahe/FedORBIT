from __future__ import annotations

import math

import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.evaluation import (
    ClassEntropySet,
    ConfusionCounts,
    CrossEntropy,
    MetricComputationError,
    Probability,
    ProposalOutcomeTally,
    TrueClassProbabilities,
    absolute_objective_error,
    absolute_risk_reduction,
    balanced_accuracy,
    beneficial_rejected_rate,
    class_conditional_cross_entropy,
    confirmation_coverage,
    confusion_counts,
    coverage_loss,
    equal_pair_absolute_risk_reduction,
    equal_pair_mean,
    equal_pair_relative_risk_reduction,
    f1_from_counts,
    harm_indicator,
    macro_cross_entropy,
    macro_f1,
    no_confirmation_coverage,
    pair_mean,
    precision_from_counts,
    proposal_rates,
    recall_from_counts,
    relative_macro_ce_gain,
    relative_objective_error,
    seed_harm_rate,
)


@pytest.fixture
def config() -> FedorbitConfig:
    return load_fedorbit_config()


def test_class_conditional_entropy_matches_hand_computation(
    config: FedorbitConfig,
) -> None:
    log_floor = config.scientific.metrics.probability_log_floor
    probabilities = (0.5, 0.25, 1.0)
    expected = (-math.log(0.5) - math.log(0.25) - math.log(1.0)) / 3
    value = class_conditional_cross_entropy(
        TrueClassProbabilities(tuple(Probability(probability) for probability in probabilities))
    )
    assert value.value == pytest.approx(expected)
    clamped = class_conditional_cross_entropy(TrueClassProbabilities((Probability(0.0),)))
    assert clamped.value == pytest.approx(-math.log(log_floor))


def test_zero_examples_in_class_is_invalid_data(config: FedorbitConfig) -> None:
    del config
    with pytest.raises(MetricComputationError):
        TrueClassProbabilities(())
    with pytest.raises(MetricComputationError):
        Probability(1.2)


def test_macro_entropy_is_arithmetic_mean(config: FedorbitConfig) -> None:
    del config
    assert macro_cross_entropy(ClassEntropySet((CrossEntropy(1.0), CrossEntropy(3.0)))).value == 2.0
    with pytest.raises(MetricComputationError):
        ClassEntropySet(())


def test_relative_gain_formula_and_na_semantics() -> None:
    gain = relative_macro_ce_gain(reference_macro_ce=2.0, method_macro_ce=1.5)
    assert not gain.is_na
    assert gain.value == pytest.approx(0.25)
    assert gain.absolute_difference == pytest.approx(0.5)

    tiny_reference = relative_macro_ce_gain(reference_macro_ce=1e-15, method_macro_ce=5e-16)
    assert tiny_reference.is_na
    assert tiny_reference.value is None
    assert tiny_reference.absolute_difference == pytest.approx(5e-16)


def test_precision_recall_f1_with_zero_denominator_rule(config: FedorbitConfig) -> None:
    del config
    assert precision_from_counts(3, 1) == pytest.approx(0.75)
    assert recall_from_counts(3, 1) == pytest.approx(0.75)
    f1 = f1_from_counts(3, 1, 1)
    expected = 2 * 0.75 * 0.75 / 1.5
    assert f1 == pytest.approx(expected)
    assert precision_from_counts(0, 0) == 0.0
    assert recall_from_counts(0, 5) == 0.0
    assert f1_from_counts(0, 0, 0) == 0.0


def test_confusion_counts_partition_predictions(config: FedorbitConfig) -> None:
    del config
    counts = confusion_counts(
        predicted_labels=(1, 1, 0, 0, 1),
        true_labels=(1, 0, 0, 1, 1),
        positive_class=1,
    )
    assert counts == ConfusionCounts(true_positives=2, false_positives=1, false_negatives=1)


def test_macro_f1_and_balanced_accuracy_are_class_means(config: FedorbitConfig) -> None:
    del config
    assert macro_f1((0.8, 0.6, 1.0)) == pytest.approx(0.8)
    assert balanced_accuracy((1.0, 0.5, 0.0)) == pytest.approx(0.5)
    with pytest.raises(MetricComputationError):
        balanced_accuracy(())


def test_absolute_and_relative_solver_errors() -> None:
    cfg = load_fedorbit_config()
    assert absolute_objective_error(1.2, 1.0) == pytest.approx(0.2)
    error = relative_objective_error(objective_value=1.05, truth_value=1.0)
    assert error == pytest.approx(0.05)
    degenerate = relative_objective_error(objective_value=1.0, truth_value=0.0)
    floor = cfg.scientific.metrics.relative_solver_error_denominator_floor
    assert degenerate == pytest.approx(1.0 / max(abs(0.0), floor))


def test_proposal_rates_and_full_na_when_no_proposals(config: FedorbitConfig) -> None:
    del config
    tally = ProposalOutcomeTally(proposed=10, accepted=4, harmful_accepted=1, useful_accepted=2)
    rates = proposal_rates(tally)
    assert rates.acceptance_rate == pytest.approx(0.4)
    assert rates.harmful_accepted_rate == pytest.approx(0.1)
    assert rates.useful_accepted_rate == pytest.approx(0.2)
    empty = proposal_rates(ProposalOutcomeTally(0, 0, 0, 0))
    assert empty.all_na
    with pytest.raises(MetricComputationError):
        ProposalOutcomeTally(proposed=2, accepted=3, harmful_accepted=0, useful_accepted=0)


def test_confirmation_coverage_and_loss(config: FedorbitConfig) -> None:
    del config
    assert confirmation_coverage(7, 10) == pytest.approx(0.7)
    assert confirmation_coverage(0, 0) is None
    assert no_confirmation_coverage(10) == 1.0
    assert no_confirmation_coverage(0) is None
    loss = coverage_loss(coverage_no_confirm=1.0, coverage_confirm=0.7)
    assert loss == pytest.approx(0.3)
    assert coverage_loss(None, 0.5) is None


def test_harm_indicators_use_configured_thresholds() -> None:
    assert harm_indicator(-0.01, -0.01)
    assert not harm_indicator(0.0, -0.01)
    rate = seed_harm_rate((-0.01, 0.5, -0.02), harmful_threshold=-0.01)
    assert rate == pytest.approx(2 / 3)
    assert seed_harm_rate((), -0.01) is None


def test_arr_rrr_definitions_and_na_rules() -> None:
    from fedorbit.evaluation import relative_risk_reduction

    arr = absolute_risk_reduction(harm_rate_no_confirm=0.5, harm_rate_confirm=0.2)
    assert arr == pytest.approx(0.3)
    rrr_value = relative_risk_reduction(harm_rate_no_confirm=0.5, risk_reduction=arr)
    assert rrr_value == pytest.approx(0.6)
    zero_base = relative_risk_reduction(harm_rate_no_confirm=0.0, risk_reduction=None)
    assert zero_base is None


def test_beneficial_rejected_rate(config: FedorbitConfig) -> None:
    del config
    assert beneficial_rejected_rate(rejected_with_counterfactual_gain=2, proposed=8) == 0.25
    assert beneficial_rejected_rate(rejected_with_counterfactual_gain=0, proposed=0) is None


def test_pair_and_equal_pair_aggregation(config: FedorbitConfig) -> None:
    del config
    assert pair_mean((0.2, 0.4, 0.6)) == pytest.approx(0.4)
    assert pair_mean(()) is None
    equal = equal_pair_mean((0.2, None, 0.4, 0.6))
    assert equal == pytest.approx(0.4)
    assert equal_pair_mean((None, None)) is None
    eq_arr = equal_pair_absolute_risk_reduction(0.5, 0.2)
    assert eq_arr == pytest.approx(0.3)
    eq_rrr = equal_pair_relative_risk_reduction(
        equal_pair_harm_no_confirm=0.5, equal_pair_risk_reduction=0.3
    )
    assert eq_rrr == pytest.approx(0.6)
    assert equal_pair_relative_risk_reduction(0.0, None) is None
