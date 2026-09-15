from __future__ import annotations

import math

import pytest

from fedorbit.analysis.metrics import (
    ClassEntropySet,
    ClassF1,
    ClassF1Set,
    ClassRecall,
    ClassRecallSet,
    ConfusionCounts,
    CrossEntropy,
    MetricComputationError,
    Probability,
    TrueClassProbabilities,
    balanced_accuracy,
    beneficial_rejected_rate,
    class_conditional_cross_entropy,
    confusion_counts,
    example_cross_entropy,
    f1_from_counts,
    macro_cross_entropy,
    macro_f1,
    precision_from_counts,
    recall_from_counts,
    relative_macro_ce_gain,
)
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.types import ClassIndex


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
        predicted_labels=(
            ClassIndex(1),
            ClassIndex(1),
            ClassIndex(0),
            ClassIndex(0),
            ClassIndex(1),
        ),
        true_labels=(
            ClassIndex(1),
            ClassIndex(0),
            ClassIndex(0),
            ClassIndex(1),
            ClassIndex(1),
        ),
        positive_class=ClassIndex(1),
    )
    assert counts == ConfusionCounts(true_positives=2, false_positives=1, false_negatives=1)


def test_macro_f1_and_balanced_accuracy_are_class_means(config: FedorbitConfig) -> None:
    del config
    assert macro_f1(ClassF1Set((ClassF1(0.8), ClassF1(0.6), ClassF1(1.0)))).value == pytest.approx(
        0.8
    )
    assert balanced_accuracy(
        ClassRecallSet((ClassRecall(1.0), ClassRecall(0.5), ClassRecall(0.0)))
    ).value == pytest.approx(0.5)
    with pytest.raises(MetricComputationError):
        ClassRecallSet(())


def test_beneficial_rejected_rate(config: FedorbitConfig) -> None:
    del config
    assert beneficial_rejected_rate(rejected_with_counterfactual_gain=2, proposed=8) == 0.25
    assert beneficial_rejected_rate(rejected_with_counterfactual_gain=0, proposed=0) is None


def test_relative_macro_ce_gain_matches_hand_computed_reference(
    config: FedorbitConfig,
) -> None:
    floor = config.scientific.metrics.relative_macro_ce_denominator_floor
    reference = CrossEntropy(0.8)
    improving = relative_macro_ce_gain(reference, CrossEntropy(0.6))
    assert improving.relative == pytest.approx((0.8 - 0.6) / max(0.8, floor))
    assert improving.absolute == pytest.approx(0.2)
    worsening = relative_macro_ce_gain(reference, CrossEntropy(1.0))
    assert worsening.relative == pytest.approx((0.8 - 1.0) / max(0.8, floor))
    assert worsening.relative is not None and worsening.relative < 0.0
    assert worsening.absolute == pytest.approx(-0.2)


def test_relative_macro_ce_gain_is_na_at_or_below_the_denominator_floor(
    config: FedorbitConfig,
) -> None:
    floor = config.scientific.metrics.relative_macro_ce_denominator_floor
    method = CrossEntropy(0.0)
    at_floor = relative_macro_ce_gain(CrossEntropy(floor), method)
    assert at_floor.relative == pytest.approx(floor / max(floor, floor))
    assert at_floor.absolute == pytest.approx(floor)
    below_floor = relative_macro_ce_gain(CrossEntropy(floor / 10.0), method)
    assert below_floor.relative is None
    assert below_floor.absolute == pytest.approx(floor / 10.0)


def test_example_and_class_conditional_cross_entropy_share_one_floor(
    config: FedorbitConfig,
) -> None:
    floor = config.scientific.metrics.probability_log_floor
    assert example_cross_entropy(Probability(0.5)).value == pytest.approx(-math.log(0.5))
    assert example_cross_entropy(Probability(0.0)).value == pytest.approx(-math.log(floor))
    class_value = class_conditional_cross_entropy(
        TrueClassProbabilities((Probability(0.5), Probability(0.0)))
    )
    assert class_value.value == pytest.approx((-math.log(0.5) - math.log(floor)) / 2)
