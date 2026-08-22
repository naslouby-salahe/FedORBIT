from __future__ import annotations

import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.models.class_weights import ClassWeights, ClassWeightsError
from fedorbit.models.pilot import (
    PILOT_SELECTION_REFERENCE_LEARNING_RATE,
    PilotConfiguration,
    PilotError,
    PilotFitResult,
    median,
    pilot_grid,
    select_pilot_configuration,
    std_dev,
)

GRID = (
    PilotConfiguration(0.001, 0.0, 0.0),
    PilotConfiguration(0.001, 0.0, 0.1),
    PilotConfiguration(0.001, 0.0001, 0.0),
    PilotConfiguration(0.001, 0.0001, 0.1),
    PilotConfiguration(0.0003, 0.0, 0.0),
    PilotConfiguration(0.0003, 0.0, 0.1),
    PilotConfiguration(0.0003, 0.0001, 0.0),
    PilotConfiguration(0.0003, 0.0001, 0.1),
    PilotConfiguration(0.003, 0.0, 0.0),
    PilotConfiguration(0.003, 0.0, 0.1),
    PilotConfiguration(0.003, 0.0001, 0.0),
    PilotConfiguration(0.003, 0.0001, 0.1),
)


def test_pilot_grid_is_cartesian_product() -> None:
    config = load_fedorbit_config()
    grid = pilot_grid(config)
    assert len(grid) == 12
    assert set(grid) == set(GRID)


def test_pilot_seed_list_only_from_config() -> None:
    config = load_fedorbit_config()
    assert tuple(config.scientific.randomness.pilot_seeds) == (101, 202, 303)


def test_selection_orders_by_median_then_std_then_lr_then_decay_then_dropout() -> None:
    lower_median = PilotConfiguration(0.001, 0.0, 0.0)
    higher_median = PilotConfiguration(0.003, 0.0001, 0.1)
    fits = (
        _fit(lower_median, 1.0),
        _fit(lower_median, 1.0),
        _fit(lower_median, 1.0),
        _fit(higher_median, 0.5),
        _fit(higher_median, 0.5),
        _fit(higher_median, 0.5),
    )
    selection = select_pilot_configuration(GRID, list(fits))
    assert selection.configuration == higher_median


def test_std_dev_breaks_median_ties() -> None:
    wide = PilotConfiguration(0.001, 0.0, 0.1)
    narrow = PilotConfiguration(0.001, 0.0, 0.0)
    fits = (
        _fit(wide, 0.8),
        _fit(wide, 1.0),
        _fit(wide, 1.2),
        _fit(narrow, 0.9),
        _fit(narrow, 1.0),
        _fit(narrow, 1.1),
    )
    selection = select_pilot_configuration(GRID, list(fits))
    assert selection.configuration == narrow


def test_learning_rate_tie_break_third() -> None:
    close_lr = PilotConfiguration(0.001, 0.0001, 0.1)
    far_lr = PilotConfiguration(0.003, 0.0, 0.0)
    fits = (
        _fit(close_lr, 1.0),
        _fit(close_lr, 1.0),
        _fit(close_lr, 1.0),
        _fit(far_lr, 1.0),
        _fit(far_lr, 1.0),
        _fit(far_lr, 1.0),
    )
    selection = select_pilot_configuration(GRID, list(fits))
    assert selection.configuration == close_lr


def test_reference_learning_rate_is_grid_center() -> None:
    assert PILOT_SELECTION_REFERENCE_LEARNING_RATE == 1e-3


def test_median_even_and_odd() -> None:
    assert median((3.0, 1.0, 2.0)) == 2.0
    assert median((4.0, 1.0, 2.0, 3.0)) == 2.5


def test_std_dev_value() -> None:
    assert std_dev((2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)) == pytest.approx(2.0)


def test_selection_raises_without_fits() -> None:
    with pytest.raises(PilotError):
        select_pilot_configuration(GRID, [])


def test_class_weights_inverse_frequency_formula() -> None:
    weights = ClassWeights.from_train_counts((60, 30, 10))
    total = 100
    n_classes = 3
    raw_expected = tuple(total / (n_classes * count) for count in (60, 30, 10))
    example_weighted_mean = (
        sum(count * raw for count, raw in zip((60, 30, 10), raw_expected, strict=True)) / total
    )
    expected = tuple(raw / example_weighted_mean for raw in raw_expected)
    assert weights.per_class == pytest.approx(expected)
    assert sum(
        count * weight for count, weight in zip((60, 30, 10), weights.per_class, strict=True)
    ) / total == pytest.approx(1.0)


def test_class_weights_normalize_by_example_weighted_mean() -> None:
    weights = ClassWeights.from_train_counts((50, 50))
    assert weights.per_class == pytest.approx((1.0, 1.0))


def test_class_weights_reject_empty_or_zero_counts() -> None:
    with pytest.raises(ClassWeightsError):
        ClassWeights.from_train_counts((0, 1))
    with pytest.raises(ClassWeightsError):
        ClassWeights.from_train_counts(())


def test_per_example_multiplier_without_renormalization() -> None:
    weights = ClassWeights.from_train_counts((60, 30, 10))
    base = weights.per_class[0]
    assert weights.per_example(0, intervention_multiplier=2.0) == pytest.approx(base * 2.0)
    assert weights.per_example(0) == pytest.approx(base)


def _fit(configuration: PilotConfiguration, valid: float) -> PilotFitResult:
    return PilotFitResult(configuration, 101, valid)
