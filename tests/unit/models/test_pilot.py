from __future__ import annotations

import pytest
import torch

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.training.losses import ClassWeights
from fedorbit.training.pilot import (
    REFERENCE_LEARNING_RATE,
    PilotConfiguration,
    PilotFitResult,
    pilot_grid,
    select_pilot_configuration,
)
from fedorbit.training.trainer import (
    BaseCheckpoint,
    ModelParameterState,
    OptimizerState,
    RngState,
    SelectedHyperparameters,
    TrainingOutcome,
)


def _outcome(configuration: PilotConfiguration, metric: float) -> TrainingOutcome:
    checkpoint = BaseCheckpoint(
        epoch=0,
        valid_macro_cross_entropy=metric,
        state_dict=ModelParameterState(()),
        optimizer_state=OptimizerState(b""),
        rng_state=RngState(torch.get_rng_state().clone(), ()),
        selected_hyperparameters=configuration.hyperparameters(),
        train_class_weights=ClassWeights(torch.ones(2)),
    )
    return TrainingOutcome(checkpoint, 1)


def _fits(configuration: PilotConfiguration, values: tuple[float, float, float]) -> tuple[PilotFitResult, ...]:
    return tuple(
        PilotFitResult(configuration, seed, _outcome(configuration, value))
        for seed, value in zip((101, 202, 303), values, strict=True)
    )


def test_pilot_grid_is_exact_registered_cartesian_product() -> None:
    config = load_fedorbit_config()
    grid = pilot_grid(config)
    assert len(grid) == 12
    assert len(set(grid)) == 12
    assert {item.learning_rate for item in grid} == {0.0003, 0.001, 0.003}
    assert {item.weight_decay for item in grid} == {0.0, 0.0001}
    assert {item.dropout for item in grid} == {0.0, 0.1}
    assert tuple(config.scientific.randomness.pilot_seeds) == (101, 202, 303)


def test_pilot_selection_uses_registered_tie_order() -> None:
    best = PilotConfiguration(REFERENCE_LEARNING_RATE, 0.0, 0.0)
    wider = PilotConfiguration(REFERENCE_LEARNING_RATE, 0.0, 0.1)
    farther = PilotConfiguration(0.003, 0.0, 0.0)
    results = (
        *_fits(wider, (0.8, 1.0, 1.2)),
        *_fits(best, (0.9, 1.0, 1.1)),
        *_fits(farther, (1.0, 1.0, 1.0)),
    )
    selection = select_pilot_configuration(results)
    assert selection.configuration == best
    assert selection.median_valid_macro_cross_entropy == pytest.approx(1.0)


def test_pilot_selection_rejects_incomplete_grid() -> None:
    candidate = PilotConfiguration(REFERENCE_LEARNING_RATE, 0.0, 0.0)
    with pytest.raises(ValueError):
        select_pilot_configuration(_fits(candidate, (1.0, 1.0, 1.0)))
