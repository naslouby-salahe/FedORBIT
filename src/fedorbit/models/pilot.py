from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import torch

from fedorbit.config.models import FedorbitConfig
from fedorbit.models.architectures import HostClassifier, NetworkFlowClassifier
from fedorbit.models.class_weights import ClassWeights
from fedorbit.models.training import TrainingOutcome, train_base_model

PILOT_SELECTION_REFERENCE_LEARNING_RATE = 1e-3


@dataclass(frozen=True, slots=True)
class PilotConfiguration:
    learning_rate: float
    weight_decay: float
    dropout_probability: float


@dataclass(frozen=True, slots=True)
class PilotFitResult:
    configuration: PilotConfiguration
    pilot_seed: int
    valid_macro_cross_entropy: float


@dataclass(frozen=True, slots=True)
class PilotSelection:
    configuration: PilotConfiguration
    median_valid_macro_cross_entropy: float
    std_dev_valid_macro_cross_entropy: float
    fitted_configurations: int


@dataclass(frozen=True, slots=True)
class PilotData:
    train_features: torch.Tensor
    train_targets: torch.Tensor
    valid_features: torch.Tensor
    valid_targets: torch.Tensor
    train_counts: tuple[int, ...]
    n_classes: int
    input_dim: int


class PilotError(ValueError):
    pass


def pilot_grid(config: FedorbitConfig) -> tuple[PilotConfiguration, ...]:
    pilot = config.scientific.base_model_pilot
    return tuple(
        PilotConfiguration(learning_rate, weight_decay, dropout_probability)
        for learning_rate, weight_decay, dropout_probability in product(
            pilot.learning_rates, pilot.weight_decays, pilot.dropouts
        )
    )


def run_base_model_pilot(
    config: FedorbitConfig,
    data: PilotData,
    modality: str,
) -> PilotSelection:
    grid = pilot_grid(config)
    pilot_seeds = config.scientific.randomness.pilot_seeds
    weights = ClassWeights.from_train_counts(data.train_counts)
    per_class_tensor = torch.tensor(weights.per_class)
    fits: list[PilotFitResult] = []
    for configuration in grid:
        for pilot_seed in pilot_seeds:
            if modality == "network":
                model = NetworkFlowClassifier(
                    data.input_dim,
                    data.n_classes,
                    dropout_probability=configuration.dropout_probability,
                )
            else:
                model = HostClassifier(
                    data.input_dim,
                    data.n_classes,
                    dropout_probability=configuration.dropout_probability,
                )
            outcome: TrainingOutcome = train_base_model(
                config,
                model,
                data.train_features,
                data.train_targets,
                data.valid_features,
                data.valid_targets,
                per_class_tensor,
                pilot_seed,
                configuration.learning_rate,
                configuration.weight_decay,
            )
            fits.append(
                PilotFitResult(
                    configuration,
                    pilot_seed,
                    outcome.valid_macro_cross_entropy,
                )
            )
    return select_pilot_configuration(grid, tuple(fits))


def select_pilot_configuration(
    grid: tuple[PilotConfiguration, ...],
    fits: tuple[PilotFitResult, ...],
) -> PilotSelection:
    by_configuration: dict[PilotConfiguration, tuple[float, float]] = {}
    for configuration in grid:
        values = tuple(
            fit.valid_macro_cross_entropy for fit in fits if fit.configuration == configuration
        )
        if not values:
            continue
        by_configuration[configuration] = (median(values), std_dev(values))
    if not by_configuration:
        raise PilotError("no pilot configuration produced valid fits")
    ordered = tuple(
        sorted(
            by_configuration.items(),
            key=lambda item: (
                item[1][0],
                item[1][1],
                abs(item[0].learning_rate - PILOT_SELECTION_REFERENCE_LEARNING_RATE),
                item[0].weight_decay,
                item[0].dropout_probability,
            ),
        )
    )
    selected, (median_metric, std_dev_metric) = ordered[0]
    return PilotSelection(selected, median_metric, std_dev_metric, len(fits))


def median(values: tuple[float, ...]) -> float:
    ordered = tuple(sorted(values))
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def std_dev(values: tuple[float, ...]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)
