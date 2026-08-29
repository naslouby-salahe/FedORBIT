from __future__ import annotations

import itertools
import statistics
from collections import defaultdict
from dataclasses import dataclass

import torch

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import DatasetId, RngNamespace
from fedorbit.models.host_classifier import HostClassifier
from fedorbit.models.network_classifier import NetworkFlowClassifier
from fedorbit.runtime.seeds import derive_seed32
from fedorbit.training.losses import ClassWeights
from fedorbit.training.trainer import SelectedHyperparameters, TrainingOutcome, train_base_model

REFERENCE_LEARNING_RATE = 1.0e-3
NETWORK_DATASETS = frozenset({DatasetId.EDGE_IIOTSET_NETWORK, DatasetId.TON_IOT_NETWORK})
HOST_DATASETS = frozenset({DatasetId.TON_IOT_WINDOWS10_HOST, DatasetId.TON_IOT_LINUX_PROCESS_HOST})


class PilotError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PilotConfiguration:
    learning_rate: float
    weight_decay: float
    dropout: float

    def hyperparameters(self) -> SelectedHyperparameters:
        return SelectedHyperparameters(self.learning_rate, self.weight_decay, self.dropout)


@dataclass(frozen=True, slots=True)
class PilotFitResult:
    configuration: PilotConfiguration
    seed: int
    outcome: TrainingOutcome


@dataclass(frozen=True, slots=True)
class PilotSelection:
    configuration: PilotConfiguration
    median_valid_macro_cross_entropy: float
    valid_macro_cross_entropy_standard_deviation: float


@dataclass(frozen=True, slots=True)
class PilotData:
    train_features: torch.Tensor
    train_targets: torch.Tensor
    valid_features: torch.Tensor
    valid_targets: torch.Tensor
    n_classes: int


def pilot_grid(config: FedorbitConfig) -> tuple[PilotConfiguration, ...]:
    pilot = config.scientific.base_model_pilot
    configurations = tuple(
        PilotConfiguration(learning_rate, weight_decay, dropout)
        for learning_rate, weight_decay, dropout in itertools.product(
            pilot.learning_rates,
            pilot.weight_decays,
            pilot.dropouts,
        )
    )
    if len(configurations) != 12:
        raise PilotError("base-model pilot grid must contain exactly 12 configurations")
    return configurations


def run_base_model_pilot(
    config: FedorbitConfig,
    data: PilotData,
    dataset: DatasetId,
) -> tuple[PilotFitResult, ...]:
    seeds = config.scientific.randomness.pilot_seeds
    if len(seeds) != 3:
        raise PilotError("base-model pilot requires exactly three pilot seeds")
    class_weights = ClassWeights.from_targets(data.train_targets, data.n_classes)
    results: list[PilotFitResult] = []
    for candidate in pilot_grid(config):
        for seed in seeds:
            model = create_classifier(
                dataset,
                data.train_features.shape[1],
                data.n_classes,
                candidate.dropout,
                seed,
            )
            outcome = train_base_model(
                config,
                model,
                data.train_features,
                data.train_targets,
                data.valid_features,
                data.valid_targets,
                class_weights,
                seed,
                candidate.hyperparameters(),
            )
            results.append(PilotFitResult(candidate, seed, outcome))
    if len(results) != 36:
        raise PilotError("base-model pilot must produce exactly 36 fits per client")
    return tuple(results)


def select_pilot_configuration(results: tuple[PilotFitResult, ...]) -> PilotSelection:
    grouped: defaultdict[PilotConfiguration, list[float]] = defaultdict(list)
    for result in results:
        grouped.setdefault(result.configuration, []).append(
            result.outcome.valid_macro_cross_entropy
        )
    candidates: list[PilotSelection] = []
    for configuration, values in grouped.items():
        if len(values) != 3:
            raise PilotError("every pilot configuration must have exactly three seed results")
        candidates.append(
            PilotSelection(
                configuration,
                statistics.median(values),
                statistics.pstdev(values),
            )
        )
    if len(candidates) != 12:
        raise PilotError("pilot selection requires all 12 registered configurations")
    return min(
        candidates,
        key=lambda item: (
            item.median_valid_macro_cross_entropy,
            item.valid_macro_cross_entropy_standard_deviation,
            abs(item.configuration.learning_rate - REFERENCE_LEARNING_RATE),
            item.configuration.weight_decay,
            item.configuration.dropout,
        ),
    )


def create_classifier(
    dataset: DatasetId,
    input_dimension: int,
    n_classes: int,
    dropout_probability: float,
    seed: int,
) -> NetworkFlowClassifier | HostClassifier:
    initialization_seed = derive_seed32(
        seed,
        RngNamespace.MODEL_INITIALIZATION,
        {"dataset": dataset.value, "input_dimension": input_dimension, "n_classes": n_classes},
    )
    generator = torch.Generator().manual_seed(initialization_seed)
    if dataset in NETWORK_DATASETS:
        model: NetworkFlowClassifier | HostClassifier = NetworkFlowClassifier(
            input_dimension, n_classes, dropout_probability
        )
    elif dataset in HOST_DATASETS:
        model = HostClassifier(input_dimension, n_classes, dropout_probability)
    else:
        raise PilotError(f"no classifier architecture registered for {dataset.value}")
    model.initialize(generator)
    return model
