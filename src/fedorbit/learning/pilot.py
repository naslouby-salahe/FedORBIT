from __future__ import annotations

import itertools
import statistics
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import cast

import torch

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.runtime import (
    RandomSeed,
    SeedDerivationRequest,
    derive_seed32,
    principal_determinism,
)
from fedorbit.learning.models import HostClassifier, NetworkFlowClassifier
from fedorbit.learning.training import (
    ClassWeights,
    SelectedHyperparameters,
    TrainingOutcome,
    train_base_model,
)
from fedorbit.types import (
    Coefficient,
    ConceptCount,
    DatasetId,
    FeatureCount,
    Fraction,
    LearningRate,
    RngNamespace,
    Score,
    StableJsonPayload,
    StandardError,
    WeightDecay,
)

NETWORK_DATASETS = frozenset({DatasetId.EDGE_IIOTSET_NETWORK, DatasetId.TON_IOT_NETWORK})
HOST_DATASETS = frozenset({DatasetId.TON_IOT_WINDOWS10_HOST, DatasetId.TON_IOT_LINUX_PROCESS_HOST})


class PilotError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PilotConfiguration:
    learning_rate: LearningRate
    weight_decay: WeightDecay
    dropout: Fraction

    def hyperparameters(self) -> SelectedHyperparameters:
        return SelectedHyperparameters(self.learning_rate, self.weight_decay, self.dropout)


@dataclass(frozen=True, slots=True)
class PilotFitResult:
    configuration: PilotConfiguration
    seed: RandomSeed
    outcome: TrainingOutcome


@dataclass(frozen=True, slots=True)
class PilotSelection:
    configuration: PilotConfiguration
    median_valid_macro_cross_entropy: Score
    valid_macro_cross_entropy_standard_deviation: StandardError


@dataclass(frozen=True, slots=True)
class PilotData:
    train_features: torch.Tensor
    train_targets: torch.Tensor
    valid_features: torch.Tensor
    valid_targets: torch.Tensor
    n_classes: ConceptCount


def pilot_grid() -> tuple[PilotConfiguration, ...]:
    pilot = active_config().scientific.base_model_pilot
    configurations = tuple(
        PilotConfiguration(learning_rate, weight_decay, dropout)
        for learning_rate, weight_decay, dropout in itertools.product(
            pilot.learning_rates,
            pilot.weight_decays,
            pilot.dropouts,
        )
    )
    return configurations


def run_base_model_pilot(
    data: PilotData,
    dataset: DatasetId,
    device: torch.device | None = None,
) -> tuple[PilotFitResult, ...]:
    seeds = active_config().scientific.randomness.pilot_seeds
    class_weights = ClassWeights.from_targets(data.train_targets, data.n_classes)
    results: list[PilotFitResult] = []
    with principal_determinism():
        for candidate in pilot_grid():
            for seed in seeds:
                model = create_classifier(
                    dataset,
                    data.train_features.shape[1],
                    data.n_classes,
                    candidate.dropout,
                    seed,
                    device,
                )
                outcome = train_base_model(
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
    return tuple(results)


def select_pilot_configuration(results: tuple[PilotFitResult, ...]) -> PilotSelection:
    registered_seed_count = len(active_config().scientific.randomness.pilot_seeds)
    registered_configuration_count = len(pilot_grid())
    grouped: defaultdict[PilotConfiguration, list[Score]] = defaultdict(list)
    for result in results:
        grouped.setdefault(result.configuration, []).append(
            result.outcome.valid_macro_cross_entropy
        )
    candidates: list[PilotSelection] = []
    for configuration, values in grouped.items():
        if len(values) != registered_seed_count:
            raise PilotError("every pilot configuration must have one result per pilot seed")
        candidates.append(
            PilotSelection(
                configuration,
                statistics.median(values),
                statistics.pstdev(values),
            )
        )
    if len(candidates) != registered_configuration_count:
        raise PilotError("pilot selection requires every registered pilot configuration")
    return min(candidates, key=_pilot_selection_order)


@dataclass(frozen=True, slots=True, order=True)
class PilotSelectionOrder:
    median_valid_macro_cross_entropy: Score
    valid_macro_cross_entropy_standard_deviation: StandardError
    learning_rate_distance: Coefficient
    weight_decay: WeightDecay
    dropout: Fraction


def _pilot_selection_order(item: PilotSelection) -> PilotSelectionOrder:
    reference_learning_rate = active_config().scientific.base_model_pilot.reference_learning_rate
    return PilotSelectionOrder(
        item.median_valid_macro_cross_entropy,
        item.valid_macro_cross_entropy_standard_deviation,
        abs(item.configuration.learning_rate - reference_learning_rate),
        item.configuration.weight_decay,
        item.configuration.dropout,
    )


def create_classifier(
    dataset: DatasetId,
    input_dimension: FeatureCount,
    n_classes: ConceptCount,
    dropout_probability: Fraction,
    seed: RandomSeed,
    device: torch.device | None = None,
) -> NetworkFlowClassifier | HostClassifier:
    initialization_seed = derive_seed32(
        SeedDerivationRequest(
            seed,
            RngNamespace.MODEL_INITIALIZATION,
            cast(
                StableJsonPayload,
                OrderedDict(
                    dataset=dataset.value,
                    input_dimension=input_dimension,
                    n_classes=n_classes,
                ),
            ),
        )
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
    if device is not None:
        model = model.to(device=device)
    return model
