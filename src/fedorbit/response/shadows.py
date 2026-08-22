from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from fedorbit.config.models import FedorbitConfig
from fedorbit.models.training import OptimizerState
from fedorbit.response.risk import equal_native_class_risk
from fedorbit.runtime.seeds import RngNamespace, derive_seed32


class ShadowError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ShadowData:
    train_features: torch.Tensor
    train_targets: torch.Tensor
    meta_features: torch.Tensor
    meta_targets: torch.Tensor
    intervention_classes: tuple[int, ...]
    outcome_native_class_sets: tuple[tuple[int, ...], ...]
    base_class_weights: torch.Tensor


@dataclass(frozen=True, slots=True)
class ShadowSettings:
    epsilon: float
    horizon: int
    learning_rate: float
    weight_decay: float


def shadow_batch_schedule(
    train_size: int,
    batch_size: int,
    rng: torch.Generator,
) -> Iterator[torch.Tensor]:
    if train_size <= 0:
        raise ShadowError("shadow TRAIN set is empty")
    if batch_size <= 0:
        raise ShadowError("shadow batch size must be positive")
    permutation = torch.randperm(train_size, generator=rng)
    position = 0
    while True:
        if position >= train_size:
            permutation = torch.randperm(train_size, generator=rng)
            position = 0
        batch = permutation[position : position + batch_size]
        position += batch_size
        yield batch


def paired_shadow_derivative(
    positive_risk: float,
    negative_risk: float,
    baseline_risk: float,
    epsilon: float,
    denominator_floor: float,
) -> float:
    if epsilon <= 0:
        raise ShadowError("intervention magnitude must be positive")
    return (negative_risk - positive_risk) / (2 * epsilon * max(baseline_risk, denominator_floor))


def run_shadow_pair(
    config: FedorbitConfig,
    model: torch.nn.Module,
    base_state_dict: dict[str, torch.Tensor],
    base_optimizer_state: OptimizerState,
    base_rng_state: torch.Tensor,
    data: ShadowData,
    settings: ShadowSettings,
    seed: int,
) -> tuple[tuple[float, float, float], ...]:
    training = config.scientific.training
    batch_size = training.batch_size
    schedule_rng = torch.Generator().manual_seed(
        derive_seed32(seed, RngNamespace.RESPONSE_SCHEDULE, "shadow-pair")
    )
    positive = _run_shadow(
        config,
        model,
        base_state_dict,
        base_optimizer_state,
        base_rng_state,
        data,
        settings,
        1.0 + settings.epsilon,
        batch_size,
        schedule_rng,
    )
    negative = _run_shadow(
        config,
        model,
        base_state_dict,
        base_optimizer_state,
        base_rng_state,
        data,
        settings,
        1.0 - settings.epsilon,
        batch_size,
        schedule_rng,
    )
    model.load_state_dict(base_state_dict)
    baseline = _evaluate_risks(
        config,
        model,
        data.meta_features,
        data.meta_targets,
        data.outcome_native_class_sets,
    )
    return tuple(
        (positive[outcome], negative[outcome], baseline[outcome])
        for outcome in range(len(data.outcome_native_class_sets))
    )


def _run_shadow(
    config: FedorbitConfig,
    model: torch.nn.Module,
    base_state_dict: dict[str, torch.Tensor],
    base_optimizer_state: OptimizerState,
    base_rng_state: torch.Tensor,
    data: ShadowData,
    settings: ShadowSettings,
    multiplier: float,
    batch_size: int,
    schedule_rng: torch.Generator,
) -> tuple[float, ...]:
    model.load_state_dict(base_state_dict)
    torch.set_rng_state(base_rng_state)
    training = config.scientific.training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        betas=(training.adamw.beta1, training.adamw.beta2),
        eps=training.adamw.epsilon,
        weight_decay=settings.weight_decay,
    )
    optimizer.load_state_dict(base_optimizer_state)
    model.train()
    steps = 0
    schedule = shadow_batch_schedule(data.train_features.shape[0], batch_size, schedule_rng)
    for batch in schedule:
        if steps >= settings.horizon:
            break
        optimizer.zero_grad()
        logits = model(data.train_features[batch].float())
        per_example_ce = _shadow_ce(logits, data.train_targets[batch], config)
        weights = _shadow_weights(
            data.base_class_weights,
            data.train_targets[batch],
            data.intervention_classes,
            multiplier,
        )
        loss = (per_example_ce * weights).mean()
        loss.backward()
        optimizer.step()
        steps += 1
    if steps == 0:
        raise ShadowError("shadow consumed no optimizer steps")
    model.eval()
    with torch.no_grad():
        shadow_logits = model(data.meta_features.float())
    return tuple(
        equal_native_class_risk(
            shadow_logits,
            data.meta_targets,
            class_set,
            config.scientific.metrics.probability_log_floor,
        )
        for class_set in data.outcome_native_class_sets
    )


def _shadow_ce(logits: torch.Tensor, targets: torch.Tensor, config: FedorbitConfig) -> torch.Tensor:
    probability_log_floor = config.scientific.metrics.probability_log_floor
    log_probabilities = torch.log_softmax(logits, dim=1)
    per_example = -log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    return torch.clamp(per_example, max=-math.log(probability_log_floor))


def _shadow_weights(
    base_class_weights: torch.Tensor,
    targets: torch.Tensor,
    intervention_classes: tuple[int, ...],
    multiplier: float,
) -> torch.Tensor:
    weights = base_class_weights[targets]
    for class_index in intervention_classes:
        weights = torch.where(targets == class_index, weights * multiplier, weights)
    return weights


def _evaluate_risks(
    config: FedorbitConfig,
    model: torch.nn.Module,
    meta_features: torch.Tensor,
    meta_targets: torch.Tensor,
    outcome_native_class_sets: tuple[tuple[int, ...], ...],
) -> tuple[float, ...]:
    model.eval()
    with torch.no_grad():
        logits = model(meta_features.float())
    return tuple(
        equal_native_class_risk(
            logits,
            meta_targets,
            class_set,
            config.scientific.metrics.probability_log_floor,
        )
        for class_set in outcome_native_class_sets
    )
