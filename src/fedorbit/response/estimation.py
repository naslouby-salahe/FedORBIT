from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from fedorbit.config.loading import active_config
from fedorbit.learning.training import (
    ClassWeights,
    ModelParameterState,
    OptimizerState,
    RngState,
    backward_value,
    make_adamw,
    optimizer_step,
)


class ResponseEstimationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ShadowData:
    train_features: torch.Tensor
    train_targets: torch.Tensor
    meta_features: torch.Tensor
    meta_targets: torch.Tensor
    intervention_classes: tuple[int, ...]
    outcome_native_class_sets: tuple[tuple[int, ...], ...]
    base_class_weights: ClassWeights


@dataclass(frozen=True, slots=True)
class ShadowSettings:
    epsilon: float
    horizon: int
    learning_rate: float
    weight_decay: float


def native_class_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_index: int,
    probability_log_floor: float,
) -> float:
    class_examples = targets == class_index
    if not bool(class_examples.any()):
        return math.nan
    probabilities = torch.softmax(logits.to(dtype=torch.float32), dim=1)
    selected = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    per_example = -torch.log(torch.clamp(selected, min=probability_log_floor)).to(
        dtype=torch.float64
    )
    return float(per_example[class_examples].mean())


def equal_native_class_risk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    native_classes: tuple[int, ...],
    probability_log_floor: float,
) -> float:
    risks = tuple(
        native_class_cross_entropy(logits, targets, class_index, probability_log_floor)
        for class_index in native_classes
    )
    if not risks or any(math.isnan(risk) for risk in risks):
        return math.nan
    return sum(risks) / len(risks)


def shadow_batch_schedule(
    train_size: int,
    batch_size: int,
    rng: torch.Generator,
) -> Iterator[torch.Tensor]:
    if train_size <= 0:
        raise ResponseEstimationError("shadow TRAIN set is empty")
    if batch_size <= 0:
        raise ResponseEstimationError("shadow batch size must be positive")
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
    if epsilon <= 0.0:
        raise ResponseEstimationError("intervention magnitude must be positive")
    if denominator_floor <= 0.0:
        raise ResponseEstimationError("risk denominator floor must be positive")
    return (negative_risk - positive_risk) / (2.0 * epsilon * max(baseline_risk, denominator_floor))


def run_shadow_pair(
    model: torch.nn.Module,
    base_state: ModelParameterState,
    base_optimizer_state: OptimizerState,
    base_rng_state: RngState,
    data: ShadowData,
    settings: ShadowSettings,
    schedule_seed: int,
) -> tuple[tuple[float, float, float], ...]:
    config = active_config()
    batch_size = config.scientific.training.batch_size
    positive_rng = torch.Generator().manual_seed(schedule_seed)
    negative_rng = torch.Generator().manual_seed(schedule_seed)
    positive = _run_shadow(
        model,
        base_state,
        base_optimizer_state,
        base_rng_state,
        data,
        settings,
        1.0 + settings.epsilon,
        batch_size,
        positive_rng,
    )
    negative = _run_shadow(
        model,
        base_state,
        base_optimizer_state,
        base_rng_state,
        data,
        settings,
        1.0 - settings.epsilon,
        batch_size,
        negative_rng,
    )
    base_state.load_into(model)
    baseline = _evaluate_risks(
        model,
        data.meta_features,
        data.meta_targets,
        data.outcome_native_class_sets,
    )
    return tuple(
        (positive[index], negative[index], baseline[index])
        for index in range(len(data.outcome_native_class_sets))
    )


def _run_shadow(
    model: torch.nn.Module,
    base_state: ModelParameterState,
    base_optimizer_state: OptimizerState,
    base_rng_state: RngState,
    data: ShadowData,
    settings: ShadowSettings,
    multiplier: float,
    batch_size: int,
    schedule_rng: torch.Generator,
) -> tuple[float, ...]:
    if settings.horizon <= 0:
        raise ResponseEstimationError("shadow optimizer horizon must be positive")
    base_state.load_into(model)
    base_rng_state.restore()
    config = active_config()
    optimizer = make_adamw(model, settings.learning_rate, settings.weight_decay)
    base_optimizer_state.load_into(optimizer)
    model.train()
    schedule = shadow_batch_schedule(data.train_features.shape[0], batch_size, schedule_rng)
    device = next(model.parameters()).device
    for step in range(settings.horizon):
        batch = next(schedule)
        targets = data.train_targets[batch].to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.train_features[batch].to(device=device, dtype=torch.float32))
        per_example_ce = _shadow_ce(logits, targets)
        weights = _shadow_weights(
            data.base_class_weights,
            targets,
            data.intervention_classes,
            multiplier,
        )
        loss = (per_example_ce * weights).sum() / per_example_ce.numel()
        if not bool(torch.isfinite(loss)):
            raise ResponseEstimationError(f"non-finite shadow loss at optimizer step {step}")
        backward_value(loss)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.scientific.training.gradient_clip_global_l2_norm,
            norm_type=2.0,
        )
        optimizer_step(optimizer)
    model.eval()
    with torch.no_grad():
        shadow_logits = model(data.meta_features.to(device=device, dtype=torch.float32))
    return tuple(
        equal_native_class_risk(
            shadow_logits,
            data.meta_targets.to(device=device, dtype=torch.long),
            class_set,
            config.scientific.metrics.probability_log_floor,
        )
        for class_set in data.outcome_native_class_sets
    )


def _shadow_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    floor = active_config().scientific.metrics.probability_log_floor
    probabilities = torch.softmax(logits.to(dtype=torch.float32), dim=1)
    selected = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -torch.log(torch.clamp(selected, min=floor))


def _shadow_weights(
    class_weights: ClassWeights,
    targets: torch.Tensor,
    intervention_classes: tuple[int, ...],
    multiplier: float,
) -> torch.Tensor:
    multipliers = torch.ones_like(class_weights.values)
    for class_index in intervention_classes:
        multipliers[class_index] = multiplier
    return class_weights.per_example(targets, multipliers)


def _evaluate_risks(
    model: torch.nn.Module,
    meta_features: torch.Tensor,
    meta_targets: torch.Tensor,
    outcome_native_class_sets: tuple[tuple[int, ...], ...],
) -> tuple[float, ...]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(meta_features.to(device=device, dtype=torch.float32))
    targets = meta_targets.to(device=device, dtype=torch.long)
    return tuple(
        equal_native_class_risk(
            logits,
            targets,
            class_set,
            active_config().scientific.metrics.probability_log_floor,
        )
        for class_set in outcome_native_class_sets
    )
