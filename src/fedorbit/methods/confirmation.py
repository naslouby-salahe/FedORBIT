from __future__ import annotations

import statistics
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.runtime import RandomSeed, SeedDerivationRequest, derive_seed32
from fedorbit.response.estimation import shadow_batch_schedule
from fedorbit.types import BatchSize, RngNamespace, SampleCount


class ConfirmationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ConfirmReplicateOutcomes:
    baseline_losses_by_class: tuple[torch.Tensor, ...]
    curriculum_losses_by_class: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if len(self.baseline_losses_by_class) != len(self.curriculum_losses_by_class):
            raise ConfirmationError("baseline and curriculum evaluation class sets differ")
        if not self.baseline_losses_by_class:
            raise ConfirmationError("confirmation requires at least one evaluation class")
        for baseline_classes, curriculum_classes in zip(
            self.baseline_losses_by_class, self.curriculum_losses_by_class, strict=True
        ):
            if len(baseline_classes.shape) != 1 or len(curriculum_classes.shape) != 1:
                raise ConfirmationError("per-class losses must be one-dimensional")
            if baseline_classes.shape[0] == 0 or curriculum_classes.shape[0] == 0:
                raise ConfirmationError(
                    "evaluation class with zero CONFIRM examples makes the cell Invalid Data"
                )


def confirmation_schedule(
    train_size: SampleCount,
    batch_size: BatchSize,
    seed: RandomSeed,
    coordinates: str,
) -> Iterator[torch.Tensor]:
    if train_size <= 0:
        raise ConfirmationError("confirmation TRAIN set is empty")
    if batch_size <= 0:
        raise ConfirmationError("confirmation batch size must be positive")
    rng = torch.Generator().manual_seed(
        derive_seed32(SeedDerivationRequest(seed, RngNamespace.CONFIRMATION_SCHEDULE, coordinates))
    )
    return shadow_batch_schedule(train_size, batch_size, rng)


def _tensor_mean(values: torch.Tensor) -> float:
    if values.shape[0] == 0:
        raise ConfirmationError("resampled class has zero examples")
    total = 0.0
    for position in range(values.shape[0]):
        total += float(values[int(position)])
    return total / values.shape[0]


def _macro_ce_from_losses(
    losses_by_class: tuple[torch.Tensor, ...],
    resample_indices: tuple[torch.Tensor, ...],
) -> float:
    if len(resample_indices) != len(losses_by_class):
        raise ConfirmationError("class resampling must cover every evaluation class")
    class_entropies = [
        _tensor_mean(losses[indices])
        for losses, indices in zip(losses_by_class, resample_indices, strict=True)
    ]
    return statistics.fmean(class_entropies)


def hierarchical_bootstrap_relative_gains(
    replicate_outcomes: tuple[ConfirmReplicateOutcomes, ...],
    seed: RandomSeed,
    contrast_coordinates: str,
) -> tuple[float, ...]:
    config = active_config()
    confirmation = config.scientific.confirmation
    denominator_floor = config.scientific.metrics.relative_macro_ce_denominator_floor
    if not replicate_outcomes:
        raise ConfirmationError("hierarchical bootstrap requires at least one replicate")
    replicate_count = len(replicate_outcomes)
    bootstrap_rng = torch.Generator().manual_seed(
        derive_seed32(
            SeedDerivationRequest(seed, RngNamespace.CONFIRMATION_BOOTSTRAP, contrast_coordinates)
        )
    )
    collected_gains: list[float] = []
    for _ in range(confirmation.hierarchical_bootstrap_resamples):
        selected = torch.randint(0, replicate_count, (replicate_count,), generator=bootstrap_rng)
        replicate_gains: list[float] = []
        for position in range(replicate_count):
            outcomes = replicate_outcomes[int(selected[position])]
            resample_indices = tuple(
                torch.randint(0, losses.shape[0], (losses.shape[0],), generator=bootstrap_rng)
                for losses in outcomes.baseline_losses_by_class
            )
            baseline = _macro_ce_from_losses(outcomes.baseline_losses_by_class, resample_indices)
            curriculum = _macro_ce_from_losses(
                outcomes.curriculum_losses_by_class, resample_indices
            )
            replicate_gains.append((baseline - curriculum) / max(baseline, denominator_floor))
        collected_gains.append(statistics.fmean(replicate_gains))
    return tuple(collected_gains)


def hierarchical_bootstrap_lower_bound(
    replicate_outcomes: tuple[ConfirmReplicateOutcomes, ...],
    seed: RandomSeed,
    contrast_coordinates: str,
) -> float:
    gains = hierarchical_bootstrap_relative_gains(replicate_outcomes, seed, contrast_coordinates)
    lower_probability = 1.0 - active_config().scientific.confirmation.one_sided_confidence_level
    return _linear_quantile(sorted(gains), lower_probability)


def _linear_quantile(sorted_values: list[float], probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ConfirmationError(f"quantile probability outside [0,1]: {probability}")
    count = len(sorted_values)
    if count == 1:
        return sorted_values[0]
    position = probability * (count - 1)
    lower_index = int(position // 1)
    upper_index = min(lower_index + 1, count - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] + fraction * (
        sorted_values[upper_index] - sorted_values[lower_index]
    )


def confirmation_decision(
    replicate_outcomes: tuple[ConfirmReplicateOutcomes, ...],
    seed: RandomSeed,
    contrast_coordinates: str,
) -> bool:
    lower_bound = hierarchical_bootstrap_lower_bound(replicate_outcomes, seed, contrast_coordinates)
    threshold = (
        active_config().scientific.confirmation.lower_bound_acceptance_threshold_relative_macro_ce
    )
    return lower_bound >= threshold
