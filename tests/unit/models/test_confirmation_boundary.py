from __future__ import annotations

import torch

from fedorbit.config.loading import active_config
from fedorbit.methods.confirmation import (
    ConfirmReplicateOutcomes,
    confirmation_decision,
    hierarchical_bootstrap_lower_bound,
    hierarchical_bootstrap_relative_gains,
)
from fedorbit.types import (
    ContrastCoordinates,
    RelativeGain,
)

_REFERENCE_LOSS = 100.0
_BOUNDARY_IMPROVEMENT = 1.0


def _constant_outcomes(
    baseline_loss: float,
    curriculum_loss: float,
    replicate_count: int,
    example_count: int = 3,
) -> tuple[ConfirmReplicateOutcomes, ...]:
    return tuple(
        ConfirmReplicateOutcomes(
            baseline_losses_by_class=(
                torch.full((example_count,), baseline_loss, dtype=torch.float64),
            ),
            curriculum_losses_by_class=(
                torch.full((example_count,), curriculum_loss, dtype=torch.float64),
            ),
        )
        for _ in range(replicate_count)
    )


def _registered_replicate_count() -> int:
    return active_config().scientific.confirmation.paired_replicates


def _coordinates() -> ContrastCoordinates:
    return ContrastCoordinates("confirmation-boundary-fixture")


def test_confirmation_accepts_exactly_at_the_registered_relative_macro_ce_threshold() -> None:
    threshold = (
        active_config().scientific.confirmation.lower_bound_acceptance_threshold_relative_macro_ce
    )
    assert threshold == 0.01
    at_threshold = _constant_outcomes(
        _REFERENCE_LOSS,
        _REFERENCE_LOSS - _BOUNDARY_IMPROVEMENT,
        _registered_replicate_count(),
    )
    above_threshold = _constant_outcomes(
        _REFERENCE_LOSS,
        _REFERENCE_LOSS - _BOUNDARY_IMPROVEMENT - 0.01,
        _registered_replicate_count(),
    )
    below_threshold = _constant_outcomes(
        _REFERENCE_LOSS,
        _REFERENCE_LOSS - _BOUNDARY_IMPROVEMENT + 0.01,
        _registered_replicate_count(),
    )
    lower_bound_at_threshold = hierarchical_bootstrap_lower_bound(
        at_threshold, 1103, _coordinates()
    )
    assert lower_bound_at_threshold == threshold
    assert confirmation_decision(at_threshold, 1103, _coordinates()) is True
    assert confirmation_decision(above_threshold, 1103, _coordinates()) is True
    assert confirmation_decision(below_threshold, 1103, _coordinates()) is False


def test_confirmation_rejects_below_the_registered_threshold_with_reproducible_bounds() -> None:
    threshold = (
        active_config().scientific.confirmation.lower_bound_acceptance_threshold_relative_macro_ce
    )
    outcomes = _constant_outcomes(
        _REFERENCE_LOSS,
        _REFERENCE_LOSS - _BOUNDARY_IMPROVEMENT + 0.5,
        _registered_replicate_count(),
    )
    first = hierarchical_bootstrap_lower_bound(outcomes, 1103, _coordinates())
    second = hierarchical_bootstrap_lower_bound(outcomes, 1103, _coordinates())
    assert first == second
    assert first < threshold
    assert confirmation_decision(outcomes, 1103, _coordinates()) is False


def test_hierarchical_bootstrap_uses_the_registered_resample_count_and_seed_derivation() -> None:
    confirmation = active_config().scientific.confirmation
    outcomes = tuple(
        ConfirmReplicateOutcomes(
            baseline_losses_by_class=(torch.tensor((1.0, 1.1, 0.9, 1.05), dtype=torch.float64),),
            curriculum_losses_by_class=(torch.tensor((0.8, 0.95, 0.7, 0.85), dtype=torch.float64),),
        )
        for _ in range(confirmation.paired_replicates)
    )
    gains = hierarchical_bootstrap_relative_gains(outcomes, 1103, _coordinates())
    assert len(gains) == confirmation.hierarchical_bootstrap_resamples
    assert len(hierarchical_bootstrap_relative_gains(outcomes, 1103, _coordinates())) == len(gains)
    assert hierarchical_bootstrap_relative_gains(outcomes, 1103, _coordinates()) == gains
    other_coordinates = ContrastCoordinates("confirmation-boundary-fixture-other")
    assert hierarchical_bootstrap_relative_gains(outcomes, 1103, other_coordinates) != gains
    lower = hierarchical_bootstrap_lower_bound(outcomes, 1103, _coordinates())
    assert min(gains) <= lower <= max(gains)
    assert confirmation.optimizer_steps_per_shadow == 100
    assert confirmation.one_sided_confidence_level == 0.95


def test_confirmation_lower_bound_is_the_registered_one_sided_quantile() -> None:
    confirmation = active_config().scientific.confirmation
    gains = hierarchical_bootstrap_relative_gains(
        _constant_outcomes(_REFERENCE_LOSS, _REFERENCE_LOSS * 0.9, confirmation.paired_replicates),
        1103,
        _coordinates(),
    )
    expected_probability = 1.0 - confirmation.one_sided_confidence_level
    ordered = sorted(gains)
    position = expected_probability * (len(ordered) - 1)
    lower_index = int(position // 1)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    expected: RelativeGain = ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )
    assert (
        hierarchical_bootstrap_lower_bound(
            _constant_outcomes(
                _REFERENCE_LOSS, _REFERENCE_LOSS * 0.9, confirmation.paired_replicates
            ),
            1103,
            _coordinates(),
        )
        == expected
    )
