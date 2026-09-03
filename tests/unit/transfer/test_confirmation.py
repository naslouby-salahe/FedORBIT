from __future__ import annotations

import pytest
import torch

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.methods.confirmation import (
    ConfirmationError,
    ConfirmReplicateOutcomes,
    confirmation_decision,
    confirmation_schedule,
    hierarchical_bootstrap_lower_bound,
    hierarchical_bootstrap_relative_gains,
)


def _outcomes(
    baseline_means: tuple[float, ...],
    curriculum_means: tuple[float, ...],
    examples_per_class: int,
    seed: int,
) -> ConfirmReplicateOutcomes:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(examples_per_class, generator=generator)
    perturbed: list[float] = [0.01 * float(noise[int(index)]) for index in range(noise.shape[0])]

    def losses(means: tuple[float, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.tensor([float(shift + offset) for offset in perturbed]) for shift in means
        )

    return ConfirmReplicateOutcomes(
        baseline_losses_by_class=losses(baseline_means),
        curriculum_losses_by_class=losses(curriculum_means),
    )


@pytest.fixture
def config() -> FedorbitConfig:
    return load_fedorbit_config()


def test_schedule_is_infinite_pass_with_fresh_permutations() -> None:
    batch_size = 128
    train_size = 1000
    schedule = confirmation_schedule(train_size, batch_size, 1103, "pair-a")
    first_pass = [next(schedule) for _ in range(2)]
    assert first_pass[0].shape[0] == batch_size
    replay = confirmation_schedule(train_size, batch_size, 1103, "pair-a")
    assert torch.equal(first_pass[0], next(replay))
    other = confirmation_schedule(train_size, batch_size, 1103, "pair-b")
    assert not torch.equal(first_pass[0], next(other))
    with pytest.raises(ConfirmationError):
        confirmation_schedule(0, batch_size, 1103, "x")
    with pytest.raises(ConfirmationError):
        confirmation_schedule(1000, 0, 1103, "x")


def test_outcomes_reject_empty_classes_and_mismatched_structures() -> None:
    losses = (torch.tensor([0.1, 0.2]),)
    empty = (torch.tensor([]), torch.tensor([0.1]))
    with pytest.raises(ConfirmationError):
        ConfirmReplicateOutcomes(losses, ())
    with pytest.raises(ConfirmationError):
        ConfirmReplicateOutcomes(empty, empty)
    good = (torch.tensor([0.3]), torch.tensor([0.4]))
    ConfirmReplicateOutcomes(good, good)


def test_bootstrap_gain_averages_across_sampled_replicates(
    config: FedorbitConfig,
) -> None:
    replicates = tuple(
        _outcomes((2.0,), (1.5,), examples_per_class=64, seed=100 + index)
        for index in range(config.scientific.confirmation.paired_replicates)
    )
    gains = hierarchical_bootstrap_relative_gains(replicates, 1103, "unit-test-contrast")
    assert len(gains) == config.scientific.confirmation.hierarchical_bootstrap_resamples
    expected = (2.0 - 1.5) / 2.0
    assert all(abs(gain - expected) < 0.05 for gain in gains)


def test_lower_bound_uses_linear_quantile_of_lower_tail(config: FedorbitConfig) -> None:
    replicates = tuple(
        _outcomes((2.0,), (1.6,), examples_per_class=128, seed=200 + index)
        for index in range(config.scientific.confirmation.paired_replicates)
    )
    lower_bound = hierarchical_bootstrap_lower_bound(replicates, 2207, "lower-bound-test")
    gains = hierarchical_bootstrap_relative_gains(replicates, 2207, "lower-bound-test")
    assert lower_bound == pytest.approx(min(gains), abs=0.05)
    assert lower_bound <= max(gains) + 1e-12


def test_acceptance_requires_threshold_on_lower_bound(config: FedorbitConfig) -> None:
    strong = tuple(
        _outcomes((2.0,), (1.0,), examples_per_class=256, seed=300 + index)
        for index in range(config.scientific.confirmation.paired_replicates)
    )
    weak = tuple(
        _outcomes((2.0,), (1.99,), examples_per_class=256, seed=400 + index)
        for index in range(config.scientific.confirmation.paired_replicates)
    )
    assert confirmation_decision(strong, 3319, "strong")
    assert not confirmation_decision(weak, 3319, "weak")


def test_harmful_curriculum_rejects_transfer(config: FedorbitConfig) -> None:
    harmful = tuple(
        _outcomes((1.5,), (2.5,), examples_per_class=256, seed=500 + index)
        for index in range(config.scientific.confirmation.paired_replicates)
    )
    assert not confirmation_decision(harmful, 4421, "harmful")
