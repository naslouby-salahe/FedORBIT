from __future__ import annotations

import pytest
import torch

from fedorbit.training.losses import (
    ClassWeights,
    LossContractError,
    minibatch_objective,
    per_example_weighted_cross_entropy,
)


def test_class_weights_follow_train_frequency_contract() -> None:
    targets = torch.tensor([0, 0, 0, 1])
    weights = ClassWeights.from_targets(targets, 2)
    assert weights.values.tolist() == pytest.approx([2.0 / 3.0, 2.0])
    mean_example_weight = float(weights.values[targets].mean())
    assert mean_example_weight == pytest.approx(1.0)


def test_weighted_cross_entropy_uses_example_count_denominator() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1])
    weights = ClassWeights(torch.tensor([1.0, 2.0]))
    per_example = per_example_weighted_cross_entropy(logits, targets, weights, 1e-12)
    objective = minibatch_objective(logits, targets, weights, 1e-12)
    assert float(objective) == pytest.approx(float(per_example.sum() / 2.0))


def test_class_weight_contract_rejects_missing_train_class() -> None:
    with pytest.raises(LossContractError):
        ClassWeights.from_targets(torch.tensor([0, 0, 0]), 2)
