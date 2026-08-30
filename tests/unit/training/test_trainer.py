from __future__ import annotations

import math

import pytest
import torch

from fedorbit.models.network_classifier import NetworkFlowClassifier
from fedorbit.training.losses import ClassWeights
from fedorbit.training.trainer import (
    SelectedHyperparameters,
    macro_cross_entropy,
    make_adamw,
    train_base_model,
)


def test_macro_cross_entropy_is_equal_class_average() -> None:
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 0, 1])
    value = macro_cross_entropy(logits, targets, 1e-12)
    probabilities = torch.softmax(logits, dim=1)
    class_zero = -torch.log(probabilities[:2, 0]).mean()
    class_one = -torch.log(probabilities[2:, 1]).mean()
    assert value == pytest.approx(float((class_zero + class_one) / 2.0))


def test_adamw_contract_disables_unregistered_variants() -> None:
    model = NetworkFlowClassifier(4, 2, 0.0)
    optimizer = make_adamw(model, 1e-3, 0.0)
    group = optimizer.param_groups[0]
    assert group["amsgrad"] is False
    assert group["maximize"] is False
    assert group["foreach"] is False
    assert group["fused"] is False


def test_training_checkpoint_contains_complete_reusable_state() -> None:
    generator = torch.Generator().manual_seed(17)
    train_features = torch.randn(32, 4, generator=generator)
    train_targets = torch.tensor([0, 1] * 16)
    valid_features = torch.randn(12, 4, generator=generator)
    valid_targets = torch.tensor([0, 1] * 6)
    class_weights = ClassWeights.from_targets(train_targets, 2)
    selected = SelectedHyperparameters(1e-3, 0.0, 0.0)
    model = NetworkFlowClassifier(4, 2, selected.dropout_probability)
    model.initialize(torch.Generator().manual_seed(19))
    outcome = train_base_model(
        model,
        train_features,
        train_targets,
        valid_features,
        valid_targets,
        class_weights,
        101,
        selected,
    )
    checkpoint = outcome.checkpoint
    assert checkpoint.epoch >= 0
    assert math.isfinite(checkpoint.valid_macro_cross_entropy)
    assert checkpoint.selected_hyperparameters == selected
    assert torch.equal(checkpoint.train_class_weights.values, class_weights.values)
    assert checkpoint.state_dict.tensors
    assert checkpoint.optimizer_state.payload
    assert checkpoint.rng_state.cpu.numel() > 0
