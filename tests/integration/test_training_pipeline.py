from __future__ import annotations

import torch

from fedorbit.learning.models import NetworkFlowClassifier
from fedorbit.learning.training import ClassWeights, SelectedHyperparameters, train_base_model


def test_training_pipeline_produces_complete_checkpoint() -> None:
    generator = torch.Generator().manual_seed(7)
    train_features = torch.randn(32, 4, generator=generator)
    train_targets = torch.tensor([0, 1] * 16)
    valid_features = torch.randn(12, 4, generator=generator)
    valid_targets = torch.tensor([0, 1] * 6)
    weights = ClassWeights.from_targets(train_targets, 2)
    selected = SelectedHyperparameters(1e-3, 0.0, 0.0)
    model = NetworkFlowClassifier(4, 2, selected.dropout_probability)
    model.initialize(torch.Generator().manual_seed(11))
    outcome = train_base_model(
        model,
        train_features,
        train_targets,
        valid_features,
        valid_targets,
        weights,
        101,
        selected,
    )
    assert outcome.checkpoint.selected_hyperparameters == selected
    assert torch.equal(outcome.checkpoint.train_class_weights.values, weights.values)
