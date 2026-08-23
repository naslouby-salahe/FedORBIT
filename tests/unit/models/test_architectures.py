from __future__ import annotations

import math

import torch
from torch import nn

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.models.architectures import HostClassifier, NetworkFlowClassifier, classifier_for
from fedorbit.models.training import (
    TrainingError,
    macro_cross_entropy,
    train_base_model,
    weighted_mean_loss,
)


def _count_linear_layers(module: nn.Module) -> int:
    return sum(1 for child in module.modules() if isinstance(child, nn.Linear))


def test_network_flow_architecture_order() -> None:
    model = NetworkFlowClassifier(input_dim=8, n_classes=5, dropout_probability=0.3)
    blocks = list(model.block1) + list(model.block2) + list(model.block3) + [model.classifier]
    assert isinstance(blocks[0], nn.Linear)
    assert blocks[0].in_features == 8
    assert blocks[0].out_features == 256
    assert isinstance(blocks[1], nn.LayerNorm)
    assert isinstance(blocks[2], nn.GELU)
    assert isinstance(blocks[3], nn.Dropout)
    assert isinstance(blocks[4], nn.Linear)
    assert blocks[4].in_features == 256
    assert blocks[4].out_features == 128
    assert isinstance(blocks[5], nn.LayerNorm)
    assert isinstance(blocks[6], nn.GELU)
    assert isinstance(blocks[7], nn.Dropout)
    assert isinstance(blocks[8], nn.Linear)
    assert blocks[8].in_features == 128
    assert blocks[8].out_features == 64
    assert isinstance(blocks[9], nn.GELU)
    assert isinstance(blocks[10], nn.Linear)
    assert blocks[10].in_features == 64
    assert blocks[10].out_features == 5
    assert _count_linear_layers(model) == 4


def test_network_flow_layer_norm_parameters() -> None:
    model = NetworkFlowClassifier(input_dim=8, n_classes=5, dropout_probability=0.3)
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            assert module.eps == 1e-5
            assert module.elementwise_affine


def test_network_flow_gelu_exact() -> None:
    model = NetworkFlowClassifier(input_dim=8, n_classes=5, dropout_probability=0.3)
    for module in model.modules():
        if isinstance(module, nn.GELU):
            assert module.approximate == "none"


def test_network_flow_dropout_uses_pilot_value() -> None:
    model = NetworkFlowClassifier(input_dim=8, n_classes=5, dropout_probability=0.4)
    assert model.dropout_probability == 0.4
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            assert module.p == 0.4


def test_network_flow_xavier_uniform_zero_bias() -> None:
    torch.manual_seed(0)
    model = NetworkFlowClassifier(input_dim=8, n_classes=5, dropout_probability=0.3)
    torch.manual_seed(0)
    reference = NetworkFlowClassifier(input_dim=8, n_classes=5, dropout_probability=0.3)
    for (name, parameter), (_, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if "bias" in name:
            assert torch.equal(parameter, torch.zeros_like(parameter))
        else:
            assert torch.equal(parameter, reference_parameter)


def test_host_architecture_order() -> None:
    model = HostClassifier(input_dim=16, n_classes=4, dropout_probability=0.2)
    blocks = list(model.block1) + list(model.block2) + list(model.block3) + [model.classifier]
    assert isinstance(blocks[0], nn.Linear)
    assert blocks[0].in_features == 16
    assert blocks[0].out_features == 192
    assert isinstance(blocks[1], nn.ReLU)
    assert not blocks[1].inplace
    assert isinstance(blocks[2], nn.BatchNorm1d)
    assert isinstance(blocks[3], nn.Dropout)
    assert isinstance(blocks[4], nn.Linear)
    assert blocks[4].out_features == 96
    assert isinstance(blocks[5], nn.ReLU)
    assert isinstance(blocks[6], nn.Dropout)
    assert isinstance(blocks[7], nn.Linear)
    assert blocks[7].out_features == 48
    assert isinstance(blocks[8], nn.ReLU)
    assert isinstance(blocks[9], nn.Linear)
    assert blocks[9].out_features == 4


def test_host_batch_norm_parameters() -> None:
    model = HostClassifier(input_dim=16, n_classes=4, dropout_probability=0.2)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            assert module.eps == 1e-5
            assert module.momentum == 0.1
            assert module.affine
            assert module.track_running_stats


def test_host_kaiming_uniform_zero_bias() -> None:
    torch.manual_seed(1)
    model = HostClassifier(input_dim=16, n_classes=4, dropout_probability=0.2)
    torch.manual_seed(1)
    reference = HostClassifier(input_dim=16, n_classes=4, dropout_probability=0.2)
    for (name, parameter), (_, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if "bias" in name:
            assert torch.equal(parameter, torch.zeros_like(parameter))
        else:
            assert torch.equal(parameter, reference_parameter)


def test_classifier_for_kind() -> None:
    assert isinstance(classifier_for("network", 8, 5, 0.3), NetworkFlowClassifier)
    assert isinstance(classifier_for("host", 16, 4, 0.2), HostClassifier)


def test_macro_cross_entropy_hand_solvable() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1])
    weights = torch.tensor([1.0, 1.0])
    metric = macro_cross_entropy(logits, targets, weights, probability_log_floor=1e-12)
    expected = -(float(torch.log(torch.softmax(torch.tensor([2.0, 0.0]), dim=0)[0])))
    assert metric == expected


def test_macro_cross_entropy_weights_classes() -> None:
    logits = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 0, 1])
    equal = macro_cross_entropy(
        logits, targets, torch.tensor([1.0, 1.0]), probability_log_floor=1e-12
    )
    weighted = macro_cross_entropy(
        logits, targets, torch.tensor([2.0, 1.0]), probability_log_floor=1e-12
    )
    assert weighted != equal


def test_weighted_mean_loss_weights_examples() -> None:
    per_example = torch.tensor([1.0, 3.0])
    targets = torch.tensor([0, 1])
    weights = torch.tensor([1.0, 2.0])
    mean = weighted_mean_loss(per_example, weights, targets)
    assert float(mean) == (1.0 * 1.0 + 3.0 * 2.0) / 2.0


def test_train_base_model_smoke() -> None:
    config = load_fedorbit_config()
    torch.manual_seed(42)
    model = NetworkFlowClassifier(input_dim=4, n_classes=2, dropout_probability=0.1)
    train_features = torch.randn(40, 4)
    train_targets = torch.randint(0, 2, (40,))
    valid_features = torch.randn(10, 4)
    valid_targets = torch.randint(0, 2, (10,))
    weights = torch.ones(2)
    outcome = train_base_model(
        config,
        model,
        train_features,
        train_targets,
        valid_features,
        valid_targets,
        weights,
        seed=7,
        learning_rate=1e-3,
        weight_decay=0.0,
    )
    assert outcome.epoch >= 0
    assert math.isfinite(outcome.valid_macro_cross_entropy)
    assert outcome.checkpoint.epoch == outcome.epoch
    assert outcome.checkpoint.valid_macro_cross_entropy == outcome.valid_macro_cross_entropy
    assert set(outcome.checkpoint.state_dict.tensors_by_name) == set(model.state_dict())


def test_train_base_model_rejects_empty_train() -> None:
    config = load_fedorbit_config()
    model = NetworkFlowClassifier(input_dim=4, n_classes=2, dropout_probability=0.1)
    try:
        train_base_model(
            config,
            model,
            torch.empty(0, 4),
            torch.empty(0, dtype=torch.long),
            torch.randn(5, 4),
            torch.randint(0, 2, (5,)),
            torch.ones(2),
            seed=7,
            learning_rate=1e-3,
            weight_decay=0.0,
        )
    except TrainingError:
        return
    raise AssertionError("expected TrainingError for empty TRAIN")
