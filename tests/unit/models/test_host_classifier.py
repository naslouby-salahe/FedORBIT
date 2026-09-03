from __future__ import annotations

import torch
from torch import nn

from fedorbit.learning.models import HostClassifier


def test_host_classifier_contract() -> None:
    model = HostClassifier(16, 4, 0.2)
    blocks = list(model.block1) + list(model.block2) + list(model.block3) + [model.classifier]
    assert [(type(block), getattr(block, "out_features", None)) for block in blocks] == [
        (nn.Linear, 192),
        (nn.ReLU, None),
        (nn.BatchNorm1d, None),
        (nn.Dropout, None),
        (nn.Linear, 96),
        (nn.ReLU, None),
        (nn.Dropout, None),
        (nn.Linear, 48),
        (nn.ReLU, None),
        (nn.Linear, 4),
    ]
    batch_norm = next(module for module in model.modules() if isinstance(module, nn.BatchNorm1d))
    assert batch_norm.eps == 1e-5
    assert batch_norm.momentum == 0.1
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_host_initialization_is_named_generator_deterministic() -> None:
    first = HostClassifier(16, 4, 0.1)
    second = HostClassifier(16, 4, 0.1)
    first.initialize(torch.Generator().manual_seed(53))
    second.initialize(torch.Generator().manual_seed(53))
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)
    for name, parameter in first.named_parameters():
        if name.endswith("bias"):
            assert torch.equal(parameter, torch.zeros_like(parameter))
