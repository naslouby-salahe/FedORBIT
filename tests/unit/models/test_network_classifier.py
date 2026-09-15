from __future__ import annotations

import torch
from torch import nn

from fedorbit.learning.models import NetworkFlowClassifier


def test_network_classifier_contract() -> None:
    model = NetworkFlowClassifier(8, 5, 0.3)
    blocks = list(model.block1) + list(model.block2) + list(model.block3) + [model.classifier]
    assert [(type(block), getattr(block, "out_features", None)) for block in blocks] == [
        (nn.Linear, 256),
        (nn.LayerNorm, None),
        (nn.GELU, None),
        (nn.Dropout, None),
        (nn.Linear, 128),
        (nn.LayerNorm, None),
        (nn.GELU, None),
        (nn.Dropout, None),
        (nn.Linear, 64),
        (nn.GELU, None),
        (nn.Linear, 5),
    ]
    layer_norms = [module for module in model.modules() if isinstance(module, nn.LayerNorm)]
    assert len(layer_norms) == 2
    assert all(module.eps == 1e-5 and module.elementwise_affine for module in layer_norms)
    gelus = [module for module in model.modules() if isinstance(module, nn.GELU)]
    assert len(gelus) == 3
    assert all(module.approximate == "none" for module in gelus)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_network_initialization_is_named_generator_deterministic() -> None:
    first = NetworkFlowClassifier(8, 5, 0.1)
    second = NetworkFlowClassifier(8, 5, 0.1)
    first.initialize(torch.Generator().manual_seed(41))
    second.initialize(torch.Generator().manual_seed(41))
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)
    for name, parameter in first.named_parameters():
        if name.endswith("bias"):
            assert torch.equal(parameter, torch.zeros_like(parameter))
    expected_generator = torch.Generator().manual_seed(41)
    for module in first.modules():
        if isinstance(module, nn.Linear):
            expected = torch.empty_like(module.weight)
            nn.init.xavier_uniform_(expected, gain=1.0, generator=expected_generator)
            assert torch.equal(module.weight, expected)
