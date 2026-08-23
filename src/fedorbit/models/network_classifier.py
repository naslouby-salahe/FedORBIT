from __future__ import annotations

import torch
from torch import nn


class NetworkFlowClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, dropout_probability: float) -> None:
        super().__init__()
        if input_dim <= 0 or n_classes <= 1:
            raise ValueError("network classifier dimensions must be positive")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.dropout_probability = dropout_probability
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256, eps=1e-5, elementwise_affine=True),
            nn.GELU(approximate="none"),
            nn.Dropout(dropout_probability),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128, eps=1e-5, elementwise_affine=True),
            nn.GELU(approximate="none"),
            nn.Dropout(dropout_probability),
        )
        self.block3 = nn.Sequential(nn.Linear(128, 64), nn.GELU(approximate="none"))
        self.classifier = nn.Linear(64, n_classes)
        self.to(dtype=torch.float32)

    def initialize(self, generator: torch.Generator) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0, generator=generator)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.block1(inputs.to(dtype=torch.float32))
        values = self.block2(values)
        values = self.block3(values)
        return self.classifier(values)
