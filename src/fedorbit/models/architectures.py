from __future__ import annotations

import torch
from torch import nn


class NetworkFlowClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
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
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(approximate="none"),
        )
        self.classifier = nn.Linear(64, n_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.block1(features)
        hidden = self.block2(hidden)
        hidden = self.block3(hidden)
        return self.classifier(hidden)


class HostClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(192, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(dropout_probability),
        )
        self.block2 = nn.Sequential(
            nn.Linear(192, 96),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_probability),
        )
        self.block3 = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Linear(48, n_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.block1(features)
        hidden = self.block2(hidden)
        hidden = self.block3(hidden)
        return self.classifier(hidden)


def classifier_for(
    kind: str, input_dim: int, n_classes: int, dropout_probability: float
) -> nn.Module:
    if kind == "network":
        return NetworkFlowClassifier(input_dim, n_classes, dropout_probability)
    if kind == "host":
        return HostClassifier(input_dim, n_classes, dropout_probability)
    raise ValueError(f"unknown classifier kind: {kind}")
