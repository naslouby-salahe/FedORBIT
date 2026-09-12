from __future__ import annotations

import torch
from torch import nn

from fedorbit.types import ConceptCount, FeatureCount, Fraction

HOST_BLOCK_WIDTHS = (192, 96, 48) #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
HOST_BATCH_NORM_EPSILON = 1e-5 #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
HOST_BATCH_NORM_MOMENTUM = 0.1 #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it


class NetworkFlowClassifier(nn.Module):
    def __init__(
        self, input_dim: FeatureCount, n_classes: ConceptCount, dropout_probability: Fraction
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_classes <= 1:
            raise ValueError("network classifier dimensions must be positive")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.dropout_probability = dropout_probability
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256), #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
            nn.LayerNorm(256, eps=1e-5, elementwise_affine=True), #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
            nn.GELU(approximate="none"),
            nn.Dropout(dropout_probability),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128), #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
            nn.LayerNorm(128, eps=1e-5, elementwise_affine=True), #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
            nn.GELU(approximate="none"),
            nn.Dropout(dropout_probability),
        )
        self.block3 = nn.Sequential(nn.Linear(128, 64), nn.GELU(approximate="none"))
        self.classifier = nn.Linear(64, n_classes) #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
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


class HostClassifier(nn.Module):
    def __init__(
        self, input_dim: FeatureCount, n_classes: ConceptCount, dropout_probability: Fraction
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_classes <= 1:
            raise ValueError("host classifier dimensions must be positive")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.dropout_probability = dropout_probability
        first_width, second_width, third_width = HOST_BLOCK_WIDTHS
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, first_width),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(
                first_width,
                eps=HOST_BATCH_NORM_EPSILON,
                momentum=HOST_BATCH_NORM_MOMENTUM,
                affine=True,
                track_running_stats=True,
            ),
            nn.Dropout(dropout_probability),
        )
        self.block2 = nn.Sequential(
            nn.Linear(first_width, second_width),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_probability),
        )
        self.block3 = nn.Sequential(nn.Linear(second_width, third_width), nn.ReLU(inplace=False))
        self.classifier = nn.Linear(third_width, n_classes) #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
        self.to(dtype=torch.float32)

    def initialize(self, generator: torch.Generator) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=0.0,
                    mode="fan_in",
                    nonlinearity="relu",
                    generator=generator,
                )
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.block1(inputs.to(dtype=torch.float32))
        values = self.block2(values)
        values = self.block3(values)
        return self.classifier(values)
