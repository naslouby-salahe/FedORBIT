from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fedorbit.config.models import AutoencoderDetectorConfig
from fedorbit.types import FloatMatrix, FloatVector, RandomSeed

StateDictionary = dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class AutoencoderScorer:
    network: nn.Module
    centre: FloatVector
    scale: FloatVector
    device: torch.device

    def score(self, rows: FloatMatrix) -> FloatVector:
        standardised = torch.as_tensor(
            (rows - self.centre) / self.scale, dtype=torch.float32, device=self.device
        )
        self.network.eval()
        with torch.no_grad():
            error = ((self.network(standardised) - standardised) ** 2).mean(dim=1)
        return error.cpu().numpy().astype(np.float64)


def build_network(width: int, config: AutoencoderDetectorConfig, seed: RandomSeed) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(width, config.hidden_width),
        nn.ReLU(),
        nn.Linear(config.hidden_width, config.bottleneck_width),
        nn.ReLU(),
        nn.Linear(config.bottleneck_width, config.hidden_width),
        nn.ReLU(),
        nn.Linear(config.hidden_width, width),
    )


def _train_steps(
    network: nn.Module,
    rows: torch.Tensor,
    steps: int,
    config: AutoencoderDetectorConfig,
    generator: torch.Generator,
) -> None:
    optimiser = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    batch = min(config.batch_size, len(rows))
    network.train()
    for _ in range(steps):
        indices = torch.randint(len(rows), (batch,), generator=generator).to(rows.device)
        batch_rows = rows[indices]
        optimiser.zero_grad()
        loss = ((network(batch_rows) - batch_rows) ** 2).mean()
        loss.backward()
        optimiser.step()


def _to_tensor(rows: FloatMatrix, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(rows, dtype=torch.float32, device=device)


def train_local_autoencoder(
    standardised_rows: FloatMatrix,
    config: AutoencoderDetectorConfig,
    seed: RandomSeed,
    device: torch.device,
) -> nn.Module:
    network = build_network(standardised_rows.shape[1], config, seed).to(device)
    generator = torch.Generator().manual_seed(seed)
    _train_steps(
        network, _to_tensor(standardised_rows, device), config.training_steps, config, generator
    )
    return network


def train_federated_autoencoder(
    client_rows: Sequence[FloatMatrix],
    config: AutoencoderDetectorConfig,
    seed: RandomSeed,
    device: torch.device,
) -> nn.Module:
    global_network = build_network(client_rows[0].shape[1], config, seed).to(device)
    generator = torch.Generator().manual_seed(seed)
    tensors = [_to_tensor(rows, device) for rows in client_rows]
    weights = np.array([len(rows) for rows in client_rows], dtype=np.float64)
    weights = weights / weights.sum()
    for _ in range(config.federated_rounds):
        global_state: StateDictionary = {
            name: value.clone() for name, value in global_network.state_dict().items()
        }
        averaged: StateDictionary = {
            name: torch.zeros_like(value) for name, value in global_state.items()
        }
        for tensor, weight in zip(tensors, weights, strict=True):
            client_network = build_network(tensor.shape[1], config, seed).to(device)
            client_network.load_state_dict(global_state)
            _train_steps(client_network, tensor, config.federated_local_steps, config, generator)
            for name, value in client_network.state_dict().items():
                averaged[name] += float(weight) * value
        global_network.load_state_dict(averaged)
    return global_network
