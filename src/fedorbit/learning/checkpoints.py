from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from fedorbit.learning.training import (
    BaseCheckpoint,
    ClassWeights,
    ModelParameterState,
    NamedTensor,
    OptimizerState,
    RngState,
    SelectedHyperparameters,
)


@dataclass(frozen=True, slots=True)
class _CheckpointPayload:
    epoch: int #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    valid_macro_cross_entropy: float #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    state_dict: tuple[tuple[str, torch.Tensor], ...] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    optimizer_state: bytes
    rng_cpu: torch.Tensor
    rng_cuda: tuple[torch.Tensor, ...]
    learning_rate: float #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    weight_decay: float #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    dropout_probability: float #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    train_class_weights: torch.Tensor


def save_base_checkpoint(checkpoint: BaseCheckpoint, destination: Path) -> None:
    payload = _CheckpointPayload(
        epoch=checkpoint.epoch,
        valid_macro_cross_entropy=checkpoint.valid_macro_cross_entropy,
        state_dict=tuple((entry.name, entry.value) for entry in checkpoint.state_dict.tensors),
        optimizer_state=checkpoint.optimizer_state.payload,
        rng_cpu=checkpoint.rng_state.cpu,
        rng_cuda=checkpoint.rng_state.cuda,
        learning_rate=checkpoint.selected_hyperparameters.learning_rate,
        weight_decay=checkpoint.selected_hyperparameters.weight_decay,
        dropout_probability=checkpoint.selected_hyperparameters.dropout_probability,
        train_class_weights=checkpoint.train_class_weights.values,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_base_checkpoint(source: Path) -> BaseCheckpoint:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    return BaseCheckpoint(
        epoch=payload.epoch,
        valid_macro_cross_entropy=payload.valid_macro_cross_entropy,
        state_dict=ModelParameterState(
            tuple(NamedTensor(name, value) for name, value in payload.state_dict)
        ),
        optimizer_state=OptimizerState(payload.optimizer_state),
        rng_state=RngState(payload.rng_cpu, payload.rng_cuda),
        selected_hyperparameters=SelectedHyperparameters(
            payload.learning_rate,
            payload.weight_decay,
            payload.dropout_probability,
        ),
        train_class_weights=ClassWeights(payload.train_class_weights),
    )
