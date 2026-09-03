from __future__ import annotations

import io
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import nn
from torch.optim.optimizer import StateDict
from torch.utils.data import DataLoader, TensorDataset

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.runtime import RandomSeed, SeedDerivationRequest, derive_seed32
from fedorbit.types import RngNamespace, StableJsonPayload


class LossContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ClassWeights:
    values: torch.Tensor

    @classmethod
    def from_targets(cls, targets: torch.Tensor, n_classes: int) -> ClassWeights:
        if targets.ndim != 1 or targets.numel() == 0:
            raise LossContractError("TRAIN targets must be a non-empty one-dimensional tensor")
        if n_classes <= 0:
            raise LossContractError("class count must be positive")
        counts = torch.bincount(targets.to(dtype=torch.long), minlength=n_classes).to(
            dtype=torch.float64
        )
        if bool((counts <= 0).any()):
            raise LossContractError("every local prediction class must have TRAIN support")
        total = float(counts.sum())
        raw = total / (float(n_classes) * counts)
        example_weighted_mean = float((raw * counts).sum()) / total
        normalized = raw / example_weighted_mean
        return cls(normalized.to(dtype=torch.float32))

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.values.numel() == 0:
            raise LossContractError("class weights must be a non-empty vector")
        if not bool(torch.isfinite(self.values).all()) or bool((self.values <= 0).any()):
            raise LossContractError("class weights must be finite and positive")

    def per_example(
        self,
        targets: torch.Tensor,
        multipliers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = self.values.to(device=targets.device)[targets]
        if multipliers is None:
            return weights
        if multipliers.shape != self.values.shape:
            raise LossContractError("class multiplier shape differs from class weights")
        return weights * multipliers.to(device=targets.device)[targets]


def per_example_weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: ClassWeights,
    probability_log_floor: float,
    multipliers: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise LossContractError("logits and targets have incompatible shapes")
    if not 0.0 < probability_log_floor < 1.0:
        raise LossContractError("probability log floor must be in (0, 1)")
    probabilities = torch.softmax(logits.to(dtype=torch.float32), dim=1)
    selected = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    losses = -torch.log(torch.clamp(selected, min=probability_log_floor))
    return losses * class_weights.per_example(targets, multipliers)


def minibatch_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: ClassWeights,
    probability_log_floor: float,
    multipliers: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = per_example_weighted_cross_entropy(
        logits,
        targets,
        class_weights,
        probability_log_floor,
        multipliers,
    )
    if losses.numel() == 0:
        raise LossContractError("minibatch must contain at least one example")
    return losses.sum() / losses.numel()


class TrainingError(ValueError):
    pass


class _BackwardValue(Protocol):
    def backward(self) -> None: ...


class _OptimizerStep(Protocol):
    def step(self) -> None: ...


def backward_value(loss: torch.Tensor) -> None:
    cast(_BackwardValue, loss).backward()


def optimizer_step(optimizer: torch.optim.Optimizer) -> None:
    cast(_OptimizerStep, optimizer).step()


@dataclass(frozen=True, slots=True)
class NamedTensor:
    name: str
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class ModelParameterState:
    tensors: tuple[NamedTensor, ...]

    @classmethod
    def capture(cls, model: nn.Module) -> ModelParameterState:
        return cls(
            tuple(
                NamedTensor(name, value.detach().cpu().clone())
                for name, value in model.state_dict().items()
            )
        )

    def load_into(self, model: nn.Module) -> None:
        model.load_state_dict(
            OrderedDict((entry.name, entry.value.clone()) for entry in self.tensors), strict=True
        )


@dataclass(frozen=True, slots=True)
class OptimizerState:
    payload: bytes

    @classmethod
    def capture(cls, optimizer: torch.optim.Optimizer) -> OptimizerState:
        buffer = io.BytesIO()
        torch.save(optimizer.state_dict(), buffer)
        return cls(buffer.getvalue())

    def load_into(self, optimizer: torch.optim.Optimizer) -> None:
        loaded = torch.load(
            io.BytesIO(self.payload),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(loaded, Mapping):
            raise TrainingError("optimizer snapshot is not a state dictionary")
        optimizer.load_state_dict(cast(StateDict, loaded))


@dataclass(frozen=True, slots=True)
class RngState:
    cpu: torch.Tensor
    cuda: tuple[torch.Tensor, ...]

    @classmethod
    def capture(cls) -> RngState:
        cuda_states = (
            tuple(state.cpu().clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else ()
        )
        return cls(torch.get_rng_state().cpu().clone(), cuda_states)

    def restore(self) -> None:
        torch.set_rng_state(self.cpu.clone())
        if self.cuda and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.clone() for state in self.cuda])


@dataclass(frozen=True, slots=True)
class SelectedHyperparameters:
    learning_rate: float
    weight_decay: float
    dropout_probability: float


@dataclass(frozen=True, slots=True)
class BaseCheckpoint:
    epoch: int
    valid_macro_cross_entropy: float
    state_dict: ModelParameterState
    optimizer_state: OptimizerState
    rng_state: RngState
    selected_hyperparameters: SelectedHyperparameters
    train_class_weights: ClassWeights

    def restore(self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> None:
        self.state_dict.load_into(model)
        if optimizer is not None:
            self.optimizer_state.load_into(optimizer)
        self.rng_state.restore()


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    checkpoint: BaseCheckpoint
    completed_epochs: int

    @property
    def epoch(self) -> int:
        return self.checkpoint.epoch

    @property
    def valid_macro_cross_entropy(self) -> float:
        return self.checkpoint.valid_macro_cross_entropy


def macro_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    probability_log_floor: float,
) -> float:
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise TrainingError("logits and targets have incompatible shapes")
    if targets.numel() == 0:
        raise TrainingError("macro cross-entropy requires at least one example")
    probabilities = torch.softmax(logits.to(dtype=torch.float32), dim=1)
    selected = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    losses = -torch.log(torch.clamp(selected, min=probability_log_floor)).to(dtype=torch.float64)
    class_losses: list[torch.Tensor] = []
    for class_index in range(int(logits.shape[1])):
        mask = targets == class_index
        if not bool(mask.any()):
            raise TrainingError("VALID split is missing a local class")
        class_losses.append(losses[mask].mean())
    return float(torch.stack(class_losses).mean())


def make_adamw(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    adamw = active_config().scientific.training.adamw
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(adamw.beta1, adamw.beta2),
        eps=adamw.epsilon,
        weight_decay=weight_decay,
        amsgrad=False,
        maximize=False,
        foreach=False,
        fused=False,
    )


def _seed_training_rng(epoch_seed: int, device: torch.device) -> None:
    cpu_state = torch.Generator().manual_seed(epoch_seed).get_state()
    torch.set_rng_state(cpu_state)
    if device.type != "cuda":
        return
    states = [
        torch.Generator(device=torch.device("cuda", index)).manual_seed(epoch_seed).get_state()
        for index in range(torch.cuda.device_count())
    ]
    torch.cuda.set_rng_state_all(states)


def train_base_model(
    model: nn.Module,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    valid_features: torch.Tensor,
    valid_targets: torch.Tensor,
    class_weights: ClassWeights,
    seed: int,
    selected_hyperparameters: SelectedHyperparameters,
) -> TrainingOutcome:
    config = active_config()
    if train_features.ndim != 2 or train_targets.ndim != 1 or train_features.shape[0] == 0:
        raise TrainingError("TRAIN must contain at least one example")
    if valid_features.ndim != 2 or valid_targets.ndim != 1 or valid_features.shape[0] == 0:
        raise TrainingError("VALID must contain at least one example")
    if (
        train_features.shape[0] != train_targets.shape[0]
        or valid_features.shape[0] != valid_targets.shape[0]
    ):
        raise TrainingError("feature and target counts differ")
    training = config.scientific.training
    if training.dataloader_workers != 0:
        raise TrainingError("scientific training requires DataLoader workers = 0")
    if training.label_smoothing != 0.0:
        raise TrainingError("scientific training requires zero label smoothing")

    device = next(model.parameters()).device
    model.to(device=device, dtype=torch.float32)
    optimizer = make_adamw(
        model,
        selected_hyperparameters.learning_rate,
        selected_hyperparameters.weight_decay,
    )
    best: BaseCheckpoint | None = None
    best_metric = math.inf
    epochs_without_improvement = 0
    completed_epochs = 0

    for epoch in range(training.maximum_epochs):
        epoch_seed = derive_seed32(
            SeedDerivationRequest(
                RandomSeed(seed),
                RngNamespace.TRAIN_EPOCH_SHUFFLE,
                cast(StableJsonPayload, OrderedDict(stage="base-training", epoch=epoch)),
            )
        ).value
        generator = torch.Generator().manual_seed(epoch_seed)
        loader = DataLoader(
            TensorDataset(train_features, train_targets),
            batch_size=training.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            drop_last=False,
        )
        caller_rng = RngState.capture()
        try:
            _seed_training_rng(epoch_seed, device)
            model.train()
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                batch_targets = batch_targets.to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=True,
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_features)
                loss = minibatch_objective(
                    logits,
                    batch_targets,
                    class_weights,
                    config.scientific.metrics.probability_log_floor,
                )
                if not bool(torch.isfinite(loss)):
                    raise TrainingError("non-finite TRAIN loss")
                backward_value(loss)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=training.gradient_clip_global_l2_norm,
                    norm_type=2.0,
                )
                optimizer_step(optimizer)

            model.eval()
            with torch.no_grad():
                valid_logits = model(valid_features.to(device=device, dtype=torch.float32))
            valid_metric = macro_cross_entropy(
                valid_logits,
                valid_targets.to(device=device, dtype=torch.long),
                config.scientific.metrics.probability_log_floor,
            )
            completed_epochs = epoch + 1
            improvement = best_metric - valid_metric
            is_new_minimum = valid_metric < best_metric - training.checkpoint.tie_tolerance
            if best is None or is_new_minimum:
                best_metric = valid_metric
                best = BaseCheckpoint(
                    epoch=epoch,
                    valid_macro_cross_entropy=valid_metric,
                    state_dict=ModelParameterState.capture(model),
                    optimizer_state=OptimizerState.capture(optimizer),
                    rng_state=RngState.capture(),
                    selected_hyperparameters=selected_hyperparameters,
                    train_class_weights=ClassWeights(class_weights.values.detach().cpu().clone()),
                )
            if improvement >= training.early_stopping.minimum_improvement:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= training.early_stopping.patience_completed_epochs:
                break
        finally:
            caller_rng.restore()

    if best is None:
        raise TrainingError("training completed without a finite VALID checkpoint")
    best.state_dict.load_into(model)
    model.eval()
    return TrainingOutcome(best, completed_epochs)
