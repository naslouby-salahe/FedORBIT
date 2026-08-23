from __future__ import annotations

import io
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import RngNamespace
from fedorbit.runtime.seeds import derive_seed32
from fedorbit.training.losses import ClassWeights, minibatch_objective


class TrainingError(ValueError):
    pass


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
            {entry.name: entry.value.clone() for entry in self.tensors}, strict=True
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
        state = torch.load(io.BytesIO(self.payload), map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TrainingError("optimizer snapshot is not a state dictionary")
        optimizer.load_state_dict(state)


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
    for class_index in torch.unique(targets, sorted=True):
        mask = targets == class_index
        class_losses.append(losses[mask].mean())
    return float(torch.stack(class_losses).mean())


def make_adamw(
    config: FedorbitConfig,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    adamw = config.scientific.training.adamw
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


def train_base_model(
    config: FedorbitConfig,
    model: nn.Module,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    valid_features: torch.Tensor,
    valid_targets: torch.Tensor,
    class_weights: ClassWeights,
    seed: int,
    selected_hyperparameters: SelectedHyperparameters,
) -> TrainingOutcome:
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
        config,
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
            seed,
            RngNamespace.TRAIN_EPOCH_SHUFFLE,
            {"stage": "base-training", "epoch": epoch},
        )
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
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(epoch_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(epoch_seed)
            model.train()
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                batch_targets = batch_targets.to(device=device, dtype=torch.long, non_blocking=True)
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
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=training.gradient_clip_global_l2_norm,
                    norm_type=2.0,
                )
                optimizer.step()

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

    if best is None:
        raise TrainingError("training completed without a finite VALID checkpoint")
    best.state_dict.load_into(model)
    model.eval()
    return TrainingOutcome(best, completed_epochs)
