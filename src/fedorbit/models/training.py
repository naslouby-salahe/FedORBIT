from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import RngNamespace
from fedorbit.runtime.seeds import derive_seed32


class TrainingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BaseCheckpoint:
    epoch: int
    valid_macro_cross_entropy: float
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    epoch: int
    valid_macro_cross_entropy: float
    checkpoint: BaseCheckpoint


def macro_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    numerical_floor: float,
) -> float:
    probabilities = torch.softmax(logits, dim=1)
    one_hot = torch.zeros_like(probabilities)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
    per_class_ce = -(one_hot * torch.log(probabilities + numerical_floor)).sum(dim=0)
    class_present = one_hot.sum(dim=0)
    present = class_present > 0
    if not bool(present.any()):
        return float("nan")
    weighted = (per_class_ce[present] / class_present[present].clamp(min=1)) * class_weights[
        present
    ]
    return float(weighted.mean())


def weighted_mean_loss(
    loss_per_example: torch.Tensor, class_weights: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    per_example_weights = class_weights[targets]
    return (loss_per_example * per_example_weights).mean()


def train_base_model(
    config: FedorbitConfig,
    model: nn.Module,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    valid_features: torch.Tensor,
    valid_targets: torch.Tensor,
    class_weights: torch.Tensor,
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainingOutcome:
    training = config.scientific.training
    adamw = training.adamw
    coordinates = {"experiment": "base-training", "seed": seed}
    shuffle_rng = torch.Generator().manual_seed(
        derive_seed32(seed, RngNamespace.TRAIN_EPOCH_SHUFFLE, coordinates)
    )
    batch_size = training.batch_size
    if train_features.shape[0] == 0:
        raise TrainingError("TRAIN set is empty")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(adamw.beta1, adamw.beta2),
        eps=adamw.epsilon,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=training.label_smoothing)
    best_epoch = -1
    best_metric = float("inf")
    epochs_without_improvement = 0
    for epoch in range(training.maximum_epochs):
        model.train()
        permutation = torch.randperm(train_features.shape[0], generator=shuffle_rng)
        for start in range(0, train_features.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            model.zero_grad()
            logits = model(train_features[indices].float())
            loss_per_example = criterion(logits, train_targets[indices])
            loss = weighted_mean_loss(loss_per_example, class_weights, train_targets[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_global_l2_norm)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_features.float())
            valid_metric = macro_cross_entropy(
                valid_logits,
                valid_targets,
                class_weights,
                config.scientific.source_response_pilot.numerical_floor,
            )

        if valid_metric < best_metric - training.checkpoint.tie_tolerance:
            best_metric = valid_metric
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training.early_stopping.patience_completed_epochs:
                break

    if best_epoch < 0:
        raise TrainingError("no VALID improvement observed; no checkpoint selected")
    return TrainingOutcome(
        epoch=best_epoch,
        valid_macro_cross_entropy=best_metric,
        checkpoint=BaseCheckpoint(
            epoch=best_epoch,
            valid_macro_cross_entropy=best_metric,
            state_dict={key: value.detach().clone() for key, value in model.state_dict().items()},
        ),
    )
