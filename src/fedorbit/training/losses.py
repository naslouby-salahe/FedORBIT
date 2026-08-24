from __future__ import annotations

from dataclasses import dataclass

import torch


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
