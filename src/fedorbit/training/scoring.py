from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class ScoringError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ScoreRow:
    row_index: int
    target: int
    predicted_class: int
    probabilities: tuple[float, ...]
    cross_entropy: float


@dataclass(frozen=True, slots=True)
class ScoreArtifact:
    rows: tuple[ScoreRow, ...]
    class_conditional_cross_entropy: tuple[float, ...]
    macro_cross_entropy: float


def score_model(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    probability_log_floor: float,
) -> ScoreArtifact:
    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.shape[0]:
        raise ScoringError("features and targets have incompatible shapes")
    if features.shape[0] == 0:
        raise ScoringError("scoring split must contain at least one example")
    if n_classes <= 1:
        raise ScoringError("scoring requires at least two classes")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device=device, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).to(dtype=torch.float64).cpu()
    target_values = targets.to(dtype=torch.long).cpu()
    if probabilities.shape != (features.shape[0], n_classes):
        raise ScoringError("model output dimension differs from registered local class count")
    selected = probabilities.gather(1, target_values.unsqueeze(1)).squeeze(1)
    losses = -torch.log(torch.clamp(selected, min=probability_log_floor))
    predictions = probabilities.argmax(dim=1)
    class_risks: list[float] = []
    for class_index in range(n_classes):
        mask = target_values == class_index
        class_risks.append(float(losses[mask].mean()) if bool(mask.any()) else math.nan)
    present_risks = tuple(value for value in class_risks if math.isfinite(value))
    if not present_risks:
        raise ScoringError("no evaluation classes are present")
    rows = tuple(
        ScoreRow(
            row_index=index,
            target=int(target_values[index]),
            predicted_class=int(predictions[index]),
            probabilities=tuple(float(value) for value in probabilities[index]),
            cross_entropy=float(losses[index]),
        )
        for index in range(features.shape[0])
    )
    return ScoreArtifact(
        rows=rows,
        class_conditional_cross_entropy=tuple(class_risks),
        macro_cross_entropy=sum(present_risks) / len(present_risks),
    )
