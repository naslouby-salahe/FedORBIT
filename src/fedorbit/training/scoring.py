from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from fedorbit.config.context import active_config
from fedorbit.evaluation.metrics import (
    ClassEntropySet,
    CrossEntropy,
    Probability,
    TrueClassProbabilities,
    macro_cross_entropy,
)


class ScoringError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LocalClassCount:
    value: int

    def __post_init__(self) -> None:
        if self.value <= 1:
            raise ScoringError("scoring requires at least two classes")


@dataclass(frozen=True, slots=True)
class ScoreRowIndex:
    value: int


@dataclass(frozen=True, slots=True)
class LocalClassIndex:
    value: int


@dataclass(frozen=True, slots=True)
class ScoreRow:
    row_index: ScoreRowIndex
    target: LocalClassIndex
    predicted_class: LocalClassIndex
    probabilities: TrueClassProbabilities
    cross_entropy: CrossEntropy


@dataclass(frozen=True, slots=True)
class ScoreArtifact:
    rows: tuple[ScoreRow, ...]
    class_conditional_cross_entropy: ClassEntropySet
    macro_cross_entropy: CrossEntropy


@dataclass(frozen=True, slots=True)
class ScoringRequest:
    model: nn.Module
    features: torch.Tensor
    targets: torch.Tensor
    local_class_count: LocalClassCount


def score_model(request: ScoringRequest) -> ScoreArtifact:
    model = request.model
    features = request.features
    targets = request.targets
    n_classes = request.local_class_count.value
    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.shape[0]:
        raise ScoringError("features and targets have incompatible shapes")
    if features.shape[0] == 0:
        raise ScoringError("scoring split must contain at least one example")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device=device, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).to(dtype=torch.float64).cpu()
    target_values = targets.to(dtype=torch.long).cpu()
    if probabilities.shape != (features.shape[0], n_classes):
        raise ScoringError("model output dimension differs from registered local class count")
    selected = probabilities.gather(1, target_values.unsqueeze(1)).squeeze(1)
    log_floor = active_config().scientific.metrics.probability_log_floor
    losses = -torch.log(torch.clamp(selected, min=log_floor))
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
            row_index=ScoreRowIndex(index),
            target=LocalClassIndex(int(target_values[index])),
            predicted_class=LocalClassIndex(int(predictions[index])),
            probabilities=TrueClassProbabilities(
                tuple(Probability(float(value)) for value in probabilities[index])
            ),
            cross_entropy=CrossEntropy(float(losses[index])),
        )
        for index in range(features.shape[0])
    )
    return ScoreArtifact(
        rows=rows,
        class_conditional_cross_entropy=ClassEntropySet(
            tuple(CrossEntropy(value) for value in class_risks)
        ),
        macro_cross_entropy=macro_cross_entropy(
            ClassEntropySet(tuple(CrossEntropy(value) for value in present_risks))
        ),
    )
