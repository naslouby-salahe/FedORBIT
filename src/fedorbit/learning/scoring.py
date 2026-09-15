from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fedorbit.analysis.metrics import (
    ClassEntropySet,
    CrossEntropy,
    InvalidEvaluationDataError,
    Probability,
    TrueClassProbabilities,
    class_conditional_cross_entropy,
    example_cross_entropy,
    macro_cross_entropy,
)
from fedorbit.types import ClassIndex, ConceptCount, Index


class ScoringError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LocalClassCount:
    value: ConceptCount

    def __post_init__(self) -> None:
        if self.value <= 1:
            raise ScoringError("scoring requires at least two classes")


@dataclass(frozen=True, slots=True)
class ScoreRowIndex:
    value: Index


@dataclass(frozen=True, slots=True)
class LocalClassIndex:
    value: Index


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
    predictions = probabilities.argmax(dim=1)
    row_probabilities = tuple(
        TrueClassProbabilities(tuple(Probability(float(value)) for value in probabilities[index]))
        for index in range(features.shape[0])
    )
    true_class_probabilities = tuple(
        Probability(float(value))
        for value in probabilities.gather(1, target_values.unsqueeze(1)).squeeze(1)
    )
    membership = tuple(int(value) for value in target_values)
    class_entropies: list[CrossEntropy] = []
    for class_index in range(n_classes):
        members = tuple(
            probability
            for probability, target_class in zip(true_class_probabilities, membership, strict=True)
            if target_class == class_index
        )
        if not members:
            raise InvalidEvaluationDataError(ClassIndex(class_index))
        class_entropies.append(class_conditional_cross_entropy(TrueClassProbabilities(members)))
    entropy_set = ClassEntropySet(tuple(class_entropies))
    rows = tuple(
        ScoreRow(
            row_index=ScoreRowIndex(index),
            target=LocalClassIndex(int(target_values[index])),
            predicted_class=LocalClassIndex(int(predictions[index])),
            probabilities=row_probabilities[index],
            cross_entropy=example_cross_entropy(true_class_probabilities[index]),
        )
        for index in range(features.shape[0])
    )
    return ScoreArtifact(
        rows=rows,
        class_conditional_cross_entropy=entropy_set,
        macro_cross_entropy=macro_cross_entropy(entropy_set),
    )
