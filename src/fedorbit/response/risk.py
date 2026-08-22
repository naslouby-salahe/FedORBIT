from __future__ import annotations

import math

import torch


def native_class_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_index: int,
    probability_log_floor: float,
) -> float:
    class_examples = targets == class_index
    if not bool(class_examples.any()):
        return math.nan
    log_probabilities = torch.log_softmax(logits, dim=1)
    per_example = -log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    ce_cap = -math.log(probability_log_floor)
    per_example = torch.clamp(per_example, max=ce_cap)
    return float(per_example[class_examples].mean())


def equal_native_class_risk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    native_classes: tuple[int, ...],
    probability_log_floor: float,
) -> float:
    risks = tuple(
        native_class_cross_entropy(logits, targets, class_index, probability_log_floor)
        for class_index in native_classes
    )
    if any(math.isnan(risk) for risk in risks):
        return math.nan
    return sum(risks) / len(risks)
