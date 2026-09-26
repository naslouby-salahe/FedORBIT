from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fedorbit.types import FloatVector


class UndefinedMetricError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OperatingPointMetrics:
    false_positive_rate: float
    true_positive_rate: float


def auroc(benign_scores: FloatVector, attack_scores: FloatVector) -> float:
    if len(benign_scores) == 0 or len(attack_scores) == 0:
        raise UndefinedMetricError("AUROC needs both benign and attack scores")
    ordered = np.sort(benign_scores)
    below = np.searchsorted(ordered, attack_scores, side="left")
    at_or_below = np.searchsorted(ordered, attack_scores, side="right")
    wins = below.sum() + 0.5 * (at_or_below - below).sum()
    return float(wins / (len(benign_scores) * len(attack_scores)))


def operating_point(
    threshold: float, benign_scores: FloatVector, attack_scores: FloatVector
) -> OperatingPointMetrics:
    return OperatingPointMetrics(
        false_positive_rate=float(np.mean(benign_scores > threshold)),
        true_positive_rate=float(np.mean(attack_scores > threshold)),
    )


def threshold_at_rate(held_out_scores: FloatVector, nominal_false_positive_rate: float) -> float:
    return float(np.quantile(held_out_scores, 1.0 - nominal_false_positive_rate))


def worst_case_auroc_standard_error(benign_count: int, attack_count: int) -> float:
    if benign_count == 0 or attack_count == 0:
        raise UndefinedMetricError("standard error needs both benign and attack rows")
    return math.sqrt((benign_count + attack_count + 1) / (12.0 * benign_count * attack_count))
