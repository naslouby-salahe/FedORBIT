from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fedorbit.types import FloatMatrix, FloatVector


@dataclass(frozen=True, slots=True)
class MomentSummary:
    weight: float
    mean: FloatVector
    covariance: FloatMatrix

    @property
    def standard_deviation(self) -> FloatVector:
        return np.sqrt(np.diag(self.covariance))


def summarise(rows: FloatMatrix) -> MomentSummary:
    centred = rows - rows.mean(axis=0)
    return MomentSummary(
        weight=float(len(rows)),
        mean=rows.mean(axis=0),
        covariance=centred.T @ centred / len(rows),
    )


def combine(components: Sequence[tuple[float, MomentSummary]]) -> MomentSummary:
    weights = np.array([weight for weight, _ in components], dtype=np.float64)
    means = np.stack([summary.mean for _, summary in components])
    total = float(weights.sum())
    mean = weights @ means / total
    offsets = means - mean
    covariances = np.stack([summary.covariance for _, summary in components])
    between = np.einsum("k,ki,kj->ij", weights, offsets, offsets)
    covariance = (np.einsum("k,kij->ij", weights, covariances) + between) / total
    return MomentSummary(weight=total, mean=mean, covariance=covariance)


def aggregate_equal_weight(summaries: Sequence[MomentSummary]) -> MomentSummary:
    return combine([(1.0, summary) for summary in summaries])


def pool_with_partners(
    local: MomentSummary, partners: MomentSummary, partner_weight: float
) -> MomentSummary:
    return combine([(local.weight, local), (partner_weight, partners)])
