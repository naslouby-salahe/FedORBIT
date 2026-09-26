from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedorbit.detection.moments import MomentSummary
from fedorbit.types import FloatMatrix, FloatVector


class DegenerateCovarianceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GaussianScorer:
    centre: FloatVector
    scale: FloatVector
    precision: FloatMatrix

    def score(self, rows: FloatMatrix) -> FloatVector:
        standardised = (rows - self.centre) / self.scale
        return ((standardised @ self.precision) * standardised).sum(axis=1)


def shrunk_precision(second_moment: FloatMatrix, effective_count: float) -> FloatMatrix:
    dimension = second_moment.shape[0]
    trace = float(np.trace(second_moment))
    trace_of_square = float((second_moment * second_moment).sum())
    denominator = (effective_count + 1.0 - 2.0 / dimension) * (
        trace_of_square - trace**2 / dimension
    )
    if denominator <= 0.0:
        shrinkage = 1.0
    else:
        numerator = (1.0 - 2.0 / dimension) * trace_of_square + trace**2
        shrinkage = min(1.0, numerator / denominator)
    target = np.eye(dimension) * (trace / dimension)
    shrunk = (1.0 - shrinkage) * second_moment + shrinkage * target
    try:
        return np.linalg.inv(shrunk)
    except np.linalg.LinAlgError as error:
        raise DegenerateCovarianceError("shrunk covariance is singular") from error


def scorer_from_rows(rows: FloatMatrix, centre: FloatVector, scale: FloatVector) -> GaussianScorer:
    standardised = (rows - centre) / scale
    second_moment = standardised.T @ standardised / len(rows)
    return GaussianScorer(
        centre=centre,
        scale=scale,
        precision=shrunk_precision(second_moment, float(len(rows))),
    )


def scorer_from_moments(moments: MomentSummary, minimum_scale: float) -> GaussianScorer:
    scale = moments.standard_deviation + minimum_scale
    second_moment = moments.covariance / np.outer(scale, scale)
    return GaussianScorer(
        centre=moments.mean,
        scale=scale,
        precision=shrunk_precision(second_moment, moments.weight),
    )
