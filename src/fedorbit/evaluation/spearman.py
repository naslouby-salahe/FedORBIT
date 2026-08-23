from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import MetricId


class SpearmanError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SpearmanReport:
    rho: float
    point_count: int
    pair: str


def descriptive_spearman(
    config: FedorbitConfig,
    predicted_values: tuple[float, ...],
    realized_values: tuple[float, ...],
    directed_pair: str,
) -> SpearmanReport | None:
    minimum = config.scientific.statistics.spearman_minimum_valid_points
    if len(predicted_values) != len(realized_values):
        raise SpearmanError("predicted and realized value counts differ")
    if len(predicted_values) < minimum:
        return None
    ranked_predicted = _ranks(predicted_values)
    ranked_realized = _ranks(realized_values)
    rho = _pearson(ranked_predicted, ranked_realized)
    return SpearmanReport(rho=rho, point_count=len(predicted_values), pair=directed_pair)


def _ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks: list[float] = [0.0] * len(values)
    position = 0
    while position < len(order):
        block_end = position
        while (
            block_end + 1 < len(order) and values[order[block_end + 1]] == values[order[position]]
        ):
            block_end += 1
        average_rank = (position + block_end) / 2 + 1
        for offset in range(position, block_end + 1):
            ranks[order[offset]] = average_rank
        position = block_end + 1
    return tuple(ranks)


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    variance_left = sum((a - mean_left) ** 2 for a in left)
    variance_right = sum((b - mean_right) ** 2 for b in right)
    denominator = math.sqrt(variance_left * variance_right)
    if denominator == 0.0:
        return 0.0
    return covariance / denominator


SPEARMAN_METRIC_NAME = MetricId.PREDICTED_REALIZED_SPEARMAN
