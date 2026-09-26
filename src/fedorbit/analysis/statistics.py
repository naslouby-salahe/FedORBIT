from __future__ import annotations

from itertools import product

import numpy as np

from fedorbit.types import ContrastDirection, FloatVector, RandomSeed

MAXIMUM_EXACT_DEVICES = 20


class InsufficientDevicesError(ValueError):
    pass


def _tail_probabilities(observed: float, null_values: FloatVector) -> tuple[float, float]:
    upper = float(np.mean(null_values >= observed - 1e-12))
    lower = float(np.mean(null_values <= observed + 1e-12))
    return upper, lower


def _directional_p(upper: float, lower: float, direction: ContrastDirection) -> float:
    if direction is ContrastDirection.GREATER:
        return upper
    if direction is ContrastDirection.LESS:
        return lower
    return min(1.0, 2.0 * min(upper, lower))


def _nonzero(deltas: FloatVector) -> FloatVector:
    nonzero = deltas[deltas != 0.0]
    if len(nonzero) > MAXIMUM_EXACT_DEVICES:
        raise InsufficientDevicesError("exact tests support at most 20 devices")
    return nonzero


def exact_sign_test(deltas: FloatVector, direction: ContrastDirection) -> float | None:
    nonzero = _nonzero(deltas)
    count = len(nonzero)
    if count == 0:
        return None
    signs = np.array(list(product((0.0, 1.0), repeat=count))).sum(axis=1)
    upper, lower = _tail_probabilities(float((nonzero > 0).sum()), signs)
    return _directional_p(upper, lower, direction)


def average_ranks(magnitudes: FloatVector) -> FloatVector:
    order = np.argsort(magnitudes, kind="stable")
    ranks = np.empty(len(magnitudes), dtype=np.float64)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and magnitudes[order[end + 1]] == magnitudes[order[position]]:
            end += 1
        ranks[order[position : end + 1]] = (position + end) / 2.0 + 1.0
        position = end + 1
    return ranks


def exact_signed_rank_test(deltas: FloatVector, direction: ContrastDirection) -> float | None:
    nonzero = _nonzero(deltas)
    count = len(nonzero)
    if count == 0:
        return None
    ranks = average_ranks(np.abs(nonzero))
    assignments = np.array(list(product((0.0, 1.0), repeat=count)))
    null_values = assignments @ ranks
    observed = float(ranks[nonzero > 0].sum())
    upper, lower = _tail_probabilities(observed, null_values)
    return _directional_p(upper, lower, direction)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def cluster_bootstrap_interval(
    device_values: FloatVector, resamples: int, confidence: float, seed: RandomSeed
) -> tuple[float, float]:
    if len(device_values) == 0:
        raise InsufficientDevicesError("bootstrap needs at least one device")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(device_values), size=(resamples, len(device_values)))
    means = device_values[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def spearman(first: FloatVector, second: FloatVector) -> float | None:
    if len(first) < 3:
        return None
    first_ranks = average_ranks(first)
    second_ranks = average_ranks(second)
    if first_ranks.std() == 0.0 or second_ranks.std() == 0.0:
        return None
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])
