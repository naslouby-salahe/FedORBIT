from __future__ import annotations

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import Split


class SplitError(ValueError):
    pass


def interval_edges(config: FedorbitConfig) -> tuple[tuple[Split, float, float], ...]:
    interval = config.scientific.split.duplicate_safe_chronological_intervals
    raw = interval.model_dump(mode="json")
    names = ("train", "meta", "valid", "confirm", "test")
    edges: list[tuple[Split, float, float]] = []
    for name in names:
        lower, upper = raw[name]
        edges.append((Split[name.upper()], float(lower), float(upper)))
    return tuple(edges)


def split_for_duplicate_group(
    config: FedorbitConfig, midpoint_fraction: float, split_seed: int
) -> Split:
    if not 0.0 <= midpoint_fraction <= 1.0:
        raise SplitError(f"midpoint fraction outside [0, 1]: {midpoint_fraction}")
    for split, lower, upper in interval_edges(config):
        if lower <= midpoint_fraction < upper:
            return split
    return Split.TEST


def duplicate_group_midpoint_fraction(
    first_timestamp_fraction: float, last_timestamp_fraction: float
) -> float:
    return (first_timestamp_fraction + last_timestamp_fraction) / 2.0


def assign_duplicate_groups_chronologically(
    config: FedorbitConfig,
    duplicate_group_midpoints: tuple[tuple[str, float], ...],
    split_seed: int,
) -> dict[str, Split]:
    ordered = sorted(duplicate_group_midpoints, key=lambda pair: pair[1])
    return {
        group_id: split_for_duplicate_group(config, midpoint, split_seed)
        for group_id, midpoint in ordered
    }
