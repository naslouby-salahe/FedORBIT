from __future__ import annotations

from dataclasses import dataclass

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import Split


class SplitError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DuplicateGroupChronology:
    group_id: str
    earliest_timestamp: float
    row_count: int

    def __post_init__(self) -> None:
        if not self.group_id:
            raise SplitError("duplicate group identifier must be non-empty")
        if self.row_count <= 0:
            raise SplitError("duplicate group row count must be positive")


@dataclass(frozen=True, slots=True)
class DuplicateGroupSplitAssignment:
    assignments: tuple[tuple[str, Split], ...]

    def split_of(self, group_id: str) -> Split | None:
        for candidate, split in self.assignments:
            if candidate == group_id:
                return split
        return None

    def groups_in_split(self, split: Split) -> tuple[str, ...]:
        return tuple(group for group, assigned in self.assignments if assigned == split)


def interval_edges(config: FedorbitConfig) -> tuple[tuple[Split, float, float], ...]:
    interval = config.scientific.split.duplicate_safe_chronological_intervals
    return (
        (Split.TRAIN, interval.train[0], interval.train[1]),
        (Split.META, interval.meta[0], interval.meta[1]),
        (Split.VALID, interval.valid[0], interval.valid[1]),
        (Split.CONFIRM, interval.confirm[0], interval.confirm[1]),
        (Split.TEST, interval.test[0], interval.test[1]),
    )


def split_for_duplicate_group(config: FedorbitConfig, midpoint_fraction: float) -> Split:
    if not 0.0 <= midpoint_fraction <= 1.0:
        raise SplitError(f"midpoint fraction outside [0, 1]: {midpoint_fraction}")
    for split, lower, upper in interval_edges(config):
        if split == Split.TEST:
            if lower <= midpoint_fraction <= upper:
                return split
        elif lower <= midpoint_fraction < upper:
            return split
    raise SplitError(
        f"midpoint fraction is not covered by configured intervals: {midpoint_fraction}"
    )


def duplicate_group_midpoint_fraction(
    rows_before: int,
    group_row_count: int,
    retained_class_row_count: int,
) -> float:
    if rows_before < 0:
        raise SplitError("rows before duplicate group must be nonnegative")
    if group_row_count <= 0:
        raise SplitError("duplicate group row count must be positive")
    if retained_class_row_count <= 0:
        raise SplitError("retained class row count must be positive")
    if rows_before + group_row_count > retained_class_row_count:
        raise SplitError("duplicate group exceeds retained class row count")
    return (rows_before + 0.5 * group_row_count) / retained_class_row_count


def assign_duplicate_groups_chronologically(
    config: FedorbitConfig,
    duplicate_groups: tuple[DuplicateGroupChronology, ...],
) -> DuplicateGroupSplitAssignment:
    if not duplicate_groups:
        return DuplicateGroupSplitAssignment(())
    ordered = tuple(
        sorted(duplicate_groups, key=lambda item: (item.earliest_timestamp, item.group_id))
    )
    retained_class_row_count = sum(item.row_count for item in ordered)
    rows_before = 0
    assignments: list[tuple[str, Split]] = []
    for item in ordered:
        midpoint = duplicate_group_midpoint_fraction(
            rows_before,
            item.row_count,
            retained_class_row_count,
        )
        assignments.append((item.group_id, split_for_duplicate_group(config, midpoint)))
        rows_before += item.row_count
    return DuplicateGroupSplitAssignment(tuple(assignments))
