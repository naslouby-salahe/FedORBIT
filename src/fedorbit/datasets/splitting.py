from __future__ import annotations

from dataclasses import dataclass

from fedorbit.config.context import active_config
from fedorbit.domain.enums import Split


class SplitError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DuplicateGroupId:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise SplitError("duplicate group identifier must be non-empty")


@dataclass(frozen=True, slots=True)
class ChronologicalFraction:
    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise SplitError(f"chronological fraction outside [0, 1]: {self.value}")


@dataclass(frozen=True, slots=True)
class ChronologicalTimestamp:
    value: float


@dataclass(frozen=True, slots=True)
class ChronologicalRowCount:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise SplitError("chronological row count must be nonnegative")


@dataclass(frozen=True, slots=True)
class SplitInterval:
    split: Split
    lower: ChronologicalFraction
    upper: ChronologicalFraction

    def __post_init__(self) -> None:
        if self.lower.value > self.upper.value:
            raise SplitError("split interval lower boundary exceeds upper boundary")

    def contains(self, fraction: ChronologicalFraction) -> bool:
        if self.split == Split.TEST:
            return self.lower.value <= fraction.value <= self.upper.value
        return self.lower.value <= fraction.value < self.upper.value


@dataclass(frozen=True, slots=True)
class SplitIntervalSchedule:
    intervals: tuple[SplitInterval, ...]

    def split_for(self, fraction: ChronologicalFraction) -> Split:
        for interval in self.intervals:
            if interval.contains(fraction):
                return interval.split
        raise SplitError(
            f"midpoint fraction is not covered by configured intervals: {fraction.value}"
        )


@dataclass(frozen=True, slots=True)
class DuplicateGroupPosition:
    rows_before: ChronologicalRowCount
    group_row_count: ChronologicalRowCount
    retained_class_row_count: ChronologicalRowCount

    def __post_init__(self) -> None:
        if self.group_row_count.value < 1:
            raise SplitError("duplicate group row count must be positive")
        if self.retained_class_row_count.value < 1:
            raise SplitError("retained class row count must be positive")
        if (
            self.rows_before.value + self.group_row_count.value
            > self.retained_class_row_count.value
        ):
            raise SplitError("duplicate group exceeds retained class row count")


@dataclass(frozen=True, slots=True)
class DuplicateGroupChronology:
    group_id: DuplicateGroupId
    earliest_timestamp: ChronologicalTimestamp
    row_count: ChronologicalRowCount

    def __post_init__(self) -> None:
        if self.row_count.value <= 0:
            raise SplitError("duplicate group row count must be positive")


@dataclass(frozen=True, slots=True)
class DuplicateGroupSplit:
    group_id: DuplicateGroupId
    split: Split


@dataclass(frozen=True, slots=True)
class DuplicateGroupSplitAssignment:
    assignments: tuple[DuplicateGroupSplit, ...]

    def split_of(self, group_id: DuplicateGroupId) -> Split | None:
        for assignment in self.assignments:
            if assignment.group_id == group_id:
                return assignment.split
        return None

    def groups_in_split(self, split: Split) -> tuple[DuplicateGroupId, ...]:
        return tuple(
            assignment.group_id for assignment in self.assignments if assignment.split == split
        )


def interval_edges() -> SplitIntervalSchedule:
    interval = active_config().scientific.split.duplicate_safe_chronological_intervals
    intervals = (
        SplitInterval(
            Split.TRAIN,
            ChronologicalFraction(interval.train[0]),
            ChronologicalFraction(interval.train[1]),
        ),
        SplitInterval(
            Split.META,
            ChronologicalFraction(interval.meta[0]),
            ChronologicalFraction(interval.meta[1]),
        ),
        SplitInterval(
            Split.VALID,
            ChronologicalFraction(interval.valid[0]),
            ChronologicalFraction(interval.valid[1]),
        ),
        SplitInterval(
            Split.CONFIRM,
            ChronologicalFraction(interval.confirm[0]),
            ChronologicalFraction(interval.confirm[1]),
        ),
        SplitInterval(
            Split.TEST,
            ChronologicalFraction(interval.test[0]),
            ChronologicalFraction(interval.test[1]),
        ),
    )
    return SplitIntervalSchedule(intervals)


def split_for_duplicate_group(midpoint_fraction: ChronologicalFraction) -> Split:
    return interval_edges().split_for(midpoint_fraction)


def duplicate_group_midpoint_fraction(position: DuplicateGroupPosition) -> ChronologicalFraction:
    return ChronologicalFraction(
        (position.rows_before.value + position.group_row_count.value / 2)
        / position.retained_class_row_count.value
    )


def assign_duplicate_groups_chronologically(
    duplicate_groups: tuple[DuplicateGroupChronology, ...],
) -> DuplicateGroupSplitAssignment:
    if not duplicate_groups:
        return DuplicateGroupSplitAssignment(())
    ordered = tuple(
        sorted(
            duplicate_groups,
            key=lambda item: (item.earliest_timestamp.value, item.group_id.value),
        )
    )
    retained_class_row_count = ChronologicalRowCount(sum(item.row_count.value for item in ordered))
    rows_before = ChronologicalRowCount(0)
    assignments: list[DuplicateGroupSplit] = []
    for item in ordered:
        midpoint = duplicate_group_midpoint_fraction(
            DuplicateGroupPosition(
                rows_before,
                item.row_count,
                retained_class_row_count,
            )
        )
        assignments.append(DuplicateGroupSplit(item.group_id, split_for_duplicate_group(midpoint)))
        rows_before = ChronologicalRowCount(rows_before.value + item.row_count.value)
    return DuplicateGroupSplitAssignment(tuple(assignments))
