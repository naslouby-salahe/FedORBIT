from __future__ import annotations

import pytest

from fedorbit.datasets.splitting import (
    ChronologicalFraction,
    ChronologicalRowCount,
    ChronologicalTimestamp,
    DuplicateGroupChronology,
    DuplicateGroupId,
    DuplicateGroupPosition,
    SplitError,
    assign_duplicate_groups_chronologically,
    duplicate_group_midpoint_fraction,
    interval_edges,
    split_for_duplicate_group,
)
from fedorbit.types import Split


def test_split_intervals_exact_from_config() -> None:
    assert tuple(
        (interval.split, interval.lower.value, interval.upper.value)
        for interval in interval_edges().intervals
    ) == (
        (Split.TRAIN, 0.0, 0.55),
        (Split.META, 0.55, 0.70),
        (Split.VALID, 0.70, 0.80),
        (Split.CONFIRM, 0.80, 0.90),
        (Split.TEST, 0.90, 1.0),
    )


def test_split_boundaries_use_half_open_intervals_except_test_end() -> None:
    assert split_for_duplicate_group(ChronologicalFraction(0.55)) == Split.META
    assert split_for_duplicate_group(ChronologicalFraction(0.70)) == Split.VALID
    assert split_for_duplicate_group(ChronologicalFraction(0.80)) == Split.CONFIRM
    assert split_for_duplicate_group(ChronologicalFraction(0.90)) == Split.TEST
    assert split_for_duplicate_group(ChronologicalFraction(1.00)) == Split.TEST


def test_midpoint_fraction_uses_row_rank_formula() -> None:
    assert duplicate_group_midpoint_fraction(
        DuplicateGroupPosition(
            ChronologicalRowCount(20), ChronologicalRowCount(10), ChronologicalRowCount(100)
        )
    ).value == pytest.approx(0.25)


def test_chronological_assignment_uses_timestamp_then_hash_and_indivisible_groups() -> None:
    assignment = assign_duplicate_groups_chronologically(
        (
            DuplicateGroupChronology(
                DuplicateGroupId("b"), ChronologicalTimestamp(1.0), ChronologicalRowCount(10)
            ),
            DuplicateGroupChronology(
                DuplicateGroupId("a"), ChronologicalTimestamp(1.0), ChronologicalRowCount(50)
            ),
            DuplicateGroupChronology(
                DuplicateGroupId("c"), ChronologicalTimestamp(2.0), ChronologicalRowCount(40)
            ),
        ),
    )
    assert assignment.assignments[0].group_id == DuplicateGroupId("a")
    assert assignment.assignments[1].group_id == DuplicateGroupId("b")
    assert assignment.assignments[2].group_id == DuplicateGroupId("c")
    assert assignment.split_of(DuplicateGroupId("a")) == Split.TRAIN
    assert assignment.split_of(DuplicateGroupId("b")) == Split.META
    assert assignment.split_of(DuplicateGroupId("c")) == Split.CONFIRM


def test_invalid_midpoint_inputs_fail_closed() -> None:
    with pytest.raises(SplitError):
        DuplicateGroupPosition(
            ChronologicalRowCount(5), ChronologicalRowCount(10), ChronologicalRowCount(12)
        )
    with pytest.raises(SplitError):
        ChronologicalFraction(-0.1)
