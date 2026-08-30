from __future__ import annotations

import pytest

from fedorbit.datasets.splitting import (
    DuplicateGroupChronology,
    SplitError,
    assign_duplicate_groups_chronologically,
    duplicate_group_midpoint_fraction,
    interval_edges,
    split_for_duplicate_group,
)
from fedorbit.domain.enums import Split


def test_split_intervals_exact_from_config() -> None:
    assert interval_edges() == (
        (Split.TRAIN, 0.0, 0.55),
        (Split.META, 0.55, 0.70),
        (Split.VALID, 0.70, 0.80),
        (Split.CONFIRM, 0.80, 0.90),
        (Split.TEST, 0.90, 1.0),
    )


def test_split_boundaries_use_half_open_intervals_except_test_end() -> None:
    assert split_for_duplicate_group(0.55) == Split.META
    assert split_for_duplicate_group(0.70) == Split.VALID
    assert split_for_duplicate_group(0.80) == Split.CONFIRM
    assert split_for_duplicate_group(0.90) == Split.TEST
    assert split_for_duplicate_group(1.00) == Split.TEST


def test_midpoint_fraction_uses_row_rank_formula() -> None:
    assert duplicate_group_midpoint_fraction(20, 10, 100) == pytest.approx(0.25)


def test_chronological_assignment_uses_timestamp_then_hash_and_indivisible_groups() -> None:
    assignment = assign_duplicate_groups_chronologically(
        (
            DuplicateGroupChronology("b", 1.0, 10),
            DuplicateGroupChronology("a", 1.0, 50),
            DuplicateGroupChronology("c", 2.0, 40),
        ),
    )
    assert assignment.assignments[0][0] == "a"
    assert assignment.assignments[1][0] == "b"
    assert assignment.assignments[2][0] == "c"
    assert assignment.split_of("a") == Split.TRAIN
    assert assignment.split_of("b") == Split.META
    assert assignment.split_of("c") == Split.CONFIRM


def test_invalid_midpoint_inputs_fail_closed() -> None:
    with pytest.raises(SplitError):
        duplicate_group_midpoint_fraction(5, 10, 12)
    with pytest.raises(SplitError):
        split_for_duplicate_group(-0.1)
