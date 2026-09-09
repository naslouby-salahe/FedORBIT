from __future__ import annotations

from fedorbit.datasets.common import (
    ChronologyValidationState,
    EventTimeTally,
    inspect_event_time,
)
from fedorbit.types import RawCellText, TabularColumnName


def test_unix_epoch_event_times_are_valid_chronology() -> None:
    tally = EventTimeTally()
    for value in ("1554206309", "1554206310.25"):
        tally.observe(RawCellText(value))
    observation = inspect_event_time(TabularColumnName("ts"), (TabularColumnName("ts"),), tally, 0)
    assert observation.state == ChronologyValidationState.VALID
    assert observation.timestamp_pattern_row_count == 2


def test_incomplete_edge_times_remain_ambiguous() -> None:
    tally = EventTimeTally()
    tally.observe(RawCellText("2021 11:44:10.081753000"))
    observation = inspect_event_time(
        TabularColumnName("frame.time"), (TabularColumnName("frame.time"),), tally, 0
    )
    assert observation.state == ChronologyValidationState.AMBIGUOUS_EVENT_TIME
