from __future__ import annotations

from fedorbit.datasets.common import (
    ChronologyValidationState,
    EventTimeTally,
    inspect_event_time,
)


def test_unix_epoch_event_times_are_valid_chronology() -> None:
    tally = EventTimeTally()
    for value in ("1554206309", "1554206310.25"):
        tally.observe(value)
    observation = inspect_event_time("ts", ("ts",), tally, 0)
    assert observation.state == ChronologyValidationState.VALID
    assert observation.timestamp_pattern_row_count == 2


def test_incomplete_edge_times_remain_ambiguous() -> None:
    tally = EventTimeTally()
    tally.observe("2021 11:44:10.081753000")
    observation = inspect_event_time("frame.time", ("frame.time",), tally, 0)
    assert observation.state == ChronologyValidationState.AMBIGUOUS_EVENT_TIME
