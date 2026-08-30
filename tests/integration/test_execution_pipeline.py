from __future__ import annotations

from fedorbit.execution.planner import build_plan


def test_execution_plan_exposes_complete_registered_catalogue() -> None:
    rows = build_plan()
    assert len(rows) == 26
    assert len({row.experiment for row in rows}) == 26
    assert all(row.classification.value for row in rows)
