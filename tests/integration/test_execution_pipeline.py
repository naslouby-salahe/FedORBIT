from __future__ import annotations

from fedorbit.execution.pipeline import plan_command


def test_execution_plan_exposes_complete_registered_catalogue() -> None:
    rows = plan_command()
    assert len(rows) == 27
    assert len({row.experiment for row in rows}) == 27
    assert all(row.layer for row in rows)
