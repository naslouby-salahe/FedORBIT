from __future__ import annotations

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.execution.planner import build_plan


def test_execution_plan_exposes_complete_registered_catalogue() -> None:
    rows = build_plan(load_fedorbit_config())
    assert len(rows) == 27
    assert len({row.experiment for row in rows}) == 27
    assert all(row.classification.value for row in rows)
