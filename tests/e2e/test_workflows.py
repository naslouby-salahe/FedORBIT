from __future__ import annotations

from typer.testing import CliRunner

from fedorbit.cli import app
from fedorbit.types import ExperimentName


def test_cli_plan_workflow() -> None:
    runner = CliRunner()
    plan = runner.invoke(app, ["plan"])
    assert plan.exit_code == 0
    assert str(len(ExperimentName)) in plan.stdout
