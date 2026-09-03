from __future__ import annotations

from typer.testing import CliRunner

from fedorbit.cli import app


def test_cli_plan_and_status_workflow() -> None:
    runner = CliRunner()
    plan = runner.invoke(app, ["plan"])
    assert plan.exit_code == 0
    assert "26" in plan.stdout
    status = runner.invoke(app, ["status"])
    assert status.exit_code == 0
    assert "Experiment" in status.stdout
