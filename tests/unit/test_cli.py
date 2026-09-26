from __future__ import annotations

from typer.testing import CliRunner

from fedorbit.cli import app
from fedorbit.config.loading import load_config
from fedorbit.types import CliCommand, ExperimentId

runner = CliRunner()


def test_help_lists_every_registered_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in CliCommand:
        assert command.value in result.output


def test_plan_declares_every_experiment_with_its_matrix() -> None:
    result = runner.invoke(app, ["plan"])
    assert result.exit_code == 0
    for experiment in load_config().experiments:
        assert experiment.id.value in result.output
        assert f"replicates={experiment.replicates}" in result.output


def test_status_reports_each_experiment_and_the_analysis() -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    for identifier in ExperimentId:
        assert identifier.value in result.output
    assert "analysis:" in result.output


def test_unknown_experiment_names_are_rejected() -> None:
    result = runner.invoke(app, ["run", "not-an-experiment"])
    assert result.exit_code != 0
