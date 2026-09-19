from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from typer.testing import CliRunner

import fedorbit.cli as cli
from fedorbit.cli import app
from fedorbit.infrastructure.runtime import ReproducibilityIdentity
from fedorbit.infrastructure.workspace import WorkspaceLayout
from fedorbit.types import (
    ExitStatus,
    ExperimentName,
)

runner = CliRunner()


def test_help_lists_only_registered_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "preprocess", "plan", "smoke", "run", "status", "report"):
        assert command in result.output


def test_doctor_reports_recorded_identity_mismatch_without_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mismatched_identity(_layout: WorkspaceLayout) -> ReproducibilityIdentity:
        return cast(ReproducibilityIdentity, SimpleNamespace())

    def incompatible(_current: ReproducibilityIdentity, _recorded: ReproducibilityIdentity) -> bool:
        return False

    monkeypatch.setattr(cli, "recorded_execution_identity", mismatched_identity)
    monkeypatch.setattr(cli, "compatible", incompatible)
    monkeypatch.setattr(cli, "reference_gpu_matches", lambda: True)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "recorded execution identity compatible: False" in result.output


def test_no_scientific_override_options_exist() -> None:
    for option in ("--method", "--seed", "--support", "--budget", "--threshold"):
        result = runner.invoke(app, ["run", "Primary Strict Cross-Telemetry Transfer", option, "x"])
        assert result.exit_code == ExitStatus.USAGE


def test_percentile_95_matches_an_independent_numpy_linear_quantile() -> None:
    import numpy as np

    values = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    expected = float(np.quantile(values, 0.95, method="linear"))
    from fedorbit.experiments.report_rows import percentile_95

    assert percentile_95(values) == pytest.approx(expected)
    assert percentile_95((3.0,)) == 3.0
    assert percentile_95(()) is None


def test_plan_is_read_only_and_derives_catalogue() -> None:
    result = runner.invoke(app, ["plan"])
    assert result.exit_code == 0
    assert f"registered experiments: {len(ExperimentName)}" in result.output
    assert "Primary Strict Cross-Telemetry Transfer" in result.output
    assert "semantic scope:" in result.output
    assert "prerequisites:" in result.output
    assert "resume boundary:" in result.output


def test_report_rejects_invented_experiment() -> None:
    result = runner.invoke(app, ["report", "Invented Experiment"])
    assert result.exit_code == ExitStatus.USAGE


def test_preprocess_rejects_display_name() -> None:
    result = runner.invoke(app, ["preprocess", "Edge-IIoTset"])
    assert result.exit_code == ExitStatus.USAGE
