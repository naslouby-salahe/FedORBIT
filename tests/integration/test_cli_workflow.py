from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fedorbit import cli
from fedorbit.types import EvidenceState, ExperimentId
from tests.support import FIXTURE_DEVICES, synthetic_config, synthetic_nbaiot_directory

runner = CliRunner()


@pytest.fixture
def workspace_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for command in (
        ["git", "init", "-q"],
        [
            "git",
            "-c",
            "user.email=a@b.c",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "x",
        ],
    ):
        subprocess.run(command, cwd=tmp_path, check=True)
    synthetic_nbaiot_directory(tmp_path / "data" / "raw", FIXTURE_DEVICES, seed=9)
    config = synthetic_config(FIXTURE_DEVICES)
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    return tmp_path


def test_doctor_reports_source_files_for_the_configured_dataset(workspace_repository: Path) -> None:
    result = runner.invoke(cli.app, ["doctor"])
    assert "nbaiot: 3 devices with source files present" in result.output
    assert result.exit_code == 1
    assert workspace_repository.is_dir()


def test_preprocess_run_report_and_status_form_a_complete_workflow(
    workspace_repository: Path,
) -> None:
    assert runner.invoke(cli.app, ["preprocess", "nbaiot"]).exit_code == 0
    run = runner.invoke(cli.app, ["run", ExperimentId.COLD_START_LADDER.value])
    assert run.exit_code == 0
    assert EvidenceState.VALID.value in run.output
    report = runner.invoke(cli.app, ["report"])
    assert report.exit_code == 0
    assert (workspace_repository / "results" / "contrasts.csv").is_file()
    status = runner.invoke(cli.app, ["status", ExperimentId.COLD_START_LADDER.value])
    assert f"{ExperimentId.COLD_START_LADDER.value}: valid" in status.output
    assert "analysis: valid" in status.output


def test_report_refuses_to_run_without_evidence(workspace_repository: Path) -> None:
    result = runner.invoke(cli.app, ["report"])
    assert result.exit_code != 0
    assert not (workspace_repository / "results").exists()


def test_smoke_command_completes_on_synthetic_devices(workspace_repository: Path) -> None:
    result = runner.invoke(cli.app, ["smoke"])
    assert result.exit_code == 0, result.output
    assert "smoke contrasts" in result.output
    assert (workspace_repository / "outputs" / "smoke" / "results" / "contrasts.csv").is_file()
