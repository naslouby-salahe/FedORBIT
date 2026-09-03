from __future__ import annotations

import pytest
from typer.testing import CliRunner

from fedorbit.cli import (
    EXIT_OK,
    EXIT_USAGE,
    CliUsageError,
    app,
    dataset_identifier,
    experiment_identifier,
)
from fedorbit.types import ExperimentName

runner = CliRunner()

REGISTERED_DATASET_IDS = (
    "edge_iiotset_network",
    "ton_iot_windows10_host",
    "ton_iot_linux_process_host",
    "ton_iot_network",
)


@pytest.mark.parametrize("identifier", REGISTERED_DATASET_IDS)
def test_registered_dataset_identifiers_accepted(identifier: str) -> None:
    assert dataset_identifier(identifier).value == identifier


@pytest.mark.parametrize(
    "rejected",
    (
        "Edge-IIoTset",
        "ToN-IoT",
        "edge_iiotset",
        "EDGE_IIOTSET_NETWORK",
        "ton-iot-network",
        "Edge IIoTset Network",
    ),
)
def test_non_identifier_names_rejected(rejected: str) -> None:
    with pytest.raises(CliUsageError):
        dataset_identifier(rejected)


def test_case_sensitive_matching() -> None:
    with pytest.raises(CliUsageError):
        dataset_identifier("EDGE_IIOTSET_NETWORK")


def test_registered_experiment_names_accepted() -> None:
    assert experiment_identifier("Primary Strict Cross-Telemetry Transfer") == (
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER
    )


def test_invented_experiment_name_rejected() -> None:
    with pytest.raises(CliUsageError):
        experiment_identifier("Invented Experiment")


def test_help_lists_only_registered_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "preprocess", "plan", "smoke", "run", "status", "report"):
        assert command in result.output


def test_no_scientific_override_options_exist() -> None:
    for option in ("--method", "--seed", "--support", "--budget", "--threshold"):
        result = runner.invoke(app, ["run", "Primary Strict Cross-Telemetry Transfer", option, "x"])
        assert result.exit_code == EXIT_USAGE


def test_doctor_is_read_only_and_validates_environment() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)
    assert "python:" in result.output or "environment mismatch" in result.output
    assert "lockfile" in result.output


def test_plan_is_read_only_and_derives_catalogue() -> None:
    result = runner.invoke(app, ["plan"])
    assert result.exit_code == 0
    assert "registered experiments: 26" in result.output
    assert "Primary Strict Cross-Telemetry Transfer" in result.output


def test_status_renders_wide_table_per_item() -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Experiment" in result.output
    assert "Primary Strict Cross-Telemetry Transfer" in result.output


def test_status_accepts_single_experiment() -> None:
    result = runner.invoke(app, ["status", "Primary Strict Cross-Telemetry Transfer"])
    assert result.exit_code == 0


def test_status_rejects_invented_experiment() -> None:
    result = runner.invoke(app, ["status", "Invented Experiment"])
    assert result.exit_code == EXIT_USAGE


def test_report_rejects_invented_experiment() -> None:
    result = runner.invoke(app, ["report", "Invented Experiment"])
    assert result.exit_code == EXIT_USAGE


def test_preprocess_accepts_registered_dataset() -> None:
    result = runner.invoke(app, ["preprocess", "edge_iiotset_network"])
    assert result.exit_code == 0


def test_preprocess_rejects_display_name() -> None:
    result = runner.invoke(app, ["preprocess", "Edge-IIoTset"])
    assert result.exit_code == EXIT_USAGE


def test_preprocess_accepts_overwrite_flag() -> None:
    result = runner.invoke(app, ["preprocess", "--overwrite"])
    assert result.exit_code == 0


def test_run_accepts_registered_experiment() -> None:
    result = runner.invoke(app, ["run", "Mathematical Primitive Validation"])
    assert result.exit_code == EXIT_OK


def test_smoke_accepts_overwrite_flag() -> None:
    result = runner.invoke(app, ["smoke", "--overwrite"])
    assert result.exit_code == 0
