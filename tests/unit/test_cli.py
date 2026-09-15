from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from typer.testing import CliRunner

import fedorbit.cli as cli
from fedorbit.analysis.records import MetricRecord
from fedorbit.cli import (
    CliUsageError,
    app,
    dataset_identifier,
    experiment_identifier,
)
from fedorbit.infrastructure.runtime import ReproducibilityIdentity
from fedorbit.infrastructure.workspace import WorkspaceLayout
from fedorbit.types import (
    ExitStatus,
    ExperimentName,
    MetricId,
    TransferMethod,
)

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


def test_plan_is_read_only_and_derives_catalogue() -> None:
    result = runner.invoke(app, ["plan"])
    assert result.exit_code == 0
    assert f"registered experiments: {len(ExperimentName)}" in result.output
    assert "Primary Strict Cross-Telemetry Transfer" in result.output
    assert "semantic scope:" in result.output
    assert "prerequisites:" in result.output
    assert "resume boundary:" in result.output


def test_predicted_vs_realized_series_reports_spearman_point_count_or_unavailable() -> None:
    pair = "source -> target"

    def record(metric: MetricId, seed: int, value: float) -> MetricRecord:
        return cast(
            MetricRecord,
            SimpleNamespace(
                valid=True,
                metric_value=value,
                pair=pair,
                seed=seed,
                method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                metric_name=metric,
            ),
        )

    insufficient = cli.predicted_vs_realized_series(
        (
            record(MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE, 1, 0.1),
            record(MetricId.RELATIVE_MACRO_CE_GAIN, 1, 0.2),
        )
    )
    eligible = cli.predicted_vs_realized_series(
        tuple(
            item
            for seed in range(5)
            for item in (
                record(MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE, seed, float(seed)),
                record(MetricId.RELATIVE_MACRO_CE_GAIN, seed, float(seed)),
            )
        )
    )

    assert insufficient[0].name == "source -> target | Spearman unavailable; n=1"
    assert "rho=" in eligible[0].name
    assert "n=5" in eligible[0].name


def test_report_rejects_invented_experiment() -> None:
    result = runner.invoke(app, ["report", "Invented Experiment"])
    assert result.exit_code == ExitStatus.USAGE


def test_preprocess_rejects_display_name() -> None:
    result = runner.invoke(app, ["preprocess", "Edge-IIoTset"])
    assert result.exit_code == ExitStatus.USAGE
