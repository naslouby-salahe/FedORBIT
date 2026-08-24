from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pytest

from fedorbit.config.models import FedorbitConfig
from fedorbit.runtime.environment import (
    DEPENDENCY_SPECS,
    environment_snapshot,
    reference_gpu_matches,
    validate_environment,
    validate_lockfile,
)


def test_dependency_versions_match_configured_contract(fedorbit_config: FedorbitConfig) -> None:
    snapshot = environment_snapshot(fedorbit_config)
    assert len(snapshot.dependencies) == len(DEPENDENCY_SPECS)
    for dependency in snapshot.dependencies:
        assert dependency.observed == importlib.metadata.version(dependency.distribution)
        assert dependency.observed == dependency.configured, dependency.configured_key


def test_snapshot_records_python_version(fedorbit_config: FedorbitConfig) -> None:
    import platform

    snapshot = environment_snapshot(fedorbit_config)
    assert snapshot.python_version == platform.python_version()
    assert snapshot.python_version == fedorbit_config.environment.python


def test_strict_validation_accepts_registered_software_environment(
    fedorbit_config: FedorbitConfig,
) -> None:
    snapshot = validate_environment(fedorbit_config, strict=True)
    assert snapshot.python_version == fedorbit_config.environment.python
    assert not snapshot.mismatches()


def test_non_strict_validation_returns_snapshot(fedorbit_config: FedorbitConfig) -> None:
    snapshot = validate_environment(fedorbit_config, strict=False)
    assert snapshot.python_version


def test_lockfile_validates_hashes_and_versions(fedorbit_config: FedorbitConfig) -> None:
    summary = validate_lockfile(fedorbit_config)
    assert summary.all_packages_hashed
    assert len(summary.package_names) >= 20
    assert "torch" in summary.package_names
    assert "typer" in summary.package_names


def test_lockfile_missing_raises(
    fedorbit_config: FedorbitConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fedorbit.runtime.environment.repository_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        validate_lockfile(fedorbit_config)


def test_reference_gpu_probe_is_boolean(fedorbit_config: FedorbitConfig) -> None:
    assert isinstance(reference_gpu_matches(fedorbit_config), bool)


def test_hardware_identity_records_available_host_capabilities(
    fedorbit_config: FedorbitConfig,
) -> None:
    hardware = environment_snapshot(fedorbit_config).hardware
    if hardware.cuda_available:
        assert hardware.gpu_name is not None
        assert hardware.gpu_memory_bytes is not None
    else:
        assert hardware.gpu_name is None
        assert hardware.gpu_memory_bytes is None
    assert hardware.ram_bytes > 0
    assert hardware.os_release


def test_fingerprint_is_deterministic(fedorbit_config: FedorbitConfig) -> None:
    first = environment_snapshot(fedorbit_config).fingerprint_sha256
    second = environment_snapshot(fedorbit_config).fingerprint_sha256
    assert first == second
    assert len(first) == 64


def test_fingerprint_changes_with_dependency_version(
    fedorbit_config: FedorbitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = environment_snapshot(fedorbit_config).fingerprint_sha256
    original_version = importlib.metadata.version

    def altered_version(distribution: str) -> str:
        if distribution == "numpy":
            return "9.9.9"
        return original_version(distribution)

    monkeypatch.setattr(importlib.metadata, "version", altered_version)
    altered = environment_snapshot(fedorbit_config).fingerprint_sha256
    assert altered != baseline


def test_environment_snapshot_is_json_serializable(fedorbit_config: FedorbitConfig) -> None:
    snapshot = environment_snapshot(fedorbit_config)
    payload = {
        "python_version": snapshot.python_version,
        "fingerprint_sha256": snapshot.fingerprint_sha256,
    }
    rendered = json.dumps(payload, sort_keys=True)
    assert json.loads(rendered)["fingerprint_sha256"] == snapshot.fingerprint_sha256


def test_reporting_precision_matches_registered_values(fedorbit_config: FedorbitConfig) -> None:
    precision = fedorbit_config.reporting.precision
    assert precision.scientific_metric_decimals == 4
    assert precision.macro_f1_decimals == 4
    assert precision.balanced_accuracy_decimals == 4
    assert precision.p_value_decimals == 4
    assert precision.p_value_less_than_threshold == 0.0001
    assert precision.runtime_seconds_decimals == 3
    assert precision.memory_decimals == 1


def test_runtime_resource_limits_registered(fedorbit_config: FedorbitConfig) -> None:
    runtime = fedorbit_config.runtime
    assert runtime.solver_cpu_worker_ceiling == 4
    assert runtime.host_ram_ceiling_gib_for_registered_efficiency_runs == 16
    assert runtime.deterministic_kernel_warmups == 3
    assert runtime.deterministic_kernel_timed_repetitions == 10
    assert runtime.full_training_timing_repetitions_per_scientific_cell == 1
