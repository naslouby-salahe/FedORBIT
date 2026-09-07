from __future__ import annotations

from pathlib import Path

import pytest

import fedorbit.infrastructure.execution as execution
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.infrastructure.execution import (
    ArtifactStore,
    ExecutionError,
    ExperimentExecutionRequest,
    ReusableArtifactManifest,
    run_experiment,
)
from fedorbit.infrastructure.failures import ProcessCrashError
from fedorbit.infrastructure.workspace import WorkspaceLayout, build_layout
from fedorbit.types import ExperimentName, OverwritePolicy


def _request(
    experiment: ExperimentName = ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION,
) -> ExperimentExecutionRequest:
    definition = build_catalogue().definition(experiment)
    return ExperimentExecutionRequest(experiment, definition, OverwritePolicy.REUSE)


def test_transient_infrastructure_failure_retries_and_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = build_layout(root=tmp_path)
    monkeypatch.setattr(execution, "build_layout", lambda: layout)
    real_producer = execution.execute_primitive_validation
    calls = {"count": 0}

    def flaky_producer(
        store: ArtifactStore, workspace: WorkspaceLayout, overwrite: OverwritePolicy
    ) -> ReusableArtifactManifest:
        calls["count"] += 1
        if calls["count"] < 2:
            raise ProcessCrashError("simulated transient crash")
        return real_producer(store, workspace, overwrite)

    monkeypatch.setattr(execution, "execute_primitive_validation", flaky_producer)
    run_experiment(_request())
    assert calls["count"] == 2


def test_infrastructure_failure_exhausts_retries_and_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = build_layout(root=tmp_path)
    monkeypatch.setattr(execution, "build_layout", lambda: layout)
    calls = {"count": 0}

    def always_crashes(
        _store: ArtifactStore, _workspace: WorkspaceLayout, _overwrite: OverwritePolicy
    ) -> ReusableArtifactManifest:
        calls["count"] += 1
        raise ProcessCrashError("simulated persistent crash")

    monkeypatch.setattr(execution, "execute_primitive_validation", always_crashes)
    with pytest.raises(ExecutionError):
        run_experiment(_request())
    failure_handling = execution.active_config().runtime.failure_handling
    retries = failure_handling.retries_after_initial_infrastructure_failure
    assert calls["count"] == retries + 1


def test_source_response_pilot_dispatches_to_its_producer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = build_layout(root=tmp_path)
    monkeypatch.setattr(execution, "build_layout", lambda: layout)
    calls = {"count": 0}

    def producer(
        _store: ArtifactStore,
        _layout: WorkspaceLayout,
        _request: ExperimentExecutionRequest,
    ) -> None:
        calls["count"] += 1

    monkeypatch.setattr(execution, "execute_source_response_estimator_pilot", producer)
    run_experiment(_request(ExperimentName.SOURCE_RESPONSE_ESTIMATOR_PILOT))
    assert calls["count"] == 1
