from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.config.loading import active_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.experiments import solvers as experiment_solvers
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.experiments.solvers import (
    UnresolvedMapWorldUnavailability,
    execute_common_action_under_unidentified_map,
    persist_map_world_metrics,
    persist_unavailable_map_world_metrics,
)
from fedorbit.experiments.synthesis import completed_experiment_metric_records
from fedorbit.experiments.synthetic import (
    MechanismGenerationError,
    UnresolvedMapWorldKind,
    UnresolvedMapWorldRequest,
    generate_unresolved_map_world,
)
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactState,
    ExperimentName,
    InvalidReason,
    MetricId,
    OverwritePolicy,
)

_UNAVAILABLE_REASON = InvalidReason(
    f"{UnresolvedMapWorldUnavailability.FIXTURE_NOT_CONSTRUCTIBLE.value}: fixture exhausted"
)


def _request(experiment: ExperimentName) -> ExperimentExecutionRequest:
    catalogue = build_catalogue()
    return ExperimentExecutionRequest(
        experiment=experiment,
        definition=catalogue.definition(experiment),
        overwrite_policy=OverwritePolicy.REPLACE,
    )


def _single_fixture_config() -> FedorbitConfig:
    config = active_config()
    experiments = config.experiments
    audit_section = experiments.common_action_under_unidentified_map
    return config.model_copy(
        update={
            "scientific": config.scientific.model_copy(
                update={
                    "randomness": config.scientific.randomness.model_copy(
                        update={"confirmatory_seeds": (1103,)}
                    )
                }
            ),
            "experiments": experiments.model_copy(
                update={
                    "common_action_under_unidentified_map": audit_section.model_copy(
                        update={"fixtures_per_seed": 1}
                    )
                }
            ),
        }
    )


def test_successful_and_unavailable_fixtures_persist_typed_states(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    experiment = ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP
    world = generate_unresolved_map_world(
        UnresolvedMapWorldRequest(UnresolvedMapWorldKind.COMMON_ACTION, 1103)
    )
    persist_map_world_metrics(
        store,
        layout,
        experiment,
        UnresolvedMapWorldKind.COMMON_ACTION,
        0,
        1103,
        world,
        OverwritePolicy.REPLACE,
    )
    persist_unavailable_map_world_metrics(
        store,
        layout,
        experiment,
        UnresolvedMapWorldKind.COMMON_ACTION,
        1,
        1103,
        _UNAVAILABLE_REASON,
        OverwritePolicy.REPLACE,
    )
    records = completed_experiment_metric_records(store, experiment)
    assert len(records) == 6
    metric_ids = {
        MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE,
        MetricId.EXACT_MAP_ACTION_VALUE,
        MetricId.ORBIT_RADIUS_MAP_BOUND,
    }
    available = tuple(record for record in records if record.condition.endswith("-0"))
    unavailable = tuple(record for record in records if record.condition.endswith("-1"))
    assert {record.metric_name for record in available} == metric_ids
    assert {record.metric_name for record in unavailable} == metric_ids
    assert all(record.valid and record.metric_value is not None for record in available)
    assert all(
        not record.valid and record.metric_value is None and record.invalid_reason is not None
        for record in unavailable
    )
    for record in available:
        assert record.input_artifact_ids == (ArtifactIdentifier("synthetic-generator"),)
    for record in unavailable:
        assert record.invalid_reason == _UNAVAILABLE_REASON
        assert record.input_artifact_ids == (ArtifactIdentifier("unavailable-synthetic-cell"),)
    for manifest in store.all_manifests():
        assert store.resolve(manifest.artifact_id).state == ArtifactState.COMPLETED


def test_unresolved_map_producer_persists_unavailability_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = _request(ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP)
    config = _single_fixture_config()
    confirmatory_seeds = config.scientific.randomness.confirmatory_seeds
    fixtures_per_seed = config.experiments.common_action_under_unidentified_map.fixtures_per_seed

    def infeasible(_request: object) -> None:
        raise MechanismGenerationError("registered fixture search exhausted its attempts")

    monkeypatch.setattr(experiment_solvers, "active_config", lambda: config)
    monkeypatch.setattr(experiment_solvers, "generate_unresolved_map_world", infeasible)
    execute_common_action_under_unidentified_map(store, layout, request)

    records = completed_experiment_metric_records(
        store, ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP
    )
    assert len(records) == len(confirmatory_seeds) * fixtures_per_seed * 3
    assert {record.seed for record in records} == set(confirmatory_seeds)
    for record in records:
        assert not record.valid
        assert record.metric_value is None
        assert record.invalid_reason is not None
        assert record.invalid_reason.startswith(
            UnresolvedMapWorldUnavailability.FIXTURE_NOT_CONSTRUCTIBLE.value
        )
        assert "registered fixture search exhausted its attempts" in record.invalid_reason
