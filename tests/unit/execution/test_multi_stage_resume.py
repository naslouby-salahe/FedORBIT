from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fedorbit.config.loading import active_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.common import file_sha256
from fedorbit.experiments import validation as experiment_validation
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.experiments.validation import execute_coupling_and_map_bound_validation
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.failures import FilesystemFailureError
from fedorbit.infrastructure.manifests import ReusableArtifactManifest
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
    build_layout,
    experiment_workspace,
)
from fedorbit.types import (
    ArtifactState,
    CouplingCompatibility,
    ExperimentName,
    OverwritePolicy,
    StorageLayoutSegment,
)


def _two_cell_coupling_config() -> FedorbitConfig:
    config = active_config()
    coupling = config.generators.coupling_structure
    return config.model_copy(
        update={
            "scientific": config.scientific.model_copy(
                update={
                    "randomness": config.scientific.randomness.model_copy(
                        update={"confirmatory_seeds": (1103,)}
                    )
                }
            ),
            "generators": config.generators.model_copy(
                update={
                    "coupling_structure": coupling.model_copy(
                        update={
                            "compatibility": (CouplingCompatibility.JOINTLY_REALIZABLE,),
                            "response_heterogeneity": (1.0,),
                            "directed_asymmetry": (0.0,),
                            "response_sparsity": (1.0,),
                            "block_patterns": ((2, 2),),
                            "supports": (1, 2),
                        }
                    )
                }
            ),
        }
    )


def _request() -> ExperimentExecutionRequest:
    catalogue = build_catalogue()
    experiment = ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION
    return ExperimentExecutionRequest(
        experiment=experiment,
        definition=catalogue.definition(experiment),
        overwrite_policy=OverwritePolicy.REUSE,
    )


def _payload_digests(store: ArtifactStore) -> dict[str, str]:
    return {
        path: file_sha256(Path(path))
        for manifest in store.all_manifests()
        for path in manifest.payload_paths
    }


def _artifact_ids(store: ArtifactStore) -> frozenset[str]:
    return frozenset(manifest.artifact_id.value for manifest in store.all_manifests())


def _cell_payload_names(layout: WorkspaceLayout) -> tuple[str, ...]:
    derived = (
        experiment_workspace(layout, ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
    )
    return tuple(sorted(path.name for path in derived.glob("coupling-validation.*.json")))


def _failing_persist(
    real_persist: Callable[..., ReusableArtifactManifest],
    calls: list[str],
    *arguments: Any,
    **keywords: Any,
) -> ReusableArtifactManifest:
    calls.append(str(arguments[7]))
    if len(calls) > 1:
        raise FilesystemFailureError("fixture interruption between coupling cells")
    return real_persist(*arguments, **keywords)


def test_coupling_resumes_from_a_partially_complete_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = _request()
    monkeypatch.setattr(experiment_validation, "active_config", _two_cell_coupling_config)
    real_persist = experiment_validation.persist_synthetic_experiment_payload
    calls: list[str] = []

    def failing_persist(*arguments: Any, **keywords: Any) -> ReusableArtifactManifest:
        return _failing_persist(real_persist, calls, *arguments, **keywords)

    monkeypatch.setattr(
        experiment_validation, "persist_synthetic_experiment_payload", failing_persist
    )
    with pytest.raises(FilesystemFailureError):
        execute_coupling_and_map_bound_validation(store, layout, request)
    assert len(calls) == 2
    partial_manifests: tuple[ReusableArtifactManifest, ...] = store.all_manifests()
    assert len(partial_manifests) == 1
    partial_digests = _payload_digests(store)
    partial_ids = _artifact_ids(store)
    assert len(_cell_payload_names(layout)) == 1

    monkeypatch.setattr(experiment_validation, "persist_synthetic_experiment_payload", real_persist)
    execute_coupling_and_map_bound_validation(store, layout, request)

    resumed_digests = _payload_digests(store)
    resumed_ids = _artifact_ids(store)
    for path, digest in partial_digests.items():
        assert resumed_digests[path] == digest
    assert partial_ids <= resumed_ids
    assert len(resumed_ids) > len(partial_ids)
    assert len(store.all_manifests()) == len(resumed_ids)
    assert len(_cell_payload_names(layout)) == 2
    for manifest in store.all_manifests():
        assert store.resolve(manifest.artifact_id).state == ArtifactState.COMPLETED
    summaries = tuple(
        manifest
        for manifest in store.all_manifests()
        if Path(manifest.payload_paths[0]).name.startswith("coupling-and-map-bound-validation.")
    )
    assert len(summaries) == 1
    assert len(store.all_manifests()) == 3
