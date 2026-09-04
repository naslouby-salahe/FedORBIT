from __future__ import annotations

import json
import time
from pathlib import Path

from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.infrastructure.execution import (
    ArtifactStore,
    ExperimentExecutionRequest,
    execute_coupling_and_map_bound_validation,
)
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import ArtifactIdentifier, ArtifactState, ExperimentName, OverwritePolicy


def test_coupling_and_map_bound_validation_has_no_failures(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    definition = build_catalogue().definition(ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION)
    request = ExperimentExecutionRequest(
        ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION,
        definition,
        OverwritePolicy.REUSE,
    )
    started = time.monotonic()
    manifest = execute_coupling_and_map_bound_validation(store, layout, request)
    elapsed = time.monotonic() - started
    assert elapsed < 2400.0
    resolved = store.resolve(ArtifactIdentifier(manifest.artifact_id))
    assert resolved.state == ArtifactState.COMPLETED
    payload = json.loads(Path(manifest.payload_paths[0]).read_text(encoding="utf-8"))
    assert payload["total_instances"] == 4050
    assert payload["total_generation_failures"] == 0
    assert payload["total_incompatible_gap_failures"] == 0
    assert payload["map_bound_fixtures"]["zero_map_value_failures"] == 0
    assert payload["map_bound_fixtures"]["high_map_value_failures"] == 0
    reused = execute_coupling_and_map_bound_validation(store, layout, request)
    assert reused.artifact_id == manifest.artifact_id
    assert len(store.all_manifests()) == 1
