from __future__ import annotations

import json
from pathlib import Path

from fedorbit.experiments.validation import execute_primitive_validation
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import ArtifactState


def test_primitive_validation_persists_verified_nonclaim_evidence(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    manifest = execute_primitive_validation(store, layout)
    resolved = store.resolve(manifest.artifact_id)
    assert resolved.state == ArtifactState.COMPLETED
    completion = store.read_completion(manifest.artifact_id)
    assert (
        completion.semantic_experiment_coordinates
        == '{"experiment":"Mathematical Primitive Validation","seed":0}'
    )
    payload = json.loads(Path(manifest.payload_paths[0]).read_text(encoding="utf-8"))
    assert payload["assignment_is_bijective"]
    assert payload["lower_bound_not_above_upper_bound"]
    assert payload["null_padding_present"]
    assert payload["orbit_mean_finite"]
    assert payload["rectangular_hull_ordered"]
    assert payload["assignment_serialization_round_trip"]
    assert payload["fixture_error_within_tolerance"]
    assert payload["orbit_size"] >= 1
    assert payload["active_image_map_count"] >= 1
    assert manifest.semantic_producer_coordinates == completion.semantic_experiment_coordinates
    assert "mathematical-primitive-validation" in Path(manifest.payload_paths[0]).parts
    reused = execute_primitive_validation(store, layout)
    assert reused.artifact_id == manifest.artifact_id
    assert len(store.all_manifests()) == 1
