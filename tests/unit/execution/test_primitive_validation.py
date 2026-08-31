from __future__ import annotations

import json
from pathlib import Path

from fedorbit.artifacts.paths import build_layout
from fedorbit.artifacts.storage import ArtifactStore
from fedorbit.domain.enums import ArtifactState
from fedorbit.domain.records import ArtifactIdentifier
from fedorbit.execution.primitive_validation import execute_primitive_validation


def test_primitive_validation_persists_verified_nonclaim_evidence(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    manifest = execute_primitive_validation(store, layout)
    resolved = store.resolve(ArtifactIdentifier(manifest.artifact_id))
    assert resolved.state == ArtifactState.COMPLETED
    completion = store.read_completion(ArtifactIdentifier(manifest.artifact_id))
    assert (
        completion.semantic_experiment_coordinates
        == '{"experiment":"Mathematical Primitive Validation","seed":0}'
    )
    payload = json.loads(Path(manifest.payload_paths[0]).read_text(encoding="utf-8"))
    assert payload["assignment_is_bijective"]
    assert payload["lower_bound_not_above_upper_bound"]
    assert manifest.semantic_producer_coordinates == completion.semantic_experiment_coordinates
    assert "mathematical-primitive-validation" in Path(manifest.payload_paths[0]).parts
