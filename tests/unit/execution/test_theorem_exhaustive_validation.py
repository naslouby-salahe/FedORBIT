from __future__ import annotations

import json
import time
from pathlib import Path

from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.infrastructure.execution import (
    ArtifactStore,
    ExperimentExecutionRequest,
    execute_exact_sparse_theorem_exhaustive_validation,
)
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import ArtifactIdentifier, ArtifactState, ExperimentName, OverwritePolicy


def test_theorem_exhaustive_validation_produces_zero_wrong_minima(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    definition = build_catalogue().definition(
        ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION
    )
    request = ExperimentExecutionRequest(
        ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION,
        definition,
        OverwritePolicy.REUSE,
    )
    started = time.monotonic()
    manifest = execute_exact_sparse_theorem_exhaustive_validation(store, layout, request)
    elapsed = time.monotonic() - started
    assert elapsed < 120.0
    resolved = store.resolve(ArtifactIdentifier(manifest.artifact_id))
    assert resolved.state == ArtifactState.COMPLETED
    payload = json.loads(Path(manifest.payload_paths[0]).read_text(encoding="utf-8"))
    assert payload["total_cells"] == 17
    assert payload["total_instances"] == 17000
    for cell in payload["cells"]:
        assert cell["wrong_minima_count"] == 0
        assert cell["invalid_certificate_count"] == 0
        assert cell["max_absolute_objective_error"] <= 1e-9
    reused = execute_exact_sparse_theorem_exhaustive_validation(store, layout, request)
    assert reused.artifact_id == manifest.artifact_id
    assert len(store.all_manifests()) == 1
