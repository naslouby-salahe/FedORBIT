from __future__ import annotations

import json
from pathlib import Path

from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.classification import execute_evidence_classification
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import ArtifactState, EvidenceStatus, ExperimentName, OverwritePolicy


def test_evidence_classification_records_not_tested_without_synthesis(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    catalogue = build_catalogue()
    request = ExperimentExecutionRequest(
        experiment=ExperimentName.EVIDENCE_CLASSIFICATION,
        definition=catalogue.definition(ExperimentName.EVIDENCE_CLASSIFICATION),
        overwrite_policy=OverwritePolicy.REUSE,
    )
    manifest = execute_evidence_classification(store, layout, request)
    resolved = store.resolve(manifest.artifact_id)
    assert resolved.state == ArtifactState.COMPLETED
    payload = json.loads(Path(manifest.payload_paths[0]).read_text(encoding="utf-8"))
    assert payload["synthesis_artifact_id"] is None
    statuses = payload["statuses"]
    assert statuses
    assert {row["final_state"] for row in statuses} == {EvidenceStatus.NOT_TESTED.value}
