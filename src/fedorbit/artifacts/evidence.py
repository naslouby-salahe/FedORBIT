from __future__ import annotations

from pathlib import Path

from fedorbit.artifacts.paths import WorkspaceLayout, results_workspace
from fedorbit.artifacts.reuse import ArtifactStore, ReuseError
from fedorbit.artifacts.serialization import atomic_write_json
from fedorbit.domain.enums import ExperimentName


class EvidenceError(ValueError):
    pass


class VerifiedEvidenceWriter:
    def __init__(self, store: ArtifactStore, layout: WorkspaceLayout) -> None:
        self._store = store
        self._layout = layout

    def write(self, experiment: ExperimentName, artifact_id: str, evidence: object) -> Path:
        try:
            self._store.resolve(artifact_id)
        except ReuseError as error:
            raise EvidenceError(
                f"evidence requires a verified completed artifact: {error}"
            ) from error
        workspace = results_workspace(self._layout, experiment)
        destination = workspace / f"{experiment.value}.evidence.json"
        atomic_write_json(destination, evidence)
        return destination
