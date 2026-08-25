from __future__ import annotations

from pathlib import Path

from fedorbit.artifacts.paths import WorkspaceLayout, results_workspace
from fedorbit.artifacts.storage import ArtifactStore, atomic_write_json
from fedorbit.domain.enums import ExperimentName
from fedorbit.domain.serialization import StableJsonPayload
from fedorbit.reporting.figures import EvidenceFigure
from fedorbit.reporting.tables import EvidenceTable


class EvidenceExportError(ValueError):
    pass


class VerifiedEvidenceWriter:
    def __init__(self, store: ArtifactStore, layout: WorkspaceLayout) -> None:
        self._store = store
        self._layout = layout

    def write(
        self,
        experiment: ExperimentName,
        artifact_id: str,
        evidence: StableJsonPayload,
    ) -> Path:
        try:
            self._store.resolve(artifact_id)
        except ValueError as error:
            raise EvidenceExportError(
                f"evidence requires a verified completed artifact: {error}"
            ) from error
        workspace = results_workspace(self._layout, experiment)
        destination = workspace / f"{experiment.value}.evidence.json"
        atomic_write_json(destination, evidence)
        return destination

    def write_table(
        self,
        experiment: ExperimentName,
        artifact_id: str,
        table: EvidenceTable,
        name: str,
    ) -> Path:
        self._store.resolve(artifact_id)
        destination = results_workspace(self._layout, experiment) / f"{name}.table.json"
        atomic_write_json(destination, table.payload())
        return destination

    def write_figure(
        self,
        experiment: ExperimentName,
        artifact_id: str,
        figure: EvidenceFigure,
        name: str,
    ) -> Path:
        self._store.resolve(artifact_id)
        destination = results_workspace(self._layout, experiment) / f"{name}.figure.json"
        atomic_write_json(destination, figure.payload())
        return destination
