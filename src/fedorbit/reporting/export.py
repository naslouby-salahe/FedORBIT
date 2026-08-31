from __future__ import annotations

from pathlib import Path

from fedorbit.artifacts.paths import WorkspaceLayout, results_workspace
from fedorbit.artifacts.storage import ArtifactStore, atomic_write_bytes, atomic_write_json
from fedorbit.domain.enums import ExperimentName
from fedorbit.domain.records import ArtifactIdentifier
from fedorbit.domain.serialization import StableJsonPayload, stable_json
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
        artifact_id: ArtifactIdentifier,
        evidence: StableJsonPayload,
        overwrite: bool = False,
    ) -> Path:
        try:
            self._store.resolve(artifact_id)
        except ValueError as error:
            raise EvidenceExportError(
                f"evidence requires a verified completed artifact: {error}"
            ) from error
        workspace = results_workspace(self._layout, experiment)
        destination = workspace / f"{experiment.value}.evidence.json"
        rendered = (stable_json(evidence) + "\n").encode("utf-8")
        if destination.is_file():
            if destination.read_bytes() == rendered:
                return destination
            if not overwrite:
                raise EvidenceExportError(
                    "evidence export already exists with different content; use --overwrite"
                )
        atomic_write_bytes(destination, rendered)
        return destination

    def write_table(
        self,
        experiment: ExperimentName,
        artifact_id: ArtifactIdentifier,
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
        artifact_id: ArtifactIdentifier,
        figure: EvidenceFigure,
        name: str,
    ) -> Path:
        self._store.resolve(artifact_id)
        destination = results_workspace(self._layout, experiment) / f"{name}.figure.json"
        atomic_write_json(destination, figure.payload())
        return destination
