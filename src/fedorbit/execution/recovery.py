from __future__ import annotations

import shutil
from dataclasses import dataclass

from fedorbit.artifacts.storage import ArtifactStore
from fedorbit.domain.enums import ArtifactState
from fedorbit.domain.records import ArtifactIdentifier, ExecutionCell, SemanticCoordinates


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    valid_artifact_ids: tuple[ArtifactIdentifier, ...]
    next_resume_coordinates: SemanticCoordinates | None
    stochastic_boundary_ok: bool


class RecoveryBoundary:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def discard_interrupted_staging(self) -> None:
        staging = self._store.staging_dir()
        if staging.is_dir():
            shutil.rmtree(staging)

    def valid_artifact_ids(self) -> tuple[ArtifactIdentifier, ...]:
        valid: list[ArtifactIdentifier] = []
        for manifest in self._store.all_manifests():
            try:
                resolved = self._store.resolve(ArtifactIdentifier(manifest.artifact_id))
            except ValueError:
                continue
            if resolved.state == ArtifactState.COMPLETED:
                valid.append(ArtifactIdentifier(resolved.artifact_id))
        return tuple(sorted(valid, key=lambda identifier: identifier.value))

    def next_resume(self, ordered_cells: tuple[ExecutionCell, ...]) -> RecoveryRecord:
        valid = frozenset(self.valid_artifact_ids())
        resume = next(
            (cell.coordinates for cell in ordered_cells if cell.artifact_identifier not in valid),
            None,
        )
        return RecoveryRecord(
            valid_artifact_ids=tuple(sorted(valid, key=lambda identifier: identifier.value)),
            next_resume_coordinates=resume,
            stochastic_boundary_ok=resume is not None,
        )
