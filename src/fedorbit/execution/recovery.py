from __future__ import annotations

import shutil
from dataclasses import dataclass

from fedorbit.artifacts.storage import ArtifactStore
from fedorbit.domain.enums import ArtifactState


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    valid_artifact_ids: tuple[str, ...]
    next_resume_coordinates: str | None
    stochastic_boundary_ok: bool


class RecoveryBoundary:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def discard_interrupted_staging(self) -> None:
        staging = self._store.staging_dir()
        if staging.is_dir():
            shutil.rmtree(staging)

    def valid_artifact_ids(self) -> tuple[str, ...]:
        valid: list[str] = []
        for manifest in self._store.all_manifests():
            try:
                resolved = self._store.resolve(manifest.artifact_id)
            except ValueError:
                continue
            if resolved.state == ArtifactState.COMPLETED:
                valid.append(resolved.artifact_id)
        return tuple(valid)

    def next_resume(self, ordered_cells: tuple[tuple[str, str], ...]) -> RecoveryRecord:
        valid = set(self.valid_artifact_ids())
        resume = next(
            (coordinates for coordinates, artifact_id in ordered_cells if artifact_id not in valid),
            None,
        )
        return RecoveryRecord(
            valid_artifact_ids=tuple(sorted(valid)),
            next_resume_coordinates=resume,
            stochastic_boundary_ok=resume is not None,
        )
