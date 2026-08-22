from __future__ import annotations

from dataclasses import dataclass

from fedorbit.artifacts.reuse import ArtifactStore
from fedorbit.domain.enums import ArtifactState


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    valid_artifact_ids: tuple[str, ...]
    next_resume_coordinates: str | None
    stochastic_boundary_ok: bool


class RecoveryBoundary:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def valid_artifact_ids(self) -> tuple[str, ...]:
        manifest_dir = self._store.manifest_dir()
        if not manifest_dir.is_dir():
            return ()
        valid: list[str] = []
        for path in sorted(manifest_dir.glob("*.json")):
            manifest = self._store.read_reusable(path.stem)
            try:
                self._store.resolve(path.stem)
            except Exception:
                continue
            if manifest.state == ArtifactState.COMPLETED:
                valid.append(manifest.artifact_id)
        return tuple(valid)

    def next_resume(self, ordered_cells: tuple[tuple[str, str], ...]) -> RecoveryRecord:
        valid = set(self.valid_artifact_ids())
        resume: str | None = None
        for coordinates, artifact_id in ordered_cells:
            if artifact_id not in valid:
                resume = coordinates
                break
        return RecoveryRecord(
            valid_artifact_ids=tuple(sorted(valid)),
            next_resume_coordinates=resume,
            stochastic_boundary_ok=resume is not None,
        )
