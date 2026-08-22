from __future__ import annotations

from dataclasses import dataclass

from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.artifacts.reuse import ArtifactStore, ReuseError
from fedorbit.domain.enums import ArtifactState


class SemanticsError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CellDecision:
    cell_coordinates: str
    action: str
    manifest: ReusableArtifactManifest | None = None

    @property
    def reuse(self) -> bool:
        return self.action == "reuse"

    @property
    def overwrite(self) -> bool:
        return self.action == "overwrite"

    @property
    def execute(self) -> bool:
        return self.action == "execute"


class ExecutionSemantics:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def decide(
        self,
        cells: tuple[tuple[str, str], ...],
        overwrite: bool,
        stale_upstreams: frozenset[str] = frozenset(),
    ) -> tuple[CellDecision, ...]:
        decisions: list[CellDecision] = []
        for coordinates, fingerprint in cells:
            manifest = None
            try:
                manifest = self._store.find_by_fingerprint(fingerprint)
            except ReuseError:
                manifest = None
            if manifest is None or not manifest.payload_paths:
                decisions.append(CellDecision(coordinates, "execute"))
                continue
            descendant_stale = manifest.artifact_id in stale_upstreams
            if descendant_stale:
                decisions.append(CellDecision(coordinates, "overwrite", manifest))
                continue
            if overwrite:
                decisions.append(CellDecision(coordinates, "overwrite", manifest))
                continue
            decisions.append(CellDecision(coordinates, "reuse", manifest))
        return tuple(decisions)

    def validate_existing(self, decisions: tuple[CellDecision, ...]) -> None:
        for decision in decisions:
            if decision.manifest is not None:
                self._store.resolve(decision.manifest.artifact_id)

    def stale_descendants(self, coordinates: str) -> frozenset[str]:
        stale: set[str] = set()
        manifests = self._store.manifest_dir()
        if not manifests.is_dir():
            return frozenset()
        for manifest_path in sorted(manifests.glob("*.json")):
            manifest = ReusableArtifactManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            if coordinates in manifest.upstream_artifact_ids:
                stale.add(manifest.artifact_id)
        return frozenset(stale)

    def promote_completed(self, completed: tuple[ReusableArtifactManifest, ...]) -> None:
        for manifest in completed:
            self._store.write_reusable(manifest)

    def terminal_state_of(self, artifact_id: str) -> ArtifactState:
        manifest = self._store.read_reusable(artifact_id)
        return manifest.state
