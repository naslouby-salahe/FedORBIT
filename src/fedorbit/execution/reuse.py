from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum

from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.artifacts.provenance import STAGE_DEPENDENCIES, STAGES
from fedorbit.artifacts.storage import ArtifactStore, StorageError
from fedorbit.domain.enums import OverwritePolicy
from fedorbit.domain.records import (
    ArtifactIdentifier,
    ExecutionCell,
    SemanticCoordinates,
)


class ReuseError(ValueError):
    pass


class ExecutionAction(StrEnum):
    EXECUTE = "execute"
    REUSE = "reuse"
    OVERWRITE = "overwrite"


@dataclass(frozen=True, slots=True)
class CellDecision:
    cell_coordinates: SemanticCoordinates
    action: ExecutionAction
    manifest: ReusableArtifactManifest | None = None


@dataclass(frozen=True, slots=True)
class StageRule:
    stage: str
    upstream_stages: tuple[str, ...]
    downstream_stages: tuple[str, ...]


def stage_rules() -> tuple[StageRule, ...]:
    return tuple(
        StageRule(
            stage,
            STAGE_DEPENDENCIES.get(stage, ()),
            tuple(
                candidate
                for candidate, dependencies in STAGE_DEPENDENCIES.items()
                if stage in dependencies
            ),
        )
        for stage in STAGES
    )


def descendants_of_stage(stage: str) -> frozenset[str]:
    rules = OrderedDict((rule.stage, rule) for rule in stage_rules())
    if stage not in rules:
        raise ReuseError(f"unknown stage: {stage}")
    visited: set[str] = set()
    frontier = list(rules[stage].downstream_stages)
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)
        frontier.extend(rules[current].downstream_stages)
    return frozenset(visited)


def changed_stage_affects(producer_stage: str, changed_stage: str) -> bool:
    return producer_stage == changed_stage or producer_stage in descendants_of_stage(changed_stage)


class ExecutionReuse:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def decide(
        self,
        cells: tuple[ExecutionCell, ...],
        overwrite_policy: OverwritePolicy,
        stale_artifact_ids: frozenset[ArtifactIdentifier] = frozenset(),
    ) -> tuple[CellDecision, ...]:
        decisions: list[CellDecision] = []
        for cell in cells:
            manifest = self._store.find_by_fingerprint(cell.dependency_fingerprint.value)
            if manifest is None or not manifest.payload_paths:
                decisions.append(CellDecision(cell.coordinates, ExecutionAction.EXECUTE))
            elif (
                overwrite_policy == OverwritePolicy.REPLACE
                or ArtifactIdentifier(manifest.artifact_id) in stale_artifact_ids
            ):
                decisions.append(
                    CellDecision(cell.coordinates, ExecutionAction.OVERWRITE, manifest)
                )
            else:
                decisions.append(CellDecision(cell.coordinates, ExecutionAction.REUSE, manifest))
        return tuple(decisions)

    def validate_existing(self, decisions: tuple[CellDecision, ...]) -> None:
        for decision in decisions:
            if decision.manifest is not None:
                self._store.resolve(decision.manifest.artifact_id)

    def stale_descendants(self, artifact_id: str) -> frozenset[str]:
        return frozenset(
            manifest.artifact_id
            for manifest in self._store.all_manifests()
            if artifact_id in manifest.upstream_artifact_ids
        )

    def promote_completed(self, manifests: tuple[ReusableArtifactManifest, ...]) -> None:
        for manifest in manifests:
            self._store.write_reusable(manifest)
            self._store.resolve(manifest.artifact_id)


class SelectiveInvalidation:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def invalidate_stage(
        self,
        changed_stage: str,
        changed_artifact_id: str | None = None,
    ) -> tuple[str, ...]:
        affected = descendants_of_stage(changed_stage) | frozenset({changed_stage})
        invalidated: list[str] = []
        for manifest in self._store.all_manifests():
            if manifest.producer_stage not in affected:
                continue
            if (
                changed_artifact_id is not None
                and changed_artifact_id not in manifest.upstream_artifact_ids
            ):
                continue
            self._store.remove_manifest(manifest.artifact_id)
            invalidated.append(manifest.artifact_id)
        return tuple(invalidated)

    def invalidate_descendants(self, upstream_artifact_id: str) -> tuple[str, ...]:
        manifests = self._store.all_manifests()
        invalidated: list[str] = []
        frontier = [upstream_artifact_id]
        visited: set[str] = set()
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            for manifest in manifests:
                if current not in manifest.upstream_artifact_ids:
                    continue
                if manifest.artifact_id in visited:
                    continue
                invalidated.append(manifest.artifact_id)
                frontier.append(manifest.artifact_id)
                self._store.remove_manifest(manifest.artifact_id)
        return tuple(invalidated)


def resolved_or_none(
    store: ArtifactStore,
    artifact_id: str,
) -> ReusableArtifactManifest | None:
    try:
        return store.resolve(artifact_id)
    except (StorageError, ValueError):
        return None
