from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum

from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.artifacts.provenance import STAGE_DEPENDENCIES
from fedorbit.artifacts.storage import ArtifactStore, StorageError
from fedorbit.domain.enums import ArtifactStage, OverwritePolicy
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
    stage: ArtifactStage
    upstream_stages: tuple[ArtifactStage, ...]
    downstream_stages: tuple[ArtifactStage, ...]


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
        for stage in ArtifactStage
    )


def descendants_of_stage(stage: ArtifactStage) -> frozenset[ArtifactStage]:
    rules = OrderedDict((rule.stage, rule) for rule in stage_rules())
    if stage not in rules:
        raise ReuseError(f"unknown stage: {stage}")
    visited: set[ArtifactStage] = set()
    frontier = list(rules[stage].downstream_stages)
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)
        frontier.extend(rules[current].downstream_stages)
    return frozenset(visited)


def changed_stage_affects(producer_stage: ArtifactStage, changed_stage: ArtifactStage) -> bool:
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
            manifest = self._store.find_by_fingerprint(cell.dependency_fingerprint)
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
                self._store.resolve(ArtifactIdentifier(decision.manifest.artifact_id))

    def stale_descendants(self, artifact_id: str) -> frozenset[str]:
        return frozenset(
            manifest.artifact_id
            for manifest in self._store.all_manifests()
            if artifact_id in manifest.upstream_artifact_ids
        )

    def promote_completed(self, manifests: tuple[ReusableArtifactManifest, ...]) -> None:
        for manifest in manifests:
            self._store.write_reusable(manifest)
            self._store.resolve(ArtifactIdentifier(manifest.artifact_id))


class SelectiveInvalidation:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def invalidate_stage(
        self,
        changed_stage: ArtifactStage,
        changed_artifact_id: ArtifactIdentifier | None = None,
    ) -> tuple[ArtifactIdentifier, ...]:
        affected = descendants_of_stage(changed_stage) | frozenset({changed_stage})
        invalidated: list[ArtifactIdentifier] = []
        for manifest in self._store.all_manifests():
            if manifest.producer_stage not in affected:
                continue
            if (
                changed_artifact_id is not None
                and changed_artifact_id.value not in manifest.upstream_artifact_ids
            ):
                continue
            self._store.remove_manifest(ArtifactIdentifier(manifest.artifact_id))
            invalidated.append(ArtifactIdentifier(manifest.artifact_id))
        return tuple(invalidated)

    def invalidate_descendants(
        self, upstream_artifact_id: ArtifactIdentifier
    ) -> tuple[ArtifactIdentifier, ...]:
        manifests = self._store.all_manifests()
        invalidated: list[ArtifactIdentifier] = []
        frontier = [upstream_artifact_id]
        visited: set[ArtifactIdentifier] = set()
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            for manifest in manifests:
                if current.value not in manifest.upstream_artifact_ids:
                    continue
                artifact_identifier = ArtifactIdentifier(manifest.artifact_id)
                if artifact_identifier in visited:
                    continue
                invalidated.append(artifact_identifier)
                frontier.append(artifact_identifier)
                self._store.remove_manifest(artifact_identifier)
        return tuple(invalidated)


def resolved_or_none(
    store: ArtifactStore,
    artifact_id: ArtifactIdentifier,
) -> ReusableArtifactManifest | None:
    try:
        return store.resolve(artifact_id)
    except (StorageError, ValueError):
        return None
