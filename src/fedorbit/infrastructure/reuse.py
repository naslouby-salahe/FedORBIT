from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from fedorbit.infrastructure.manifests import (
    CompletionManifest,
    ReusableArtifactManifest,
    completion_manifest_self_hash,
    file_sha256,
)
from fedorbit.infrastructure.provenance import STAGE_DEPENDENCIES
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactStage,
    ArtifactState,
    ExecutionCell,
    OverwritePolicy,
    SemanticCoordinates,
)

if TYPE_CHECKING:
    from fedorbit.infrastructure.execution import ArtifactStore


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
    except ValueError:
        return None


class ArtifactValidationError(ValueError):
    pass


def validate_reusable_artifact(manifest: ReusableArtifactManifest) -> None:
    if manifest.state != ArtifactState.COMPLETED:
        raise ArtifactValidationError(f"artifact {manifest.artifact_id} is not completed")
    if not manifest.payload_paths:
        raise ArtifactValidationError(f"artifact {manifest.artifact_id} has no payload")
    for payload_path in manifest.payload_paths:
        path = Path(payload_path)
        if not path.is_file():
            raise ArtifactValidationError(
                f"missing payload for {manifest.artifact_id}: {payload_path}"
            )
        observed = file_sha256(path)
        if observed != manifest.payload_sha256:
            raise ArtifactValidationError(
                f"payload checksum mismatch for {manifest.artifact_id}: "
                f"expected {manifest.payload_sha256}, observed {observed}"
            )


def validate_completion_manifest(manifest: CompletionManifest) -> None:
    if not manifest.completion_written_last:
        raise ArtifactValidationError("completion manifest was not written last")
    if manifest.completion_manifest_sha256 != completion_manifest_self_hash(manifest):
        raise ArtifactValidationError("completion manifest self-hash mismatch")


def validate_completed_artifact(
    manifest: ReusableArtifactManifest,
    completion: CompletionManifest,
) -> None:
    validate_reusable_artifact(manifest)
    validate_completion_manifest(completion)
    if completion.terminal_state.value != ArtifactState.COMPLETED.value:
        raise ArtifactValidationError("completion record is not completed")
    if completion.dependency_fingerprint_sha256 != manifest.dependency_fingerprint_sha256:
        raise ArtifactValidationError(
            "completion record fingerprint does not match reusable manifest"
        )
    if completion.producer_stage != manifest.producer_stage:
        raise ArtifactValidationError("completion record stage does not match reusable manifest")
    if completion.completion_manifest_sha256 != manifest.completion_manifest_sha256:
        raise ArtifactValidationError("completion record hash does not match reusable manifest")


def validate_upstream_lineage(
    manifest: ReusableArtifactManifest,
    available_artifact_ids: frozenset[str],
) -> None:
    missing = tuple(
        artifact_id
        for artifact_id in manifest.upstream_artifact_ids
        if artifact_id not in available_artifact_ids
    )
    if missing:
        raise ArtifactValidationError(f"missing upstream artifacts: {missing}")
