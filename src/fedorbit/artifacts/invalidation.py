from __future__ import annotations

from dataclasses import dataclass

from fedorbit.artifacts.fingerprints import STAGE_DEPENDENCIES, STAGES
from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.artifacts.reuse import ArtifactStore


class InvalidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StageRule:
    stage: str
    upstream_stages: tuple[str, ...]
    downstream_stages: tuple[str, ...]


def stage_rules() -> tuple[StageRule, ...]:
    rules: dict[str, StageRule] = {}
    for stage in STAGES:
        upstream = STAGE_DEPENDENCIES.get(stage, ())
        downstream = tuple(
            candidate
            for candidate, dependencies in STAGE_DEPENDENCIES.items()
            if stage in dependencies
        )
        rules[stage] = StageRule(stage, upstream, downstream)
    return tuple(rules.values())


def descendants_of_stage(stage: str) -> frozenset[str]:
    rules_by_stage = {rule.stage: rule for rule in stage_rules()}
    if stage not in rules_by_stage:
        raise InvalidationError(f"unknown stage: {stage}")
    visited: set[str] = set()
    frontier = list(rules_by_stage[stage].downstream_stages)
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)
        frontier.extend(rules_by_stage[current].downstream_stages)
    return frozenset(visited)


def changed_stage_affects(producer_stage: str, changed_stage: str) -> bool:
    return producer_stage in descendants_of_stage(changed_stage) or producer_stage == changed_stage


class SelectiveInvalidation:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def invalidate_stage(
        self, changed_stage: str, changed_artifact_id: str | None = None
    ) -> tuple[str, ...]:
        affected_stages = descendants_of_stage(changed_stage) | frozenset({changed_stage})
        manifests = self._manifests()
        invalidated: list[str] = []
        for manifest in manifests:
            if manifest.producer_stage not in affected_stages:
                continue
            if (
                changed_artifact_id is not None
                and changed_artifact_id not in manifest.upstream_artifact_ids
            ):
                continue
            self._store.manifest_path(manifest.artifact_id).unlink(missing_ok=True)
            invalidated.append(manifest.artifact_id)
        return tuple(invalidated)

    def invalidate_descendants(self, upstream_artifact_id: str) -> tuple[str, ...]:
        manifests = self._manifests()
        invalidated: list[str] = []
        frontier = [upstream_artifact_id]
        visited: set[str] = set()
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            for manifest in manifests:
                if (
                    current in manifest.upstream_artifact_ids
                    and manifest.artifact_id not in visited
                ):
                    invalidated.append(manifest.artifact_id)
                    frontier.append(manifest.artifact_id)
                    self._store.manifest_path(manifest.artifact_id).unlink(missing_ok=True)
        return tuple(invalidated)

    def _manifests(self) -> list[ReusableArtifactManifest]:
        manifest_dir = self._store.manifest_dir()
        if not manifest_dir.is_dir():
            return []
        return [
            ReusableArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(manifest_dir.glob("*.json"))
        ]
