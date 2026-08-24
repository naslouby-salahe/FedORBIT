from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.artifacts.storage import ArtifactStore
from fedorbit.execution.reuse import CellDecision, ExecutionAction


class ExecutionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    decision: CellDecision
    manifest: ReusableArtifactManifest | None


class ExecutionExecutor:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def execute(
        self,
        decisions: tuple[CellDecision, ...],
        producer: Callable[[CellDecision], ReusableArtifactManifest],
    ) -> tuple[ExecutionResult, ...]:
        results: list[ExecutionResult] = []
        for decision in decisions:
            if decision.action == ExecutionAction.REUSE:
                if decision.manifest is None:
                    raise ExecutionError("reuse decision has no manifest")
                manifest = self._store.resolve(decision.manifest.artifact_id)
                results.append(ExecutionResult(decision, manifest))
                continue
            manifest = producer(decision)
            self._store.write_reusable(manifest)
            validated = self._store.resolve(manifest.artifact_id)
            results.append(ExecutionResult(decision, validated))
        return tuple(results)
