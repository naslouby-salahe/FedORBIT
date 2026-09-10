from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from fedorbit.infrastructure.manifests import (
    CompletionManifest,
    ReusableArtifactManifest,
)
from fedorbit.infrastructure.reuse import validate_completed_artifact, validate_reusable_artifact
from fedorbit.infrastructure.runtime import ExecutionLogEvent, execution_logger
from fedorbit.infrastructure.storage import StorageError, atomic_write_json
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import (
    ArtifactFingerprint,
    ArtifactIdentifier,
    ArtifactState,
    ExecutionCell,
    ExecutionStageName,
    SemanticCoordinates,
    StableJsonPayload,
    StorageLayoutSegment,
)


class ExecutionError(ValueError):
    pass


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._manifests = root / StorageLayoutSegment.MANIFESTS
        self._completions = root / StorageLayoutSegment.COMPLETIONS
        self._staging = root / StorageLayoutSegment.STAGING
        self._index_path = self._root / "fingerprint-index.json"

    @property
    def root(self) -> Path:
        return self._root

    def manifest_path(self, artifact_id: ArtifactIdentifier) -> Path:
        return self._manifests / f"{artifact_id.value}.json"

    def manifest_dir(self) -> Path:
        return self._manifests

    def completion_path(self, artifact_id: ArtifactIdentifier) -> Path:
        return self._completions / f"{artifact_id.value}.json"

    def staging_dir(self) -> Path:
        return self._staging

    def write_reusable(self, manifest: ReusableArtifactManifest) -> None:
        atomic_write_json(
            self.manifest_path(manifest.artifact_id),
            manifest.model_dump(mode="json"),
        )
        self._index_fingerprint(manifest.dependency_fingerprint_sha256, manifest.artifact_id)

    def write_completed(
        self,
        manifest: ReusableArtifactManifest,
        completion: CompletionManifest,
    ) -> None:
        if not manifest.completion_required:
            raise StorageError("completed artifacts must require a completion record")
        try:
            validate_completed_artifact(manifest, completion)
        except ValueError as error:
            raise StorageError(str(error)) from error
        self.write_reusable(manifest)
        atomic_write_json(
            self.completion_path(manifest.artifact_id),
            completion.model_dump(mode="json"),
        )
        execution_logger().record(
            ExecutionLogEvent(
                occurred_at=datetime.now(UTC),
                cell_coordinates=SemanticCoordinates(str(manifest.semantic_producer_coordinates)),
                artifact_id=manifest.artifact_id,
                state=ArtifactState.COMPLETED,
                stage=ExecutionStageName(manifest.producer_stage.value),
            )
        )

    def read_reusable(self, artifact_id: ArtifactIdentifier) -> ReusableArtifactManifest:
        path = self.manifest_path(artifact_id)
        if not path.is_file():
            raise StorageError(f"no artifact manifest for {artifact_id.value}")
        return ReusableArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def read_completion(self, artifact_id: ArtifactIdentifier) -> CompletionManifest:
        path = self.completion_path(artifact_id)
        if not path.is_file():
            raise StorageError(f"no completion manifest for {artifact_id.value}")
        return CompletionManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def resolve(self, artifact_id: ArtifactIdentifier) -> ReusableArtifactManifest:
        manifest = self.read_reusable(artifact_id)
        validate_reusable_artifact(manifest)
        if manifest.completion_required:
            try:
                validate_completed_artifact(manifest, self.read_completion(artifact_id))
            except ValueError as error:
                raise StorageError(str(error)) from error
        return manifest

    def find_by_fingerprint(
        self, fingerprint_sha256: ArtifactFingerprint
    ) -> ReusableArtifactManifest | None:
        logger = execution_logger()
        indexed = self._lookup_index(fingerprint_sha256.value)
        if indexed is not None:
            try:
                manifest = self.resolve(indexed)
            except ValueError:
                logger.event(
                    "cache_miss",
                    fingerprint=fingerprint_sha256.value,
                    reason="indexed_manifest_invalid",
                )
                return None
            if manifest.dependency_fingerprint_sha256 == fingerprint_sha256.value:
                logger.event(
                    "cache_hit",
                    fingerprint=fingerprint_sha256.value,
                    artifact_id=manifest.artifact_id.value,
                    artifact_path=manifest.payload_paths[0] if manifest.payload_paths else None,
                )
                return manifest
        if not self._manifests.is_dir():
            logger.event("cache_miss", fingerprint=fingerprint_sha256.value, reason="no_manifests")
            return None
        for path in sorted(self._manifests.glob(StorageLayoutSegment.MANIFEST_GLOB)):
            if path.name == self._index_path.name:
                continue
            manifest = ReusableArtifactManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if manifest.dependency_fingerprint_sha256 != fingerprint_sha256.value:
                continue
            try:
                resolved = self.resolve(manifest.artifact_id)
            except ValueError:
                logger.event(
                    "cache_miss",
                    fingerprint=fingerprint_sha256.value,
                    reason="manifest_invalid",
                )
                return None
            self._index_fingerprint(fingerprint_sha256.value, resolved.artifact_id)
            logger.event(
                "cache_hit",
                fingerprint=fingerprint_sha256.value,
                artifact_id=resolved.artifact_id.value,
                artifact_path=resolved.payload_paths[0] if resolved.payload_paths else None,
            )
            return resolved
        logger.event("cache_miss", fingerprint=fingerprint_sha256.value, reason="absent")
        return None

    def remove_manifest(self, artifact_id: ArtifactIdentifier) -> None:
        self.manifest_path(artifact_id).unlink(missing_ok=True)
        self.completion_path(artifact_id).unlink(missing_ok=True)
        index = self._read_index()
        retained = OrderedDict(
            (fingerprint, identifier)
            for fingerprint, identifier in index.items()
            if identifier != artifact_id.value
        )
        if retained != index:
            atomic_write_json(self._index_path, cast(StableJsonPayload, retained))

    def all_manifests(self) -> tuple[ReusableArtifactManifest, ...]:
        if not self._manifests.is_dir():
            return ()
        manifests: list[ReusableArtifactManifest] = []
        for path in sorted(self._manifests.glob(StorageLayoutSegment.MANIFEST_GLOB)):
            if path.name == self._index_path.name:
                continue
            manifests.append(
                ReusableArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
            )
        return tuple(manifests)

    def _read_index(self) -> OrderedDict[str, str]:
        if not self._index_path.is_file():
            return OrderedDict()
        parsed = json.loads(self._index_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            return OrderedDict()
        entries: OrderedDict[str, str] = OrderedDict()
        for key, value in parsed.items():
            if isinstance(key, str) and isinstance(value, str):
                entries[key] = value
        return entries

    def _lookup_index(self, fingerprint: str) -> ArtifactIdentifier | None:
        identifier = self._read_index().get(fingerprint)
        if identifier is None:
            return None
        return ArtifactIdentifier(identifier)

    def _index_fingerprint(self, fingerprint: str, artifact_id: ArtifactIdentifier) -> None:
        index = self._read_index()
        if index.get(fingerprint) == artifact_id.value:
            return
        index[fingerprint] = artifact_id.value
        atomic_write_json(self._index_path, cast(StableJsonPayload, index))


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
                resolved = self._store.resolve(manifest.artifact_id)
            except ValueError:
                continue
            if resolved.state == ArtifactState.COMPLETED:
                valid.append(resolved.artifact_id)
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


def execution_store() -> ArtifactStore:
    return ArtifactStore(build_layout().execution_root)
