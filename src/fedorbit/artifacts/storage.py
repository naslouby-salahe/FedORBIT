from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.domain.records import ArtifactFingerprint, ArtifactIdentifier
from fedorbit.domain.serialization import StableJsonPayload, stable_json


class StorageError(ValueError):
    pass


def atomic_write_json(path: Path, payload: StableJsonPayload) -> None:
    atomic_write_bytes(path, (stable_json(payload) + "\n").encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._manifests = root / "manifests"
        self._staging = root / "staging"

    @property
    def root(self) -> Path:
        return self._root

    def manifest_path(self, artifact_id: ArtifactIdentifier) -> Path:
        return self._manifests / f"{artifact_id.value}.json"

    def manifest_dir(self) -> Path:
        return self._manifests

    def staging_dir(self) -> Path:
        return self._staging

    def write_reusable(self, manifest: ReusableArtifactManifest) -> None:
        atomic_write_json(
            self.manifest_path(ArtifactIdentifier(manifest.artifact_id)),
            manifest.model_dump(mode="json"),
        )

    def read_reusable(self, artifact_id: ArtifactIdentifier) -> ReusableArtifactManifest:
        path = self.manifest_path(artifact_id)
        if not path.is_file():
            raise StorageError(f"no artifact manifest for {artifact_id.value}")
        return ReusableArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def resolve(self, artifact_id: ArtifactIdentifier) -> ReusableArtifactManifest:
        from fedorbit.artifacts.validation import validate_reusable_artifact

        manifest = self.read_reusable(artifact_id)
        validate_reusable_artifact(manifest)
        return manifest

    def find_by_fingerprint(
        self, fingerprint_sha256: ArtifactFingerprint
    ) -> ReusableArtifactManifest | None:
        if not self._manifests.is_dir():
            return None
        for path in sorted(self._manifests.glob("*.json")):
            manifest = ReusableArtifactManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if manifest.dependency_fingerprint_sha256 != fingerprint_sha256.value:
                continue
            try:
                from fedorbit.artifacts.validation import validate_reusable_artifact

                validate_reusable_artifact(manifest)
            except ValueError:
                return None
            return manifest
        return None

    def remove_manifest(self, artifact_id: ArtifactIdentifier) -> None:
        self.manifest_path(artifact_id).unlink(missing_ok=True)

    def all_manifests(self) -> tuple[ReusableArtifactManifest, ...]:
        if not self._manifests.is_dir():
            return ()
        return tuple(
            ReusableArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self._manifests.glob("*.json"))
        )
