from __future__ import annotations

import json
from pathlib import Path

from fedorbit.artifacts.manifests import ReusableArtifactManifest, file_sha256
from fedorbit.domain.enums import ArtifactState


class ReuseError(ValueError):
    pass


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._manifests = root / "manifests"

    def manifest_path(self, artifact_id: str) -> Path:
        return self._manifests / f"{artifact_id}.json"

    def manifest_dir(self) -> Path:
        return self._manifests

    def write_reusable(self, manifest: ReusableArtifactManifest) -> None:
        self._manifests.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        manifest_path = self.manifest_path(manifest.artifact_id)
        manifest_path.write_text(payload + "\n", encoding="utf-8")

    def read_reusable(self, artifact_id: str) -> ReusableArtifactManifest:
        manifest_path = self.manifest_path(artifact_id)
        if not manifest_path.is_file():
            raise ReuseError(f"no reusable artifact manifest for {artifact_id}")
        return ReusableArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

    def _validate_payload(self, manifest: ReusableArtifactManifest) -> None:
        if manifest.state != ArtifactState.COMPLETED:
            raise ReuseError(f"artifact {manifest.artifact_id} is not reusable")
        for payload_path in manifest.payload_paths:
            path = Path(payload_path)
            if not path.is_file():
                raise ReuseError(f"missing payload for {manifest.artifact_id}: {payload_path}")
            observed = file_sha256(path)
            if observed != manifest.payload_sha256:
                raise ReuseError(
                    f"payload checksum mismatch for {manifest.artifact_id}: "
                    f"expected {manifest.payload_sha256}, observed {observed}"
                )

    def resolve(self, artifact_id: str) -> ReusableArtifactManifest:
        manifest = self.read_reusable(artifact_id)
        self._validate_payload(manifest)
        return manifest

    def find_by_fingerprint(self, fingerprint_sha256: str) -> ReusableArtifactManifest | None:
        if not self._manifests.is_dir():
            return None
        for manifest_path in sorted(self._manifests.glob("*.json")):
            manifest = ReusableArtifactManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            if manifest.dependency_fingerprint_sha256 == fingerprint_sha256:
                self._validate_payload(manifest)
                return manifest
        return None
