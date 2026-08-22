from __future__ import annotations

import hashlib
from pathlib import Path

from fedorbit.config.models import FrozenModel
from fedorbit.domain.canonical import canonical_json
from fedorbit.domain.enums import ArtifactState, TerminalState


class CompletionManifest(FrozenModel):
    schema_version: str
    semantic_experiment_coordinates: str
    producer_stage: str
    terminal_state: TerminalState
    dependency_fingerprint_sha256: str
    upstream_artifact_ids: tuple[str, ...]
    mandatory_artifact_paths: tuple[str, ...]
    mandatory_artifact_sha256: str
    scientific_configuration_sha256: str
    relevant_code_sha256: str
    material_runtime_sha256: str
    upstream_lineage: str
    completion_validation_state: str
    completion_written_last: bool
    completion_manifest_sha256: str


class ReusableArtifactManifest(FrozenModel):
    artifact_id: str
    artifact_type: str
    semantic_producer_coordinates: str
    producer_stage: str
    dependency_fingerprint_sha256: str
    upstream_artifact_ids: tuple[str, ...]
    applicable_configuration_sha256: str
    relevant_code_sha256: str
    material_runtime_sha256: str
    payload_paths: tuple[str, ...]
    payload_sha256: str
    schema_version: str
    created_git_commit: str
    created_environment_sha256: str
    state: ArtifactState
    completion_manifest_sha256: str


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_fingerprint(
    coordinates: object,
    upstream_artifact_ids: tuple[str, ...],
    configuration_sha256: str,
    code_sha256: str,
    runtime_sha256: str,
) -> str:
    payload = canonical_json(
        {
            "coordinates": coordinates,
            "upstream_artifact_ids": list(upstream_artifact_ids),
            "configuration_sha256": configuration_sha256,
            "code_sha256": code_sha256,
            "runtime_sha256": runtime_sha256,
        }
    )
    return _sha256(payload)


def artifact_id(artifact_type: str, coordinates: object, fingerprint_sha256: str) -> str:
    payload = canonical_json(
        {
            "artifact_type": artifact_type,
            "coordinates": coordinates,
            "dependency_fingerprint_sha256": fingerprint_sha256,
        }
    )
    return _sha256(payload)


def completion_manifest_self_hash(manifest: CompletionManifest) -> str:
    payload = canonical_json(
        manifest.model_dump(mode="json", exclude={"completion_manifest_sha256"})
    )
    return _sha256(payload)
