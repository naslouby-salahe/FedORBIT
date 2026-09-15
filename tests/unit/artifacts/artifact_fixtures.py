from __future__ import annotations

from pathlib import Path

from fedorbit.datasets.common import file_sha256
from fedorbit.infrastructure.manifests import (
    CompletionManifest,
    ReusableArtifactManifest,
    artifact_id,
    completion_manifest_self_hash,
)
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactSchemaVersion,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    CompletionValidationState,
    SemanticCoordinateText,
    Sha256Digest,
    TerminalState,
)

EXPERIMENT_COORDINATES = SemanticCoordinateText(
    '{"experiment":"Mathematical Primitive Validation","seed":0}'
)
RAW_COORDINATES = "{}"
COMMIT = "e" * 40
CONFIGURATION_SHA256 = Sha256Digest("b" * 64)
CODE_SHA256 = Sha256Digest("c" * 64)


def payload_file(directory: Path, name: str, content: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def artifact_manifest(
    payload_path: Path,
    fingerprint: Sha256Digest,
    stage: ArtifactStage,
    upstream_artifact_ids: ArtifactIdentifiers = (),
    artifact_type: ArtifactType = ArtifactType.OTHER,
    state: ArtifactState = ArtifactState.COMPLETED,
    coordinates: SemanticCoordinateText = EXPERIMENT_COORDINATES,
) -> ReusableArtifactManifest:
    return ReusableArtifactManifest.model_validate(
        {
            "artifact_id": artifact_id(artifact_type, RAW_COORDINATES, fingerprint),
            "artifact_type": artifact_type,
            "semantic_producer_coordinates": coordinates,
            "producer_stage": stage,
            "dependency_fingerprint_sha256": fingerprint,
            "upstream_artifact_ids": upstream_artifact_ids,
            "applicable_configuration_sha256": CONFIGURATION_SHA256,
            "relevant_code_sha256": CODE_SHA256,
            "payload_paths": (str(payload_path),),
            "payload_sha256": file_sha256(payload_path),
            "schema_version": ArtifactSchemaVersion.V1,
            "created_git_commit": COMMIT,
            "created_environment_sha256": Sha256Digest("f" * 64),
            "state": state,
            "completion_required": True,
            "completion_manifest_sha256": Sha256Digest("0" * 64),
        }
    )


def artifact_completion(
    manifest: ReusableArtifactManifest,
    upstream_lineage: str = "{}",
) -> CompletionManifest:
    draft = CompletionManifest.model_validate(
        {
            "schema_version": ArtifactSchemaVersion.V1,
            "semantic_experiment_coordinates": manifest.semantic_producer_coordinates,
            "producer_stage": manifest.producer_stage,
            "terminal_state": TerminalState.COMPLETED,
            "dependency_fingerprint_sha256": manifest.dependency_fingerprint_sha256,
            "upstream_artifact_ids": manifest.upstream_artifact_ids,
            "mandatory_artifact_paths": manifest.payload_paths,
            "mandatory_artifact_sha256": manifest.payload_sha256,
            "scientific_configuration_sha256": manifest.applicable_configuration_sha256,
            "relevant_code_sha256": manifest.relevant_code_sha256,
            "upstream_lineage": upstream_lineage,
            "completion_validation_state": CompletionValidationState.VALIDATED,
            "completion_written_last": True,
            "completion_manifest_sha256": "",
        }
    )
    return draft.model_copy(
        update={"completion_manifest_sha256": completion_manifest_self_hash(draft)}
    )


def published_artifact_ids(
    manifests: tuple[ReusableArtifactManifest, ...],
) -> tuple[ArtifactIdentifier, ...]:
    return tuple(manifest.artifact_id for manifest in manifests)
