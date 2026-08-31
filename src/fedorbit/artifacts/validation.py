from __future__ import annotations

from pathlib import Path

from fedorbit.artifacts.manifests import (
    CompletionManifest,
    ReusableArtifactManifest,
    completion_manifest_self_hash,
    file_sha256,
)
from fedorbit.domain.enums import ArtifactState


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
