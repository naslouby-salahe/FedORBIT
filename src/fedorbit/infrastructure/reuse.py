from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from fedorbit.datasets.common import file_sha256
from fedorbit.infrastructure.manifests import (
    ArtifactLineageRecord,
    ArtifactProvenance,
    CompletionManifest,
    ReusableArtifactManifest,
    completion_manifest_self_hash,
)
from fedorbit.infrastructure.provenance import STAGE_DEPENDENCIES
from fedorbit.types import ArtifactIdentifiers, ArtifactState, TerminalState


class ArtifactValidationError(ValueError):
    pass


def validate_payload_checksum(manifest: ReusableArtifactManifest) -> None:
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


def validate_reusable_artifact(manifest: ReusableArtifactManifest) -> None:
    if manifest.state != ArtifactState.COMPLETED:
        raise ArtifactValidationError(f"artifact {manifest.artifact_id} is not completed")
    validate_payload_checksum(manifest)


def validate_completion_manifest(manifest: CompletionManifest) -> None:
    if not manifest.completion_written_last:
        raise ArtifactValidationError("completion manifest was not written last")
    if manifest.completion_manifest_sha256 != completion_manifest_self_hash(manifest):
        raise ArtifactValidationError("completion manifest self-hash mismatch")


def recorded_upstream_artifact_ids(completion: CompletionManifest) -> ArtifactIdentifiers:
    try:
        record = ArtifactLineageRecord.from_lineage_text(completion.upstream_lineage)
    except (ValidationError, ValueError) as error:
        raise ArtifactValidationError(
            "recorded upstream lineage is not a valid lineage record"
        ) from error
    return record.upstream_artifact_ids


def recorded_lineage(completion: CompletionManifest) -> ArtifactLineageRecord:
    return ArtifactLineageRecord.from_lineage_text(completion.upstream_lineage)


def normalize_completion_lineage(
    manifest: ReusableArtifactManifest,
    completion: CompletionManifest,
    provenance: ArtifactProvenance | None = None,
) -> CompletionManifest:
    validate_artifact_lineage_shape(manifest, completion)
    record = ArtifactLineageRecord(
        upstream_artifact_ids=manifest.upstream_artifact_ids,
        provenance=provenance,
    )
    if recorded_lineage(completion) == record:
        return completion
    if not manifest.upstream_artifact_ids and provenance is None:
        return completion
    draft = completion.model_copy(
        update={
            "upstream_lineage": record.lineage_text(),
            "completion_manifest_sha256": "",
        }
    )
    return draft.model_copy(
        update={"completion_manifest_sha256": completion_manifest_self_hash(draft)}
    )


def validate_artifact_lineage_shape(
    manifest: ReusableArtifactManifest,
    completion: CompletionManifest,
) -> None:
    declared = tuple(identifier.value for identifier in manifest.upstream_artifact_ids)
    recorded_completion = tuple(identifier.value for identifier in completion.upstream_artifact_ids)
    if declared != recorded_completion:
        raise ArtifactValidationError(
            "completion record and reusable manifest disagree on upstream artifact identities"
        )
    if len(set(declared)) != len(declared):
        raise ArtifactValidationError("reusable manifest repeats an upstream artifact identity")


def validate_artifact_lineage(
    manifest: ReusableArtifactManifest,
    completion: CompletionManifest,
) -> None:
    validate_artifact_lineage_shape(manifest, completion)
    declared = tuple(identifier.value for identifier in manifest.upstream_artifact_ids)
    lineage = tuple(identifier.value for identifier in recorded_upstream_artifact_ids(completion))
    if set(lineage) != set(declared):
        raise ArtifactValidationError(
            "reusable manifest and recorded lineage disagree on upstream artifact identities: "
            f"declared={sorted(declared)}, recorded={sorted(lineage)}"
        )


def validate_stage_lineage_completeness(
    manifest: ReusableArtifactManifest,
    completion: CompletionManifest,
) -> None:
    if not STAGE_DEPENDENCIES.get(manifest.producer_stage):
        return
    if not manifest.upstream_artifact_ids:
        raise ArtifactValidationError(
            f"artifact {manifest.artifact_id} at stage {manifest.producer_stage.value} does not "
            "record the upstream artifact identities it was computed from"
        )
    if not recorded_upstream_artifact_ids(completion):
        raise ArtifactValidationError(
            f"artifact {manifest.artifact_id} at stage {manifest.producer_stage.value} did not "
            "persist its upstream artifact lineage"
        )


def validate_completed_artifact(
    manifest: ReusableArtifactManifest,
    completion: CompletionManifest,
) -> None:
    validate_reusable_artifact(manifest)
    validate_completion_manifest(completion)
    if completion.terminal_state != TerminalState.COMPLETED:
        raise ArtifactValidationError("completion record is not completed")
    if completion.dependency_fingerprint_sha256 != manifest.dependency_fingerprint_sha256:
        raise ArtifactValidationError(
            "completion record fingerprint does not match reusable manifest"
        )
    if completion.producer_stage != manifest.producer_stage:
        raise ArtifactValidationError("completion record stage does not match reusable manifest")
    if completion.completion_manifest_sha256 != manifest.completion_manifest_sha256:
        raise ArtifactValidationError("completion record hash does not match reusable manifest")
    validate_artifact_lineage(manifest, completion)
