from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from fedorbit.types import EvidenceState, FrozenModel, Sha256Digest

MANIFEST_SUFFIX = ".manifest.json"
DIGEST_BLOCK_BYTES = 1 << 20


class InputDigest(FrozenModel):
    name: str
    sha256: Sha256Digest


class Provenance(FrozenModel):
    config_digest: Sha256Digest
    source_digest: Sha256Digest
    inputs: tuple[InputDigest, ...]


class ArtifactManifest(FrozenModel):
    payload_sha256: Sha256Digest
    payload_bytes: int
    record_count: int
    complete: bool
    code_revision: str
    provenance: Provenance


def file_digest(path: Path) -> Sha256Digest:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(DIGEST_BLOCK_BYTES):
            digest.update(block)
    return Sha256Digest(digest.hexdigest())


def bytes_digest(payload: bytes) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(payload).hexdigest())


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging_name, path)
    except BaseException:
        Path(staging_name).unlink(missing_ok=True)
        raise


def manifest_path(payload_path: Path) -> Path:
    return payload_path.with_name(payload_path.name + MANIFEST_SUFFIX)


def publish_artifact(
    payload_path: Path,
    payload: bytes,
    record_count: int,
    code_revision: str,
    provenance: Provenance,
) -> ArtifactManifest:
    write_atomic(payload_path, payload)
    manifest = ArtifactManifest(
        payload_sha256=bytes_digest(payload),
        payload_bytes=len(payload),
        record_count=record_count,
        complete=True,
        code_revision=code_revision,
        provenance=provenance,
    )
    manifest_text = json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2)
    write_atomic(manifest_path(payload_path), manifest_text.encode("utf-8"))
    return manifest


def read_manifest(payload_path: Path) -> ArtifactManifest | None:
    path = manifest_path(payload_path)
    if not path.is_file():
        return None
    try:
        return ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError):
        return None


def classify_artifact(
    payload_path: Path,
    expected: Provenance,
    expected_records: int | None,
    compare_inputs: bool,
) -> EvidenceState:
    if not payload_path.is_file() and not manifest_path(payload_path).is_file():
        return EvidenceState.MISSING
    manifest = read_manifest(payload_path)
    if manifest is None or not payload_path.is_file():
        return EvidenceState.MALFORMED
    if (
        payload_path.stat().st_size != manifest.payload_bytes
        or file_digest(payload_path) != manifest.payload_sha256
    ):
        return EvidenceState.MALFORMED
    recorded = manifest.provenance
    if (
        recorded.config_digest != expected.config_digest
        or recorded.source_digest != expected.source_digest
    ):
        return EvidenceState.STALE
    if compare_inputs and recorded.inputs != expected.inputs:
        return EvidenceState.STALE
    if not manifest.complete:
        return EvidenceState.INCOMPLETE
    if expected_records is not None and manifest.record_count != expected_records:
        return EvidenceState.INCOMPLETE
    return EvidenceState.VALID
