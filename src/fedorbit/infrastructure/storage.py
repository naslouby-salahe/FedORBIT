from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from filelock import FileLock

from fedorbit.types import (
    Sha256Digest,
    StableJsonPayload,
    StorageLayoutSegment,
    TabularColumnName,
    stable_json,
)


class StorageError(ValueError):
    pass


class PayloadIntegrityError(ValueError):
    pass


PAYLOAD_LINE_TERMINATOR = "\n"
TABLE_WRITE_ENGINE = "pyarrow"
TABLE_COMPRESSION = "zstd"
TABLE_FILE_SUFFIX = ".parquet"


def payload_sha256(payload: bytes) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(payload).hexdigest())


def verify_payload_bytes(expected_sha256: Sha256Digest, payload: bytes) -> Sha256Digest:
    observed = payload_sha256(payload)
    if observed != expected_sha256:
        raise PayloadIntegrityError(
            f"payload checksum mismatch: expected {expected_sha256}, observed {observed}"
        )
    return observed


def serialized_payload_bytes(serialized: str) -> bytes:
    return (serialized + PAYLOAD_LINE_TERMINATOR).encode("utf-8")


def stage_bytes(destination: Path, payload: bytes, staging_root: Path) -> Path:
    staged = staged_payload_path(destination, staging_root)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(payload)
    return staged


def staged_payload_path(destination: Path, staging_root: Path) -> Path:
    return staging_root / f"{destination.parent.name}-{destination.name}"


def promote_staged(staged: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, destination)
    return destination


def promote_bytes(destination: Path, payload: bytes, staging_root: Path) -> Path:
    return promote_staged(stage_bytes(destination, payload, staging_root), destination)


def promote_json(destination: Path, payload: StableJsonPayload, staging_root: Path) -> Path:
    return promote_bytes(
        destination,
        (stable_json(payload) + PAYLOAD_LINE_TERMINATOR).encode("utf-8"),
        staging_root,
    )


def table_payload_bytes(
    frame: pd.DataFrame,
    sort_columns: Sequence[TabularColumnName] | None = None,
) -> bytes:
    ordered: pd.DataFrame = frame if sort_columns is None else sort_table(frame, sort_columns)
    sink = io.BytesIO()
    ordered.to_parquet(sink, engine=TABLE_WRITE_ENGINE, compression=TABLE_COMPRESSION, index=False)
    return sink.getvalue()


def promote_table_payload(
    destination: Path,
    frame: pd.DataFrame,
    sort_columns: Sequence[TabularColumnName] | None,
    staging_root: Path,
) -> Path:
    target = promote_bytes(destination, table_payload_bytes(frame, sort_columns), staging_root)
    observed = read_table_payload(target, sort_columns)
    if len(observed) != len(frame):
        raise StorageError(f"promoted tabular payload row count changed: {destination}")
    return target


def sort_table(
    frame: pd.DataFrame,
    sort_columns: Sequence[TabularColumnName],
) -> pd.DataFrame:
    if not sort_columns:
        raise StorageError("tabular payloads require the scientific sort columns")
    missing = tuple(column for column in sort_columns if column not in frame.columns)
    if missing:
        raise StorageError(f"tabular payload lacks its scientific sort columns: {sorted(missing)}")
    ordered: pd.DataFrame = frame.sort_values(list(sort_columns), kind="stable")
    return ordered.reset_index(drop=True)


def read_table_payload(
    path: Path,
    sort_columns: Sequence[TabularColumnName] | None = None,
) -> pd.DataFrame:
    frame: pd.DataFrame = pd.read_parquet(path)
    if sort_columns is None:
        return frame
    return sort_table(frame, sort_columns)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=StorageLayoutSegment.TEMPORARY_FILE_PREFIX
        )
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


def atomic_write_json(path: Path, payload: StableJsonPayload) -> None:
    atomic_write_bytes(path, (stable_json(payload) + "\n").encode("utf-8"))
