from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from filelock import FileLock

from fedorbit.types import StableJsonPayload, StorageLayoutSegment, stable_json


class StorageError(ValueError):
    pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"): #TODO: use enum instead of hardcoded strings
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
