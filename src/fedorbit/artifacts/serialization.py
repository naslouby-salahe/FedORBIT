from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from fedorbit.domain.canonical import canonical_json


class SerializationError(ValueError):
    pass


def atomic_write_json(path: Path, payload: object) -> None:
    rendered = canonical_json(payload) + "\n"
    _atomic_write(path, rendered.encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    _atomic_write(path, data)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
