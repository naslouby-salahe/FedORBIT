from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path


class EdgeLoaderError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EdgeTabularFile:
    relative_path: str
    byte_size: int
    sha256: str
    columns: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_edge_tabular_files(raw_root: Path) -> tuple[Path, ...]:
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    candidates = tuple(
        path
        for path in raw_root.rglob("*.csv")
        if path.is_file() and "description_stats_datasets" not in path.name.casefold()
    )
    if not candidates:
        raise EdgeLoaderError("no Edge-IIoTset tabular traffic CSV files found")
    return tuple(
        sorted(candidates, key=lambda path: path.relative_to(raw_root).as_posix().encode())
    )


def inspect_edge_tabular_files(raw_root: Path) -> tuple[EdgeTabularFile, ...]:
    inspected: list[EdgeTabularFile] = []
    for path in discover_edge_tabular_files(raw_root):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        if not header:
            raise EdgeLoaderError(f"empty tabular file: {path}")
        columns = tuple(header)
        if len(set(columns)) != len(columns):
            raise EdgeLoaderError(f"duplicate columns in {path}")
        inspected.append(
            EdgeTabularFile(
                path.relative_to(raw_root).as_posix(),
                path.stat().st_size,
                _file_sha256(path),
                columns,
            )
        )
    canonical_columns = set(inspected[0].columns)
    for file in inspected[1:]:
        if set(file.columns) != canonical_columns:
            raise EdgeLoaderError(
                f"feature-name set differs from canonical Edge-IIoTset file: {file.relative_path}"
            )
    return tuple(inspected)
