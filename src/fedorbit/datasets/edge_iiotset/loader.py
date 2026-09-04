from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from fedorbit.types import ByteCount


class EdgeLoaderError(ValueError):
    pass


EDGE_NETWORK_RELATIVE_PATH = (
    "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
)


@dataclass(frozen=True, slots=True)
class EdgeTabularFile:
    relative_path: str
    byte_size: ByteCount
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
    selected = raw_root / EDGE_NETWORK_RELATIVE_PATH
    if not selected.is_file():
        raise EdgeLoaderError(f"selected Edge-IIoTset network table is absent: {selected}")
    return (selected,)


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
    expected_columns = set(inspected[0].columns)
    for file in inspected[1:]:
        if set(file.columns) != expected_columns:
            raise EdgeLoaderError(
                f"feature-name set differs from stable Edge-IIoTset file: {file.relative_path}"
            )
    return tuple(inspected)
