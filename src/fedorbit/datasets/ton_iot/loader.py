from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from fedorbit.datasets.ton_iot.components import TonIotComponent


class TonIotLoaderError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TonIotTabularFile:
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


def _matches_component(path: Path, raw_root: Path, component: TonIotComponent) -> bool:
    relative = path.relative_to(raw_root).as_posix().casefold()
    return all(token in relative for token in component.required_path_tokens) and not any(
        token in relative for token in component.forbidden_path_tokens
    )


def discover_ton_iot_component_files(
    raw_root: Path,
    component: TonIotComponent,
) -> tuple[Path, ...]:
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    candidates = tuple(
        path
        for path in raw_root.rglob("*.csv")
        if path.is_file() and _matches_component(path, raw_root, component)
    )
    if not candidates:
        raise TonIotLoaderError(f"no files found for ToN-IoT component {component.component_name}")
    return tuple(
        sorted(candidates, key=lambda path: path.relative_to(raw_root).as_posix().encode())
    )


def inspect_ton_iot_component_files(
    raw_root: Path,
    component: TonIotComponent,
) -> tuple[TonIotTabularFile, ...]:
    inspected: list[TonIotTabularFile] = []
    for path in discover_ton_iot_component_files(raw_root, component):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        if not header:
            raise TonIotLoaderError(f"empty tabular file: {path}")
        columns = tuple(header)
        if len(set(columns)) != len(columns):
            raise TonIotLoaderError(f"duplicate columns in {path}")
        inspected.append(
            TonIotTabularFile(
                path.relative_to(raw_root).as_posix(),
                path.stat().st_size,
                _file_sha256(path),
                columns,
            )
        )
    canonical_columns = set(inspected[0].columns)
    for file in inspected[1:]:
        if set(file.columns) != canonical_columns:
            message = (
                f"feature-name set differs within component {component.component_name}: "
                f"{file.relative_path}"
            )
            raise TonIotLoaderError(message)
    return tuple(inspected)
