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


def discover_ton_iot_component_files(
    raw_root: Path,
    component: TonIotComponent,
) -> tuple[Path, ...]:
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    selected = tuple(raw_root / relative_path for relative_path in component.relative_paths)
    missing = tuple(path for path in selected if not path.is_file())
    if missing:
        raise TonIotLoaderError(
            "selected ToN-IoT component table is absent for "
            f"{component.component_name}: {missing[0]}"
        )
    return selected


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
    from fedorbit.datasets.common import DatasetInspectionError, reconcile_component_columns

    try:
        reconcile_component_columns(tuple(file.columns for file in inspected))
    except DatasetInspectionError as exc:
        raise TonIotLoaderError(str(exc)) from exc
    return tuple(inspected)
