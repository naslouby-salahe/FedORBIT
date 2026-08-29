from __future__ import annotations

import csv
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class TabularMaterializationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TabularMaterializationRequest:
    files: tuple[Path, ...]

    def __post_init__(self) -> None:
        if not self.files:
            raise TabularMaterializationError("tabular materialization requires input files")


@dataclass(frozen=True, slots=True)
class MaterializedTabularRow:
    source_path: Path
    values: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class MaterializedTabularRows:
    columns: tuple[str, ...]
    rows: tuple[MaterializedTabularRow, ...]


def materialize_tabular_rows(request: TabularMaterializationRequest) -> MaterializedTabularRows:
    columns: tuple[str, ...] | None = None
    rows: list[MaterializedTabularRow] = []
    for path in request.files:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            observed = tuple(reader.fieldnames or ())
            if not observed or len(set(observed)) != len(observed):
                raise TabularMaterializationError(f"invalid CSV header: {path}")
            if columns is None:
                columns = observed
            elif observed != columns:
                raise TabularMaterializationError(f"CSV schema differs from first input: {path}")
            for raw in reader:
                rows.append(
                    MaterializedTabularRow(
                        source_path=path,
                        values=OrderedDict(
                            (column, raw[column] if raw[column] is not None else "")
                            for column in observed
                        ),
                    )
                )
    if columns is None:
        raise TabularMaterializationError("tabular materialization produced no schema")
    return MaterializedTabularRows(columns=columns, rows=tuple(rows))
