from __future__ import annotations

from dataclasses import dataclass


class TableError(ValueError):
    pass


TableScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class EvidenceTablePayload:
    columns: tuple[str, ...]
    rows: tuple[tuple[TableScalar, ...], ...]


@dataclass(frozen=True, slots=True)
class EvidenceTable:
    columns: tuple[str, ...]
    rows: tuple[tuple[TableScalar, ...], ...]

    def __post_init__(self) -> None:
        if not self.columns:
            raise TableError("evidence table requires at least one column")
        if len(set(self.columns)) != len(self.columns):
            raise TableError("evidence table columns must be unique")
        if any(len(row) != len(self.columns) for row in self.rows):
            raise TableError("evidence table row width differs from column count")

    def payload(self) -> EvidenceTablePayload:
        return EvidenceTablePayload(self.columns, self.rows)
