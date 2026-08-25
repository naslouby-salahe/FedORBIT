from __future__ import annotations

from dataclasses import dataclass

from fedorbit.datasets.common import AdapterSchema, FieldRole
from fedorbit.datasets.edge_iiotset.schema import (
    EDGE_BINARY_LABEL,
    EDGE_LEAKAGE_SAFEGUARD_EXCLUSIONS,
    EDGE_MULTICLASS_LABEL,
)
from fedorbit.datasets.ontology import normalize_label


class EdgeValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LabelObservation:
    multiclass_label: str
    binary_label: int


def validate_edge_schema(schema: AdapterSchema) -> None:
    if schema.timestamp_column is None:
        raise EdgeValidationError("Edge-IIoTset schema has no resolved timestamp")
    if schema.multiclass_label_column != EDGE_MULTICLASS_LABEL:
        raise EdgeValidationError("Edge-IIoTset multiclass label semantics are unresolved")
    if schema.binary_label_column != EDGE_BINARY_LABEL:
        raise EdgeValidationError("Edge-IIoTset binary label semantics are unresolved")
    for field in EDGE_LEAKAGE_SAFEGUARD_EXCLUSIONS:
        if field in schema.observed_columns and schema.role_of(field) in (
            FieldRole.BEHAVIORAL_NUMERIC,
            FieldRole.BEHAVIORAL_CATEGORICAL,
        ):
            raise EdgeValidationError(f"Edge-IIoTset leakage safeguard field retained: {field}")


def validate_binary_multiclass_consistency(rows: tuple[LabelObservation, ...]) -> None:
    for row in rows:
        is_normal = normalize_label(row.multiclass_label) == "normal"
        if row.binary_label not in (0, 1):
            raise EdgeValidationError("binary label must be 0 or 1")
        if is_normal != (row.binary_label == 0):
            raise EdgeValidationError("binary and multiclass label partitions disagree")
