from __future__ import annotations

from dataclasses import dataclass

from fedorbit.datasets.common import AdapterSchema, FieldRole
from fedorbit.datasets.ontology import canonicalize_label
from fedorbit.datasets.ton_iot.components import TonIotComponent


class TonIotValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TonIotLabelObservation:
    multiclass_label: str
    binary_label: int


def validate_ton_iot_schema(schema: AdapterSchema, component: TonIotComponent) -> None:
    if schema.dataset_id != component.dataset_id:
        raise TonIotValidationError("ToN-IoT schema dataset does not match component")
    if schema.timestamp_column is None:
        raise TonIotValidationError("ToN-IoT schema has no resolved timestamp")
    if schema.multiclass_label_column != "type" or schema.binary_label_column != "label":
        raise TonIotValidationError("ToN-IoT label semantics are unresolved")
    identity_markers = ("src_ip", "dst_ip", "pid", "process_id")
    for column in schema.observed_columns:
        lowered = column.casefold()
        if any(marker in lowered for marker in identity_markers) and (
            schema.role_of(column) != FieldRole.FORBIDDEN_IDENTITY
        ):
            raise TonIotValidationError(f"ToN-IoT identity field retained: {column}")


def validate_ton_iot_label_consistency(rows: tuple[TonIotLabelObservation, ...]) -> None:
    for row in rows:
        is_normal = canonicalize_label(row.multiclass_label) == "normal"
        if row.binary_label not in (0, 1):
            raise TonIotValidationError("binary label must be 0 or 1")
        if is_normal != (row.binary_label == 0):
            raise TonIotValidationError("binary and multiclass label partitions disagree")
