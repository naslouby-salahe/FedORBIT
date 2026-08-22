from __future__ import annotations

import hashlib
import math
import struct
import unicodedata
from dataclasses import dataclass

import numpy as np

from fedorbit.datasets.adapters.schema import (
    BEHAVIORAL_CATEGORICAL_ROLE,
    BEHAVIORAL_NUMERIC_ROLE,
    AdapterSchema,
)
from fedorbit.datasets.feature_quality import is_missing_token


class DuplicateError(ValueError):
    pass


RawFeatureValue = str | int | float | np.float64 | None


@dataclass(frozen=True, slots=True)
class CanonicalRow:
    features: dict[str, RawFeatureValue]
    label: str
    timestamp_fraction: float
    group_id: str


def canonical_missing_value(is_categorical: bool) -> RawFeatureValue:
    if is_categorical:
        return ""
    return np.float64(np.nan)


def normalize_value(value: RawFeatureValue, is_categorical: bool) -> RawFeatureValue:
    if is_missing_token(str(value).strip(), categorical=is_categorical):
        return canonical_missing_value(is_categorical)
    return value


def _numeric_bytes(value: RawFeatureValue) -> bytes:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return struct.pack("<d", float("nan"))
    return struct.pack("<d", float(str(value)))


def canonical_row_bytes(
    features: dict[str, RawFeatureValue],
    schema_order: tuple[str, ...],
    roles: dict[str, str],
) -> bytes:
    validity: list[int] = []
    data_buffers: list[bytes] = []
    for column in schema_order:
        role = roles.get(column)
        if role not in (BEHAVIORAL_NUMERIC_ROLE, BEHAVIORAL_CATEGORICAL_ROLE):
            continue
        value = features[column]
        if role == BEHAVIORAL_NUMERIC_ROLE:
            missing = value is None or (isinstance(value, float) and math.isnan(value))
            validity.append(0 if missing else 1)
            data_buffers.append(_numeric_bytes(value))
        else:
            text = unicodedata.normalize("NFC", str(value)).encode("utf-8")
            validity.append(1)
            data_buffers.append(struct.pack("<2i", 0, len(text)) + text)
    validity_bytes = bytearray((len(validity) + 7) // 8)
    for index, bit in enumerate(validity):
        if bit:
            validity_bytes[index // 8] |= 1 << (index % 8)
    return bytes(validity_bytes) + b"".join(data_buffers)


def exact_duplicate_hash(
    features: dict[str, RawFeatureValue],
    schema_order: tuple[str, ...],
    roles: dict[str, str],
) -> str:
    return hashlib.sha256(canonical_row_bytes(features, schema_order, roles)).hexdigest()


def deduplicate_rows(
    schema: AdapterSchema,
    rows: tuple[CanonicalRow, ...],
) -> tuple[tuple[str, tuple[CanonicalRow, ...]], ...]:
    groups: dict[str, list[CanonicalRow]] = {}
    for row in rows:
        row_hash = exact_duplicate_hash(row.features, schema.canonical_feature_order, schema.roles)
        groups.setdefault(row_hash, []).append(row)
    return tuple((row_hash, tuple(members)) for row_hash, members in groups.items())


def validate_duplicate_groups(
    groups: tuple[tuple[str, tuple[CanonicalRow, ...]], ...],
) -> None:
    for row_hash, members in groups:
        labels = {member.label for member in members}
        if len(labels) > 1:
            raise DuplicateError(
                f"duplicate group {row_hash[:16]} contains conflicting labels: {sorted(labels)}"
            )


def partition_features(
    schema: AdapterSchema, row: CanonicalRow
) -> tuple[dict[str, RawFeatureValue], dict[str, RawFeatureValue]]:
    numeric: dict[str, RawFeatureValue] = {}
    categorical: dict[str, RawFeatureValue] = {}
    for column in schema.canonical_feature_order:
        role = schema.role_of(column)
        if role == BEHAVIORAL_NUMERIC_ROLE:
            numeric[column] = row.features[column]
        elif role == BEHAVIORAL_CATEGORICAL_ROLE:
            categorical[column] = row.features[column]
    return numeric, categorical
