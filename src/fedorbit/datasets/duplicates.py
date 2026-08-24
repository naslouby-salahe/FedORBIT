from __future__ import annotations

import hashlib
import math
import struct
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from fedorbit.datasets.adapters.schema import (
    BEHAVIORAL_CATEGORICAL_ROLE,
    BEHAVIORAL_NUMERIC_ROLE,
    AdapterSchema,
)
from fedorbit.datasets.preprocessing import is_missing_token


class DuplicateError(ValueError):
    pass


RawFeatureValue = str | int | float | np.float64 | None


@dataclass(frozen=True, slots=True)
class CanonicalFeatureVector:
    values_by_feature: Mapping[str, RawFeatureValue]

    def value_of(self, feature_name: str) -> RawFeatureValue:
        return self.values_by_feature[feature_name]


@dataclass(frozen=True, slots=True)
class PartitionedFeatureValues:
    numeric: CanonicalFeatureVector
    categorical: CanonicalFeatureVector


@dataclass(frozen=True, slots=True)
class CanonicalRow:
    features: CanonicalFeatureVector
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


def _column_bytes_and_validity(value: RawFeatureValue, role: str) -> tuple[int, bytes]:
    if role == BEHAVIORAL_NUMERIC_ROLE:
        missing = value is None or (isinstance(value, float) and math.isnan(value))
        return (0 if missing else 1), _numeric_bytes(value)
    text = unicodedata.normalize("NFC", str(value)).encode("utf-8")
    return 1, struct.pack("<2i", 0, len(text)) + text


def _validity_mask_bits(validity: list[int]) -> bytes:
    mask = bytearray((len(validity) + 7) // 8)
    for index, bit in enumerate(validity):
        if bit:
            mask[index // 8] |= 1 << (index % 8)
    return bytes(mask)


def canonical_row_bytes(row_features: CanonicalFeatureVector, schema: AdapterSchema) -> bytes:
    validity: list[int] = []
    data_buffers: list[bytes] = []
    for column in schema.canonical_feature_order:
        role = schema.role_of(column)
        if role not in (BEHAVIORAL_NUMERIC_ROLE, BEHAVIORAL_CATEGORICAL_ROLE):
            continue
        bit, payload = _column_bytes_and_validity(row_features.value_of(column), role)
        validity.append(bit)
        data_buffers.append(payload)
    return _validity_mask_bits(validity) + b"".join(data_buffers)


def exact_duplicate_hash(row_features: CanonicalFeatureVector, schema: AdapterSchema) -> str:
    return hashlib.sha256(canonical_row_bytes(row_features, schema)).hexdigest()


def deduplicate_rows(
    schema: AdapterSchema,
    rows: tuple[CanonicalRow, ...],
) -> DuplicateGroups:
    groups: dict[str, list[CanonicalRow]] = {}
    for row in rows:
        row_hash = exact_duplicate_hash(row.features, schema)
        groups.setdefault(row_hash, []).append(row)
    return DuplicateGroups(
        groups=tuple((row_hash, tuple(members)) for row_hash, members in groups.items())
    )


@dataclass(frozen=True, slots=True)
class DuplicateGroupMembers:
    group_sha256: str
    members: tuple[CanonicalRow, ...]

    def has_conflicting_labels(self) -> bool:
        return len({member.label for member in self.members}) > 1

    def conflicting_labels(self) -> tuple[str, ...]:
        return tuple(sorted({member.label for member in self.members}))


@dataclass(frozen=True, slots=True)
class DuplicateGroups:
    groups: tuple[tuple[str, tuple[CanonicalRow, ...]], ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for group_sha256, _ in self.groups:
            if group_sha256 in seen:
                raise DuplicateError(f"duplicate group {group_sha256[:16]} appears more than once")
            seen.add(group_sha256)

    @property
    def group_count(self) -> int:
        return len(self.groups)

    def members_of(self, group_sha256: str) -> tuple[CanonicalRow, ...] | None:
        for candidate_sha256, members in self.groups:
            if candidate_sha256 == group_sha256:
                return members
        return None

    def as_member_records(self) -> tuple[DuplicateGroupMembers, ...]:
        return tuple(
            DuplicateGroupMembers(group_sha256=group_sha256, members=members)
            for group_sha256, members in self.groups
        )


def validate_duplicate_groups(groups: DuplicateGroups) -> None:
    for members in groups.as_member_records():
        if members.has_conflicting_labels():
            raise DuplicateError(
                f"duplicate group {members.group_sha256[:16]} contains conflicting labels: "
                f"{members.conflicting_labels()}"
            )


def partition_features(schema: AdapterSchema, row: CanonicalRow) -> PartitionedFeatureValues:
    numeric: dict[str, RawFeatureValue] = {}
    categorical: dict[str, RawFeatureValue] = {}
    for column in schema.canonical_feature_order:
        role = schema.role_of(column)
        if role == BEHAVIORAL_NUMERIC_ROLE:
            numeric[column] = row.features.value_of(column)
        elif role == BEHAVIORAL_CATEGORICAL_ROLE:
            categorical[column] = row.features.value_of(column)
    return PartitionedFeatureValues(
        numeric=CanonicalFeatureVector(numeric),
        categorical=CanonicalFeatureVector(categorical),
    )
