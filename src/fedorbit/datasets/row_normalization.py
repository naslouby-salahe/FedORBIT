from __future__ import annotations

import hashlib
import math
import struct
import unicodedata
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from fedorbit.datasets.common import AdapterSchema, FieldRole
from fedorbit.datasets.preprocessing import is_missing_token


class RowNormalizationError(ValueError):
    pass


RawFeatureValue = str | int | float | np.float64 | None


@dataclass(frozen=True, slots=True)
class NormalizedFeatureVector:
    values_by_feature: Mapping[str, RawFeatureValue]

    def value_of(self, feature_name: str) -> RawFeatureValue:
        return self.values_by_feature[feature_name]


@dataclass(frozen=True, slots=True)
class PartitionedFeatureValues:
    numeric: NormalizedFeatureVector
    categorical: NormalizedFeatureVector


@dataclass(frozen=True, slots=True)
class NormalizedRow:
    features: NormalizedFeatureVector
    label: str
    timestamp_fraction: float
    group_id: str


@dataclass(frozen=True, slots=True)
class DuplicateGroupMembers:
    group_sha256: str
    members: tuple[NormalizedRow, ...]

    def has_conflicting_labels(self) -> bool:
        return len({member.label for member in self.members}) > 1

    def conflicting_labels(self) -> tuple[str, ...]:
        return tuple(sorted({member.label for member in self.members}))


@dataclass(frozen=True, slots=True)
class DuplicateGroups:
    groups: tuple[tuple[str, tuple[NormalizedRow, ...]], ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for group_sha256, _ in self.groups:
            if group_sha256 in seen:
                raise RowNormalizationError(
                    f"duplicate group {group_sha256[:16]} appears more than once"
                )
            seen.add(group_sha256)

    @property
    def group_count(self) -> int:
        return len(self.groups)

    def members_of(self, group_sha256: str) -> tuple[NormalizedRow, ...] | None:
        for candidate_sha256, members in self.groups:
            if candidate_sha256 == group_sha256:
                return members
        return None

    def as_member_records(self) -> tuple[DuplicateGroupMembers, ...]:
        return tuple(
            DuplicateGroupMembers(group_sha256, members) for group_sha256, members in self.groups
        )


def normalized_missing_value(is_categorical: bool) -> RawFeatureValue:
    return "" if is_categorical else np.float64(np.nan)


def normalize_value(value: RawFeatureValue, is_categorical: bool) -> RawFeatureValue:
    if is_missing_token(str(value).strip(), categorical=is_categorical):
        return normalized_missing_value(is_categorical)
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, float) and not math.isfinite(float(value)):
        return normalized_missing_value(False)
    return value


def _numeric_bytes(value: RawFeatureValue) -> bytes:
    if value is None:
        return struct.pack("<d", float("nan"))
    numeric = float(value)
    if math.isnan(numeric):
        return struct.pack("<d", float("nan"))
    return struct.pack("<d", numeric)


def _column_bytes_and_validity(value: RawFeatureValue, role: FieldRole) -> tuple[int, bytes]:
    if role == FieldRole.BEHAVIORAL_NUMERIC:
        missing = value is None or (
            isinstance(value, (float, np.float64)) and math.isnan(float(value))
        )
        return (0 if missing else 1), _numeric_bytes(value)
    text = unicodedata.normalize("NFC", str(value)).encode("utf-8")
    return 1, struct.pack("<2i", 0, len(text)) + text


def _validity_mask_bits(validity: tuple[int, ...]) -> bytes:
    mask = bytearray((len(validity) + 7) // 8)
    for index, bit in enumerate(validity):
        if bit:
            mask[index // 8] |= 1 << (index % 8)
    return bytes(mask)


def normalized_row_bytes(row_features: NormalizedFeatureVector, schema: AdapterSchema) -> bytes:
    encoded = tuple(
        _column_bytes_and_validity(row_features.value_of(column), schema.role_of(column))
        for column in schema.feature_order
        if schema.role_of(column)
        in (FieldRole.BEHAVIORAL_NUMERIC, FieldRole.BEHAVIORAL_CATEGORICAL)
    )
    validity = tuple(bit for bit, _ in encoded)
    return _validity_mask_bits(validity) + b"".join(payload for _, payload in encoded)


def exact_duplicate_hash(row_features: NormalizedFeatureVector, schema: AdapterSchema) -> str:
    return hashlib.sha256(normalized_row_bytes(row_features, schema)).hexdigest()


def deduplicate_rows(schema: AdapterSchema, rows: tuple[NormalizedRow, ...]) -> DuplicateGroups:
    groups: defaultdict[str, list[NormalizedRow]] = defaultdict(list)
    for row in rows:
        row_hash = exact_duplicate_hash(row.features, schema)
        groups.setdefault(row_hash, []).append(row)
    return DuplicateGroups(
        tuple((row_hash, tuple(members)) for row_hash, members in sorted(groups.items()))
    )


def validate_duplicate_groups(groups: DuplicateGroups) -> None:
    for members in groups.as_member_records():
        if members.has_conflicting_labels():
            raise RowNormalizationError(
                f"duplicate group {members.group_sha256[:16]} contains conflicting labels: "
                f"{members.conflicting_labels()}"
            )


def partition_features(schema: AdapterSchema, row: NormalizedRow) -> PartitionedFeatureValues:
    numeric: OrderedDict[str, RawFeatureValue] = OrderedDict()
    categorical: OrderedDict[str, RawFeatureValue] = OrderedDict()
    for column in schema.feature_order:
        role = schema.role_of(column)
        if role == FieldRole.BEHAVIORAL_NUMERIC:
            numeric[column] = row.features.value_of(column)
        elif role == FieldRole.BEHAVIORAL_CATEGORICAL:
            categorical[column] = row.features.value_of(column)
    return PartitionedFeatureValues(
        NormalizedFeatureVector(numeric),
        NormalizedFeatureVector(categorical),
    )
