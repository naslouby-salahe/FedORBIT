from __future__ import annotations

import hashlib
import math
import struct
import unicodedata
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fedorbit.config.loading import active_config
from fedorbit.datasets.common import AdapterSchema, FieldRole
from fedorbit.datasets.splitting import (
    ChronologicalRowCount,
    ChronologicalTimestamp,
    DuplicateGroupChronology,
    DuplicateGroupId,
    DuplicateGroupSplitAssignment,
    assign_duplicate_groups_chronologically,
)

if TYPE_CHECKING:
    from fedorbit.datasets.preprocessing import DuplicateGroups, NormalizedRow

MISSING_TOKEN_VOCABULARY = frozenset({"", "0", "0.0", "nan", "none", "null"})
ABSENT_TOKEN = "<ABSENT>"
RARE_TOKEN = "<RARE>"
UNK_TOKEN = "<UNK>"


class PreprocessingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NormalizedSplitRows:
    duplicate_groups: DuplicateGroups
    split_assignment: DuplicateGroupSplitAssignment


@dataclass(frozen=True, slots=True)
class CandidateFeature:
    name: str
    is_categorical: bool
    train_missing_fraction: float
    train_nonfinite_fraction: float
    dropped: bool
    missing_indicator: bool


@dataclass(frozen=True, slots=True)
class FeatureQualityReport:
    candidate_features: tuple[CandidateFeature, ...]
    dropped_feature_count: int
    client_invalid: bool
    client_invalid_reason: str | None = None

    @property
    def candidate_count_before_filtering(self) -> int:
        return len(self.candidate_features)


@dataclass(frozen=True, slots=True)
class TrainingFeatureValues:
    arrays_by_feature: Mapping[str, np.ndarray]

    def array_of(self, feature_name: str) -> np.ndarray:
        return self.arrays_by_feature[feature_name]


@dataclass(frozen=True, slots=True)
class NumericPreprocessor:
    median: float
    iqr: float
    scale: float
    constant_after_imputation: bool


@dataclass(frozen=True, slots=True)
class CategoricalPreprocessor:
    vocabulary: tuple[str, ...]
    rare_categories: frozenset[str]


def is_missing_token(token: str, categorical: bool) -> bool:
    lowered = token.strip().casefold()
    if lowered in ("nan", "none", "null", ""):
        return True
    return categorical and lowered in ("0", "0.0")


def numeric_zero_is_not_missing(value: float) -> bool:
    return not math.isnan(value) and value == 0.0


def _missing_fraction(values: np.ndarray, categorical: bool) -> float:
    if values.size == 0:
        return 1.0
    if categorical:
        return sum(is_missing_token(str(value), True) for value in values) / values.size
    numeric = values.astype(np.float64)
    return float(np.isnan(numeric).mean())


def _nonfinite_fraction(values: np.ndarray, categorical: bool) -> float:
    if categorical:
        return 0.0
    numeric = values.astype(np.float64)
    return float(np.logical_and(~np.isfinite(numeric), ~np.isnan(numeric)).mean())


def evaluate_feature_quality(
    feature_names: tuple[str, ...],
    categorical_features: frozenset[str],
    train_values: TrainingFeatureValues,
    excluded_features: frozenset[str] = frozenset(),
) -> FeatureQualityReport:
    settings = active_config().scientific.preprocessing
    candidates: list[CandidateFeature] = []
    for name in feature_names:
        if name in excluded_features:
            continue
        categorical = name in categorical_features
        values = train_values.array_of(name)
        missing = _missing_fraction(values, categorical)
        nonfinite = _nonfinite_fraction(values, categorical)
        combined = min(1.0, missing + nonfinite)
        dropped = combined > settings.feature_missing_or_nonfinite_drop_threshold
        candidates.append(
            CandidateFeature(
                name=name,
                is_categorical=categorical,
                train_missing_fraction=missing,
                train_nonfinite_fraction=nonfinite,
                dropped=dropped,
                missing_indicator=(
                    not dropped and missing >= settings.missing_indicator_train_rate_threshold
                ),
            )
        )
    dropped_count = sum(candidate.dropped for candidate in candidates)
    if not candidates:
        return FeatureQualityReport(
            (), 0, True, "zero candidate features after mandatory semantic exclusions"
        )
    invalid = (
        dropped_count / len(candidates)
        > settings.client_invalidity_dropped_feature_fraction_threshold
    )
    return FeatureQualityReport(
        tuple(candidates),
        dropped_count,
        invalid,
        "dropped-feature fraction exceeds the client-invalidity threshold" if invalid else None,
    )


def normalize_training_rows(
    schema: AdapterSchema,
    rows: tuple[NormalizedRow, ...],
) -> DuplicateGroups:
    from fedorbit.datasets.preprocessing import deduplicate_rows, validate_duplicate_groups

    groups = deduplicate_rows(schema, rows)
    validate_duplicate_groups(groups)
    return groups


def assign_duplicate_groups(groups: DuplicateGroups) -> DuplicateGroupSplitAssignment:
    chronology = tuple(
        DuplicateGroupChronology(
            DuplicateGroupId(group_sha256),
            ChronologicalTimestamp(min(member.timestamp_fraction for member in members)),
            ChronologicalRowCount(len(members)),
        )
        for group_sha256, members in groups.groups
    )
    return assign_duplicate_groups_chronologically(chronology)


def normalize_and_split_training_rows(
    schema: AdapterSchema,
    rows: tuple[NormalizedRow, ...],
) -> NormalizedSplitRows:
    groups = normalize_training_rows(schema, rows)
    return NormalizedSplitRows(groups, assign_duplicate_groups(groups))


def fit_numeric_preprocessor(values: np.ndarray) -> NumericPreprocessor:
    numeric = values.astype(np.float64)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        raise PreprocessingError("numeric TRAIN feature has no finite value")
    median = float(np.median(finite))
    q1 = float(np.percentile(finite, 25, method="linear"))
    q3 = float(np.percentile(finite, 75, method="linear"))
    iqr = q3 - q1
    scale = iqr if iqr != 0.0 else 1.0
    imputed = np.where(np.isfinite(numeric), numeric, median)
    constant = bool(np.all(imputed == imputed[0]))
    return NumericPreprocessor(median, iqr, scale, constant)


def transform_numeric(values: np.ndarray, fitted: NumericPreprocessor) -> np.ndarray:
    numeric = values.astype(np.float64)
    imputed = np.where(np.isfinite(numeric), numeric, fitted.median)
    scaled = (imputed - fitted.median) / fitted.scale
    clip = active_config().scientific.preprocessing.numeric_clip
    return np.clip(scaled, clip.lower, clip.upper).astype(np.float64, copy=False)


def categorical_vocabulary(train_categories: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(sorted(set(train_categories), key=lambda token: token.encode("utf-8")))
    return (ABSENT_TOKEN, RARE_TOKEN, UNK_TOKEN, *normalized)


def fit_categorical_preprocessor(values: tuple[str, ...]) -> CategoricalPreprocessor:
    observed = tuple(ABSENT_TOKEN if is_missing_token(value, True) else value for value in values)
    non_missing = tuple(value for value in observed if value != ABSENT_TOKEN)
    total = len(observed)
    counts = OrderedDict((value, non_missing.count(value)) for value in set(non_missing))
    threshold = active_config().scientific.preprocessing.rare_category_train_frequency_threshold
    rare = frozenset(value for value, count in counts.items() if count / total < threshold)
    retained = tuple(value for value in non_missing if value not in rare)
    return CategoricalPreprocessor(categorical_vocabulary(retained), rare)


def transform_categorical(value: str, fitted: CategoricalPreprocessor) -> str:
    if is_missing_token(value, True):
        return ABSENT_TOKEN
    if value in fitted.rare_categories:
        return RARE_TOKEN
    if value not in fitted.vocabulary:
        return UNK_TOKEN
    return value


def one_hot(value: str, fitted: CategoricalPreprocessor) -> tuple[float, ...]:
    transformed = transform_categorical(value, fitted)
    return tuple(1.0 if candidate == transformed else 0.0 for candidate in fitted.vocabulary)


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
