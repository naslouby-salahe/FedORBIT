from __future__ import annotations

import hashlib
import math
import struct
import unicodedata
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
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
from fedorbit.types import (
    Coefficient,
    CategoryName,
    CategorySet,
    CategoryVocabulary,
    DuplicateGroupIdentifier,
    FeatureCount,
    FeatureValue,
    FeatureValueMap,
    Fraction,
    FineLabel,
    Index,
    LocalClassNames,
    NonNegativeInt,
    NumericFeatureValue,
    NormalizedGroupIdentifier,
    RawCellText,
    ScaleFactor,
    Sha256Digest,
    TabularColumnName,
    TabularColumns,
    TabularColumnSet,
    TextToken,
    ValidationReason,
)

if TYPE_CHECKING:
    from fedorbit.datasets.preprocessing import DuplicateGroups, NormalizedRow

class PreprocessingToken(StrEnum):
    EMPTY = ""
    ZERO = "0"
    DECIMAL_ZERO = "0.0"
    NAN = "nan"
    NONE = "none"
    NULL = "null"
    ABSENT = "<ABSENT>"
    RARE = "<RARE>"
    UNKNOWN = "<UNK>"


class UnicodeNormalizationForm(StrEnum):
    NFC = "NFC"


MISSING_TOKEN_VOCABULARY = frozenset(
    token.value
    for token in (
        PreprocessingToken.EMPTY,
        PreprocessingToken.ZERO,
        PreprocessingToken.DECIMAL_ZERO,
        PreprocessingToken.NAN,
        PreprocessingToken.NONE,
        PreprocessingToken.NULL,
    )
)


class PreprocessingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NormalizedSplitRows:
    duplicate_groups: DuplicateGroups
    split_assignment: DuplicateGroupSplitAssignment


@dataclass(frozen=True, slots=True)
class CandidateFeature:
    name: TabularColumnName
    is_categorical: bool
    train_missing_fraction: Fraction
    train_nonfinite_fraction: Fraction
    dropped: bool
    missing_indicator: bool


@dataclass(frozen=True, slots=True)
class FeatureQualityReport:
    candidate_features: tuple[CandidateFeature, ...]
    dropped_feature_count: Index
    client_invalid: bool
    client_invalid_reason: ValidationReason | None = None

    @property
    def candidate_count_before_filtering(self) -> FeatureCount:
        return FeatureCount(len(self.candidate_features))


@dataclass(frozen=True, slots=True)
class TrainingFeatureValues:
    arrays_by_feature: Mapping[TabularColumnName, np.ndarray]

    def array_of(self, feature_name: TabularColumnName) -> np.ndarray:
        return self.arrays_by_feature[feature_name]


@dataclass(frozen=True, slots=True)
class NumericPreprocessor:
    median: Coefficient
    iqr: Coefficient
    scale: ScaleFactor
    constant_after_imputation: bool


@dataclass(frozen=True, slots=True)
class CategoricalPreprocessor:
    vocabulary: CategoryVocabulary
    rare_categories: CategorySet


def is_missing_token(token: TextToken, categorical: bool) -> bool:
    lowered = token.strip().casefold()
    if lowered in {
        PreprocessingToken.NAN,
        PreprocessingToken.NONE,
        PreprocessingToken.NULL,
        PreprocessingToken.EMPTY,
    }:
        return True
    return categorical and lowered in {
        PreprocessingToken.ZERO,
        PreprocessingToken.DECIMAL_ZERO,
    }


def numeric_zero_is_not_missing(value: NumericFeatureValue) -> bool:
    return not math.isnan(value) and value == 0.0


def _missing_fraction(values: np.ndarray, categorical: bool) -> Fraction:
    if values.size == 0:
        return Fraction(1.0)
    if categorical:
        return Fraction(
            sum(is_missing_token(RawCellText(str(value)), True) for value in values) / values.size
        )
    numeric = values.astype(np.float64)
    return Fraction(float(np.isnan(numeric).mean()))


def _nonfinite_fraction(values: np.ndarray, categorical: bool
                        ) -> Fraction:
    if categorical:
        return Fraction(0.0)
    numeric = values.astype(np.float64)
    return Fraction(float(np.logical_and(~np.isfinite(numeric), ~np.isnan(numeric)).mean()))


def evaluate_feature_quality(
    feature_names: TabularColumns,
    categorical_features: TabularColumnSet,
    train_values: TrainingFeatureValues,
    excluded_features: TabularColumnSet = frozenset(),
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
            (),
            0,
            True,
            ValidationReason("zero candidate features after mandatory semantic exclusions"),
        )
    invalid = (
        dropped_count / len(candidates)
        > settings.client_invalidity_dropped_feature_fraction_threshold
    )
    return FeatureQualityReport(
        tuple(candidates),
        dropped_count,
        invalid,
        ValidationReason("dropped-feature fraction exceeds the client-invalidity threshold")
        if invalid
        else None,
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
            DuplicateGroupId(DuplicateGroupIdentifier(group_sha256)),
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


def categorical_vocabulary(train_categories: CategoryVocabulary) -> CategoryVocabulary:
    normalized = tuple(sorted(set(train_categories), key=lambda token: token.encode("utf-8")))
    return (
        CategoryName(PreprocessingToken.ABSENT),
        CategoryName(PreprocessingToken.RARE),
        CategoryName(PreprocessingToken.UNKNOWN),
        *normalized,
    )


def fit_categorical_preprocessor(values: tuple[RawCellText, ...]) -> CategoricalPreprocessor:
    observed = tuple(
        CategoryName(PreprocessingToken.ABSENT)
        if is_missing_token(value, True)
        else CategoryName(value)
        for value in values
    )
    non_missing = tuple(
        value for value in observed if value != CategoryName(PreprocessingToken.ABSENT)
    )
    total = len(observed)
    counts = OrderedDict((value, non_missing.count(value)) for value in set(non_missing))
    threshold = active_config().scientific.preprocessing.rare_category_train_frequency_threshold
    rare = frozenset(value for value, count in counts.items() if count / total < threshold)
    retained = tuple(value for value in non_missing if value not in rare)
    return CategoricalPreprocessor(categorical_vocabulary(retained), rare)


def transform_categorical(
    value: RawCellText, fitted: CategoricalPreprocessor
) -> CategoryName:
    if is_missing_token(value, True):
        return CategoryName(PreprocessingToken.ABSENT)
    category = CategoryName(value)
    if category in fitted.rare_categories:
        return CategoryName(PreprocessingToken.RARE)
    if category not in fitted.vocabulary:
        return CategoryName(PreprocessingToken.UNKNOWN)
    return category


def one_hot(
    value: RawCellText, fitted: CategoricalPreprocessor
) -> tuple[Coefficient, ...]:
    transformed = transform_categorical(value, fitted)
    return tuple(
        Coefficient(1.0 if candidate == transformed else 0.0) for candidate in fitted.vocabulary
    )


class RowNormalizationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NormalizedFeatureVector:
    values_by_feature: FeatureValueMap

    def value_of(self, feature_name: TabularColumnName) -> FeatureValue:
        return self.values_by_feature[feature_name]


@dataclass(frozen=True, slots=True)
class NormalizedRow:
    features: NormalizedFeatureVector
    label: FineLabel
    timestamp_fraction: Fraction
    group_id: NormalizedGroupIdentifier


@dataclass(frozen=True, slots=True)
class DuplicateGroupMembers:
    group_sha256: Sha256Digest
    members: tuple[NormalizedRow, ...]

    def has_conflicting_labels(self) -> bool:
        return len({member.label for member in self.members}) > 1

    def conflicting_labels(self) -> LocalClassNames:
        return tuple(sorted({member.label for member in self.members}))


@dataclass(frozen=True, slots=True)
class DuplicateGroups:
    groups: tuple[tuple[Sha256Digest, tuple[NormalizedRow, ...]], ...]

    def __post_init__(self) -> None:
        seen: set[Sha256Digest] = set()
        for group_sha256, _ in self.groups:
            if group_sha256 in seen:
                raise RowNormalizationError(
                    f"duplicate group {group_sha256[:16]} appears more than once"
                )
            seen.add(group_sha256)

    @property
    def group_count(self) -> NonNegativeInt:
        return len(self.groups)

    def members_of(self, group_sha256: Sha256Digest) -> tuple[NormalizedRow, ...] | None:
        for candidate_sha256, members in self.groups:
            if candidate_sha256 == group_sha256:
                return members
        return None

    def as_member_records(self) -> tuple[DuplicateGroupMembers, ...]:
        return tuple(
            DuplicateGroupMembers(group_sha256, members) for group_sha256, members in self.groups
        )


def normalized_missing_value(is_categorical: bool) -> FeatureValue:
    return RawCellText("") if is_categorical else float("nan")


def normalize_value(value: FeatureValue, is_categorical: bool) -> FeatureValue:
    if is_missing_token(RawCellText(str(value).strip()), categorical=is_categorical):
        return normalized_missing_value(is_categorical)
    if isinstance(value, str):
        return RawCellText(unicodedata.normalize(UnicodeNormalizationForm.NFC, value))
    if isinstance(value, float) and not math.isfinite(float(value)):
        return normalized_missing_value(False)
    return value


@lru_cache(maxsize=None)
def _numeric_scalar_bytes(value: NumericFeatureValue) -> bytes:
    return struct.pack("<d", value)


@lru_cache(maxsize=None)
def _categorical_scalar_bytes(value: RawCellText) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<i", len(encoded)) + encoded


def normalized_row_bytes(row_features: NormalizedFeatureVector, schema: AdapterSchema) -> bytes:
    parts: list[bytes] = [] #TODO: PERF: duplicate-hash path serializes every feature of every row in Python - memoize per-value encodings and prefer column-wise row hashing
    for column in schema.feature_order:
        role = schema.role_of(column)
        if role not in (FieldRole.BEHAVIORAL_NUMERIC, FieldRole.BEHAVIORAL_CATEGORICAL):
            continue
        value = row_features.value_of(column)
        if role == FieldRole.BEHAVIORAL_NUMERIC:
            numeric_value = NumericFeatureValue(
                float("nan") if value is None else float(value)
            )
            parts.append(_numeric_scalar_bytes(numeric_value))
        else:
            categorical_value = RawCellText(
                unicodedata.normalize(UnicodeNormalizationForm.NFC, str(value))
            )
            parts.append(_categorical_scalar_bytes(categorical_value))
    return b"".join(parts)


def exact_duplicate_hash(
    row_features: NormalizedFeatureVector, schema: AdapterSchema
) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(normalized_row_bytes(row_features, schema)).hexdigest())


def deduplicate_rows(schema: AdapterSchema, rows: tuple[NormalizedRow, ...]) -> DuplicateGroups:
    groups: defaultdict[Sha256Digest, list[NormalizedRow]] = defaultdict(list)
    for row in rows:
        row_hash = exact_duplicate_hash(row.features, schema) #TODO: PERF: group duplicate rows column-wise (sort/tabulate normalized column tuples) instead of one sha256 per row; encode each distinct row once
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
