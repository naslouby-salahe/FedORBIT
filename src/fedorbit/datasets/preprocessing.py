from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from fedorbit.config.models import FedorbitConfig

MISSING_TOKEN_VOCABULARY = frozenset({"", "0", "0.0", "nan", "none", "null"})
ABSENT_TOKEN = "<ABSENT>"
RARE_TOKEN = "<RARE>"
UNK_TOKEN = "<UNK>"


class PreprocessingError(ValueError):
    pass


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
    config: FedorbitConfig,
    feature_names: tuple[str, ...],
    categorical_features: frozenset[str],
    train_values: TrainingFeatureValues,
    excluded_features: frozenset[str] = frozenset(),
) -> FeatureQualityReport:
    settings = config.scientific.preprocessing
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


def fit_numeric_preprocessor(values: np.ndarray) -> NumericPreprocessor:
    numeric = values.astype(np.float64)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        raise PreprocessingError("numeric TRAIN feature has no finite value")
    median = float(np.quantile(finite, 0.5, method="linear"))
    q1 = float(np.quantile(finite, 0.25, method="linear"))
    q3 = float(np.quantile(finite, 0.75, method="linear"))
    iqr = q3 - q1
    scale = iqr if iqr != 0.0 else 1.0
    imputed = np.where(np.isfinite(numeric), numeric, median)
    constant = bool(np.all(imputed == imputed[0]))
    return NumericPreprocessor(median, iqr, scale, constant)


def transform_numeric(
    config: FedorbitConfig,
    values: np.ndarray,
    fitted: NumericPreprocessor,
) -> np.ndarray:
    numeric = values.astype(np.float64)
    imputed = np.where(np.isfinite(numeric), numeric, fitted.median)
    scaled = (imputed - fitted.median) / fitted.scale
    clip = config.scientific.preprocessing.numeric_clip
    return np.clip(scaled, clip.lower, clip.upper).astype(np.float64, copy=False)


def categorical_vocabulary(train_categories: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(sorted(set(train_categories), key=lambda token: token.encode("utf-8")))
    return (ABSENT_TOKEN, RARE_TOKEN, UNK_TOKEN, *normalized)


def fit_categorical_preprocessor(
    config: FedorbitConfig,
    values: tuple[str, ...],
) -> CategoricalPreprocessor:
    observed = tuple(ABSENT_TOKEN if is_missing_token(value, True) else value for value in values)
    non_missing = tuple(value for value in observed if value != ABSENT_TOKEN)
    total = len(observed)
    counts = {value: non_missing.count(value) for value in set(non_missing)}
    threshold = config.scientific.preprocessing.rare_category_train_frequency_threshold
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
