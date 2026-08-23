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


class FeatureQualityError(ValueError):
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


def _missing_fraction(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    missing_mask = np.zeros(values.shape, dtype=bool)
    if values.dtype.kind in "f":
        missing_mask = np.isnan(values.astype(float))
    elif values.dtype.kind in "OUS":
        lowered = np.char.lower(values.astype(str))
        for token in MISSING_TOKEN_VOCABULARY:
            missing_mask |= lowered == token
    return float(missing_mask.mean())


def _nonfinite_fraction(values: np.ndarray) -> float:
    if values.dtype.kind not in "f":
        return 0.0
    return float(np.logical_not(np.isfinite(values.astype(float))).mean())


@dataclass(frozen=True, slots=True)
class TrainingFeatureValues:
    arrays_by_feature: Mapping[str, np.ndarray]

    def array_of(self, feature_name: str) -> np.ndarray:
        return self.arrays_by_feature[feature_name]


def evaluate_feature_quality(
    config: FedorbitConfig,
    feature_names: tuple[str, ...],
    categorical_features: frozenset[str],
    train_values: TrainingFeatureValues,
    excluded_features: frozenset[str] = frozenset(),
) -> FeatureQualityReport:
    preprocessing = config.scientific.preprocessing
    drop_threshold = preprocessing.feature_missing_or_nonfinite_drop_threshold
    indicator_threshold = preprocessing.missing_indicator_train_rate_threshold
    invalid_fraction_threshold = preprocessing.client_invalidity_dropped_feature_fraction_threshold

    candidates: list[CandidateFeature] = []
    for name in feature_names:
        if name in excluded_features:
            continue
        values = train_values.array_of(name)
        missing = _missing_fraction(values)
        nonfinite = _nonfinite_fraction(values)
        combined = missing + (nonfinite if values.dtype.kind in "f" else 0.0)
        combined = min(combined, 1.0)
        dropped = combined > drop_threshold
        indicator = (not dropped) and missing >= indicator_threshold
        candidates.append(
            CandidateFeature(
                name=name,
                is_categorical=name in categorical_features,
                train_missing_fraction=missing,
                train_nonfinite_fraction=nonfinite,
                dropped=dropped,
                missing_indicator=indicator,
            )
        )

    dropped_count = sum(1 for candidate in candidates if candidate.dropped)
    candidate_count = len(candidates)
    client_invalid = False
    reason: str | None = None
    if candidate_count == 0:
        client_invalid = True
        reason = "zero candidate features after mandatory semantic exclusions"
    elif dropped_count / candidate_count > invalid_fraction_threshold:
        client_invalid = True
        reason = "dropped-feature fraction exceeds the client-invalidity threshold"
    return FeatureQualityReport(
        candidate_features=tuple(candidates),
        dropped_feature_count=dropped_count,
        client_invalid=client_invalid,
        client_invalid_reason=reason,
    )


def categorical_vocabulary(train_categories: tuple[str, ...]) -> tuple[str, ...]:
    return (
        ABSENT_TOKEN,
        RARE_TOKEN,
        UNK_TOKEN,
        *tuple(sorted(train_categories, key=lambda token: token.encode("utf-8"))),
    )


def is_missing_token(token: str, categorical: bool) -> bool:
    lowered = token.strip().lower()
    if lowered in ("nan", "none", "null", ""):
        return True
    return bool(categorical and lowered in ("0", "0.0"))


def numeric_zero_is_not_missing(value: float) -> bool:
    return not math.isnan(value) and math.isclose(value, 0.0)
