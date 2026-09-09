from __future__ import annotations

import numpy as np

from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.preprocessing import (
    MISSING_TOKEN_VOCABULARY,
    PreprocessingToken,
    TrainingFeatureValues,
    categorical_vocabulary,
    evaluate_feature_quality,
    fit_categorical_preprocessor,
    fit_numeric_preprocessor,
    is_missing_token,
    numeric_zero_is_not_missing,
    one_hot,
    transform_categorical,
    transform_numeric,
)
from fedorbit.types import (
    CategoryName,
    NumericFeatureValue,
    RawCellText,
    TabularColumnName,
)


def test_missing_token_contract_is_type_scoped() -> None:
    assert frozenset({"", "0", "0.0", "nan", "none", "null"}) == MISSING_TOKEN_VOCABULARY
    assert is_missing_token(RawCellText("0"), True)
    assert not is_missing_token(RawCellText("0"), False)
    assert numeric_zero_is_not_missing(NumericFeatureValue(0.0))


def test_feature_quality_uses_raw_semantic_features_once() -> None:
    good = TabularColumnName("good")
    bad = TabularColumnName("bad")
    category = TabularColumnName("category")
    report = evaluate_feature_quality(
        (good, bad, category),
        frozenset({category}),
        TrainingFeatureValues(
            {
                good: np.array([0.0, 1.0, 2.0]),
                bad: np.array([np.nan, np.nan, 1.0]),
                category: np.array(["a", "b", "c"], dtype=object),
            }
        ),
    )
    assert report.candidate_count_before_filtering == 3
    bad = next(item for item in report.candidate_features if item.name == "bad")
    assert bad.dropped


def test_numeric_preprocessor_uses_linear_quartiles_and_configured_clip(
    fedorbit_config: FedorbitConfig,
) -> None:
    fitted = fit_numeric_preprocessor(np.array([0.0, 1.0, 2.0, 3.0, np.nan]))
    assert fitted.median == 1.5
    assert fitted.iqr == 1.5
    transformed = transform_numeric(
        np.array([np.nan, -100.0, 100.0]),
        fitted,
    )
    clip = fedorbit_config.scientific.preprocessing.numeric_clip
    assert transformed[0] == 0.0
    assert transformed[1] == clip.lower
    assert transformed[2] == clip.upper


def test_zero_iqr_constant_feature_is_identified() -> None:
    fitted = fit_numeric_preprocessor(np.array([3.0, 3.0, np.nan]))
    assert fitted.iqr == 0.0
    assert fitted.scale == 1.0
    assert fitted.constant_after_imputation


def test_categorical_vocabulary_and_mapping_are_deterministic() -> None:
    vocabulary = categorical_vocabulary((CategoryName("z"), CategoryName("a"), CategoryName("z")))
    assert vocabulary == (
        PreprocessingToken.ABSENT,
        PreprocessingToken.RARE,
        PreprocessingToken.UNKNOWN,
        "a",
        "z",
    )
    fitted = fit_categorical_preprocessor(
        (RawCellText("a"), RawCellText("a"), RawCellText("b"), RawCellText(""))
    )
    assert transform_categorical(RawCellText(""), fitted) == PreprocessingToken.ABSENT
    assert transform_categorical(RawCellText("never-seen"), fitted) == PreprocessingToken.UNKNOWN
    encoded = one_hot(RawCellText("a"), fitted)
    assert len(encoded) == len(fitted.vocabulary)
    assert sum(encoded) == 1.0
