from __future__ import annotations

import numpy as np

from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.preprocessing import (
    ABSENT_TOKEN,
    MISSING_TOKEN_VOCABULARY,
    RARE_TOKEN,
    UNK_TOKEN,
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


def test_missing_token_contract_is_type_scoped() -> None:
    assert MISSING_TOKEN_VOCABULARY == frozenset({"", "0", "0.0", "nan", "none", "null"})
    assert is_missing_token("0", True)
    assert not is_missing_token("0", False)
    assert numeric_zero_is_not_missing(0.0)


def test_feature_quality_uses_raw_semantic_features_once(fedorbit_config: FedorbitConfig) -> None:
    report = evaluate_feature_quality(
        fedorbit_config,
        ("good", "bad", "category"),
        frozenset({"category"}),
        TrainingFeatureValues(
            {
                "good": np.array([0.0, 1.0, 2.0]),
                "bad": np.array([np.nan, np.nan, 1.0]),
                "category": np.array(["a", "b", "c"], dtype=object),
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
        fedorbit_config,
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


def test_categorical_vocabulary_and_mapping_are_deterministic(
    fedorbit_config: FedorbitConfig,
) -> None:
    vocabulary = categorical_vocabulary(("z", "a", "z"))
    assert vocabulary == (ABSENT_TOKEN, RARE_TOKEN, UNK_TOKEN, "a", "z")
    fitted = fit_categorical_preprocessor(fedorbit_config, ("a", "a", "b", ""))
    assert transform_categorical("", fitted) == ABSENT_TOKEN
    assert transform_categorical("never-seen", fitted) == UNK_TOKEN
    encoded = one_hot("a", fitted)
    assert len(encoded) == len(fitted.vocabulary)
    assert sum(encoded) == 1.0
