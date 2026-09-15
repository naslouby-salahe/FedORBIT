from __future__ import annotations

import numpy as np
import pytest

from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.common import (
    DatasetInspectionError,
    FieldRole,
    infer_feature_type,
    reconcile_component_columns,
)
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


def test_type_inference_requires_every_observed_value_to_be_losslessly_numeric() -> None:
    assert infer_feature_type(("1", "2.5", "nan", None)) is FieldRole.BEHAVIORAL_NUMERIC
    assert infer_feature_type(("1", "not-a-number")) is FieldRole.BEHAVIORAL_CATEGORICAL
    assert infer_feature_type(("", None)) is FieldRole.BEHAVIORAL_CATEGORICAL


def test_component_schema_reconciliation_preserves_reference_order_and_rejects_drift() -> None:
    timestamp = TabularColumnName("ts")
    binary = TabularColumnName("label")
    multiclass = TabularColumnName("type")
    alpha = TabularColumnName("alpha")
    beta = TabularColumnName("beta")
    gamma = TabularColumnName("gamma")

    assert reconcile_component_columns(
        ((timestamp, binary, multiclass, beta, alpha), (alpha, timestamp, beta, binary, multiclass))
    ) == (timestamp, binary, multiclass, beta, alpha)
    with pytest.raises(DatasetInspectionError, match="schema diverges"):
        reconcile_component_columns(
            ((timestamp, binary, multiclass, alpha), (timestamp, binary, multiclass, gamma))
        )


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


def test_feature_quality_and_indicator_thresholds_are_strict_and_train_scoped() -> None:
    threshold_feature = TabularColumnName("threshold_feature")
    dropped_feature = TabularColumnName("dropped_feature")
    retained_features = tuple(TabularColumnName(f"retained_{index}") for index in range(4))
    at_drop_threshold = np.asarray([np.nan, *range(19)], dtype=np.float64)
    above_drop_threshold = np.asarray([np.nan, np.nan, *range(18)], dtype=np.float64)
    retained = np.zeros(20, dtype=np.float64)
    report = evaluate_feature_quality(
        (threshold_feature, dropped_feature, *retained_features),
        frozenset(),
        TrainingFeatureValues(
            {
                threshold_feature: at_drop_threshold,
                dropped_feature: above_drop_threshold,
                **dict.fromkeys(retained_features, retained),
            }
        ),
    )
    threshold_candidate, dropped_candidate, *_ = report.candidate_features

    assert threshold_candidate.train_missing_fraction == 0.05
    assert not threshold_candidate.dropped
    assert threshold_candidate.missing_indicator
    assert dropped_candidate.train_missing_fraction == 0.1
    assert dropped_candidate.dropped
    assert not report.client_invalid

    invalid_report = evaluate_feature_quality(
        (threshold_feature, dropped_feature, *retained_features),
        frozenset(),
        TrainingFeatureValues(
            {
                threshold_feature: above_drop_threshold,
                dropped_feature: above_drop_threshold,
                **dict.fromkeys(retained_features, retained),
            }
        ),
    )

    assert invalid_report.dropped_feature_count == 2
    assert invalid_report.client_invalid

    indicator_feature = TabularColumnName("indicator_feature")
    indicator_report = evaluate_feature_quality(
        (indicator_feature,),
        frozenset(),
        TrainingFeatureValues(
            {indicator_feature: np.asarray([np.nan, *range(999)], dtype=np.float64)}
        ),
    )

    indicator_candidate = indicator_report.candidate_features[0]
    assert indicator_candidate.train_missing_fraction == 0.001
    assert not indicator_candidate.dropped
    assert indicator_candidate.missing_indicator


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


def test_categorical_rare_threshold_is_strict() -> None:
    boundary = fit_categorical_preprocessor(
        (*(RawCellText("common") for _ in range(999)), RawCellText("boundary"))
    )
    rare = fit_categorical_preprocessor(
        (*(RawCellText("common") for _ in range(1000)), RawCellText("rare"))
    )

    assert CategoryName("boundary") not in boundary.rare_categories
    assert CategoryName("boundary") in boundary.vocabulary
    assert CategoryName("rare") in rare.rare_categories
    assert transform_categorical(RawCellText("rare"), rare) == PreprocessingToken.RARE
