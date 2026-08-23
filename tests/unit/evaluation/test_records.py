from __future__ import annotations

import pytest
from pydantic import ValidationError

from fedorbit.domain.enums import ExperimentName, MetricId, MultiplicityFamily, Split, TransferMethod
from fedorbit.evaluation.records import (
    ComparisonDecision,
    MetricDirection,
    MetricRecord,
    PairedComparisonRecord,
    PredictionRecord,
    StatisticalAlternative,
    StatisticalExactness,
    StatisticalMetadataRecord,
)
from fedorbit.evaluation.validation import (
    EvaluationValidationError,
    validate_comparison_metadata,
    validate_metric_records,
    validate_prediction_records,
)

SHA = "a" * 64


def _prediction(row_hash: str = "row-1") -> PredictionRecord:
    return PredictionRecord(
        experiment=ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        pair="source -> target",
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        condition="principal",
        seed=1103,
        row_hash=row_hash,
        split=Split.TEST,
        true_local_class_id="ddos",
        predicted_local_class_id="ddos",
        probabilities=(0.8, 0.2),
        loss=0.2,
        checkpoint_artifact_id="checkpoint-1",
        processed_split_artifact_id="split-1",
        dependency_fingerprint_sha256=SHA,
    )


def test_prediction_record_rejects_non_probability_vector() -> None:
    with pytest.raises(ValidationError):
        _prediction().model_copy(update={"probabilities": (0.8, 0.8)}).model_validate(
            _prediction().model_copy(update={"probabilities": (0.8, 0.8)}).model_dump()
        )


def test_prediction_semantic_identity_is_unique() -> None:
    first = _prediction()
    second = _prediction("row-2")
    assert validate_prediction_records((first, second)) == (first, second)
    with pytest.raises(EvaluationValidationError):
        validate_prediction_records((first, first))


def test_metric_validity_contract() -> None:
    metric = MetricRecord(
        experiment=ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        pair="source -> target",
        method=TransferMethod.LOCAL_ONLY,
        condition="principal",
        seed=1103,
        metric_name=MetricId.MACRO_CROSS_ENTROPY,
        metric_value=0.4,
        metric_unit="cross-entropy",
        direction=MetricDirection.LOWER_IS_BETTER,
        evaluation_class_set_sha256=SHA,
        input_artifact_ids=("prediction-1",),
        dependency_fingerprint_sha256=SHA,
        valid=True,
        invalid_reason=None,
    )
    assert validate_metric_records((metric,)) == (metric,)
    with pytest.raises(ValidationError):
        MetricRecord(**{**metric.model_dump(), "metric_value": None})


def test_comparison_and_statistical_metadata_are_jointly_validated() -> None:
    comparison = PairedComparisonRecord(
        contrast_name="principal vs local-only",
        family=MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
        pair="source -> target",
        method_a=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        method_b=TransferMethod.LOCAL_ONLY,
        metric=MetricId.RELATIVE_MACRO_CE_GAIN,
        paired_seed_count=10,
        mean_difference=0.03,
        median_difference=0.02,
        bca_ci_low=0.01,
        bca_ci_high=0.05,
        raw_p=0.01,
        holm_p=0.02,
        materiality_threshold=0.01,
        equivalence_margin_low=-0.01,
        equivalence_margin_high=0.01,
        input_metric_artifact_ids=("metric-a", "metric-b"),
        dependency_fingerprint_sha256=SHA,
        decision=ComparisonDecision.SUPERIOR,
    )
    metadata = StatisticalMetadataRecord(
        test_name="exact paired sign-flip",
        exact_or_asymptotic=StatisticalExactness.EXACT,
        alternative=StatisticalAlternative.TWO_SIDED,
        zero_difference_count=0,
        bootstrap_resamples=10000,
        bootstrap_seed=42,
        holm_rank=1,
        family_size=4,
        statistical_code_sha256=SHA,
    )
    validate_comparison_metadata(comparison, metadata)
