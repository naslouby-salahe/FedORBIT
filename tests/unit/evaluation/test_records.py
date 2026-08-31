from __future__ import annotations

import pytest
from pydantic import ValidationError

from fedorbit.domain.enums import (
    ExperimentName,
    MetricId,
    MultiplicityFamily,
    Split,
    TransferMethod,
)
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
    MetricRecordCollection,
    PredictionRecordCollection,
    validate_comparison_metadata,
    validate_metric_records,
    validate_prediction_records,
)

SHA = "a" * 64
ROW_SHA = "b" * 64


def _prediction(row_hash: str = ROW_SHA) -> PredictionRecord:
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


def test_prediction_schema_has_exact_registered_fields() -> None:
    assert tuple(PredictionRecord.model_fields) == (
        "experiment",
        "pair",
        "method",
        "condition",
        "seed",
        "row_hash",
        "split",
        "true_local_class_id",
        "predicted_local_class_id",
        "probabilities",
        "loss",
        "checkpoint_artifact_id",
        "processed_split_artifact_id",
        "dependency_fingerprint_sha256",
    )


def test_prediction_record_rejects_non_probability_vector() -> None:
    invalid = _prediction().model_copy(update={"probabilities": (0.8, 0.8)})
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(invalid.model_dump())


def test_prediction_record_requires_sha256_row_identity() -> None:
    with pytest.raises(ValidationError):
        _prediction("row-1")


def test_prediction_semantic_identity_is_unique_per_condition_split_and_row() -> None:
    first = _prediction()
    second = first.model_copy(update={"condition": "alternate"})
    third = first.model_copy(update={"split": Split.VALID})
    assert validate_prediction_records(
        PredictionRecordCollection((first, second, third))
    ) == PredictionRecordCollection((first, second, third))
    with pytest.raises(EvaluationValidationError):
        validate_prediction_records(PredictionRecordCollection((first, first)))


def _metric(metric_value: float | None, valid: bool, invalid_reason: str | None) -> MetricRecord:
    return MetricRecord(
        experiment=ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        pair="source -> target",
        method=TransferMethod.LOCAL_ONLY,
        condition="principal",
        seed=1103,
        metric_name=MetricId.MACRO_CROSS_ENTROPY,
        metric_value=metric_value,
        metric_unit="cross-entropy",
        direction=MetricDirection.LOWER_IS_BETTER,
        evaluation_class_set_sha256=SHA,
        input_artifact_ids=("prediction-1",),
        dependency_fingerprint_sha256=SHA,
        valid=valid,
        invalid_reason=invalid_reason,
    )


def test_metric_schema_has_exact_registered_fields() -> None:
    assert tuple(MetricRecord.model_fields) == (
        "experiment",
        "pair",
        "method",
        "condition",
        "seed",
        "metric_name",
        "metric_value",
        "metric_unit",
        "direction",
        "evaluation_class_set_sha256",
        "input_artifact_ids",
        "dependency_fingerprint_sha256",
        "valid",
        "invalid_reason",
    )


def test_metric_validity_contract() -> None:
    metric = _metric(0.4, True, None)
    assert validate_metric_records(MetricRecordCollection((metric,))) == MetricRecordCollection(
        (metric,)
    )
    with pytest.raises(ValidationError):
        _metric(None, True, None)


def _comparison() -> PairedComparisonRecord:
    return PairedComparisonRecord(
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


def test_paired_comparison_schema_has_exact_registered_fields() -> None:
    assert tuple(PairedComparisonRecord.model_fields) == (
        "contrast_name",
        "family",
        "pair",
        "method_a",
        "method_b",
        "metric",
        "paired_seed_count",
        "mean_difference",
        "median_difference",
        "bca_ci_low",
        "bca_ci_high",
        "raw_p",
        "holm_p",
        "materiality_threshold",
        "equivalence_margin_low",
        "equivalence_margin_high",
        "input_metric_artifact_ids",
        "dependency_fingerprint_sha256",
        "decision",
    )


def _metadata() -> StatisticalMetadataRecord:
    return StatisticalMetadataRecord(
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


def test_statistical_metadata_schema_has_exact_registered_fields() -> None:
    assert tuple(StatisticalMetadataRecord.model_fields) == (
        "test_name",
        "exact_or_asymptotic",
        "alternative",
        "zero_difference_count",
        "bootstrap_resamples",
        "bootstrap_seed",
        "holm_rank",
        "family_size",
        "statistical_code_sha256",
    )


def test_comparison_and_statistical_metadata_are_jointly_validated() -> None:
    validate_comparison_metadata(_comparison(), _metadata())


def test_comparison_metadata_rejects_missing_holm_rank() -> None:
    with pytest.raises(EvaluationValidationError, match="Holm rank"):
        validate_comparison_metadata(
            _comparison(), _metadata().model_copy(update={"holm_rank": None})
        )


def test_comparison_rejects_partial_bca_interval() -> None:
    with pytest.raises(ValidationError):
        PairedComparisonRecord.model_validate(
            _comparison().model_copy(update={"bca_ci_high": None}).model_dump()
        )
