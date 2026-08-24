from __future__ import annotations

from collections.abc import Iterable

from fedorbit.evaluation.records import (
    MetricRecord,
    PairedComparisonRecord,
    PredictionRecord,
    StatisticalMetadataRecord,
)


class EvaluationValidationError(ValueError):
    pass


def validate_prediction_records(
    records: Iterable[PredictionRecord],
) -> tuple[PredictionRecord, ...]:
    materialized = tuple(records)
    identities: set[tuple[str, str, str, str, int, str, str]] = set()
    for record in materialized:
        identity = (
            record.experiment.value,
            record.pair,
            record.method.value,
            record.condition,
            record.seed,
            record.split.value,
            record.row_hash,
        )
        if identity in identities:
            raise EvaluationValidationError("duplicate prediction semantic identity")
        identities.add(identity)
    return materialized


def validate_metric_records(records: Iterable[MetricRecord]) -> tuple[MetricRecord, ...]:
    materialized = tuple(records)
    identities: set[tuple[str, str, str, str, int, str]] = set()
    for record in materialized:
        identity = (
            record.experiment.value,
            record.pair,
            record.method.value,
            record.condition,
            record.seed,
            record.metric_name.value,
        )
        if identity in identities:
            raise EvaluationValidationError("duplicate metric semantic identity")
        identities.add(identity)
    return materialized


def validate_comparison_metadata(
    comparison: PairedComparisonRecord,
    metadata: StatisticalMetadataRecord,
) -> None:
    if comparison.holm_p is not None and metadata.holm_rank is None:
        raise EvaluationValidationError("Holm-adjusted comparison requires Holm rank metadata")
    if comparison.holm_p is None and metadata.holm_rank is not None:
        raise EvaluationValidationError("Holm rank metadata requires an adjusted comparison p-value")
    if metadata.bootstrap_resamples > 0 and metadata.bootstrap_seed is None:
        raise EvaluationValidationError("bootstrap procedure requires a derived bootstrap seed")
    if metadata.bootstrap_resamples == 0 and metadata.bootstrap_seed is not None:
        raise EvaluationValidationError("bootstrap seed is invalid when no bootstrap was performed")
    if comparison.raw_p is None and comparison.holm_p is not None:
        raise EvaluationValidationError("Holm adjustment requires a raw p-value")
