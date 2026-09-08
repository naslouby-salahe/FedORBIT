from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from fedorbit.types import (
    Estimate,
    ArtifactIdentifier,
    ContrastName,
    DirectedPairName,
    EvaluationConditionName,
    FieldDescription,
    FineLabel,
    ExperimentName,
    Index,
    InvalidReason,
    MetricId,
    MetricUnit,
    MultiplicityFamily,
    RandomSeed,
    RelativeGain,
    ResampleCount,
    SampleCount,
    Fraction,
    Sha256Digest,
    SignificanceLevel,
    Split,
    StatisticalTestName,
    TransferMethod,
    is_sha256_digest,
)

type PredictionSemanticIdentity = tuple[
    ExperimentName,
    DirectedPairName,
    TransferMethod,
    EvaluationConditionName,
    RandomSeed,
    Split,
    Sha256Digest,
]
type MetricSemanticIdentity = tuple[
    ExperimentName,
    DirectedPairName,
    TransferMethod,
    EvaluationConditionName,
    RandomSeed,
    MetricId,
]
type EvaluationSemanticIdentity = PredictionSemanticIdentity | MetricSemanticIdentity


class MetricDirection(StrEnum):
    LOWER_IS_BETTER = "Lower is better"
    HIGHER_IS_BETTER = "Higher is better"
    DESCRIPTIVE = "Descriptive"


class StatisticalExactness(StrEnum):
    EXACT = "Exact"
    ASYMPTOTIC = "Asymptotic"
    BOOTSTRAP = "Bootstrap"


class StatisticalAlternative(StrEnum):
    TWO_SIDED = "Two-sided"
    GREATER = "Greater"
    LESS = "Less"
    EQUIVALENCE = "Equivalence"


class ComparisonDecision(StrEnum):
    SUPERIOR = "Superior"
    EQUIVALENT = "Equivalent"
    NOT_SUPPORTED = "Not Supported"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"
    DEGENERATE = "Degenerate"


class FrozenRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PredictionRecord(FrozenRecord):
    experiment: ExperimentName
    pair: DirectedPairName
    method: TransferMethod
    condition: EvaluationConditionName
    seed: RandomSeed
    row_hash: Sha256Digest
    split: Split
    true_local_class_id: FineLabel
    predicted_local_class_id: FineLabel
    probabilities: tuple[Fraction, ...]
    loss: Estimate
    checkpoint_artifact_id: ArtifactIdentifier
    processed_split_artifact_id: ArtifactIdentifier
    dependency_fingerprint_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_record(self) -> PredictionRecord:
        if not self.pair or not self.condition:
            raise ValueError("prediction pair and condition must be non-empty")
        _require_sha256(self.row_hash, FieldDescription("prediction row hash"))
        if not self.true_local_class_id or not self.predicted_local_class_id:
            raise ValueError("prediction class identities must be non-empty")
        if not self.checkpoint_artifact_id or not self.processed_split_artifact_id:
            raise ValueError("prediction artifact identities must be non-empty")
        if not self.probabilities:
            raise ValueError("prediction probability vector must be non-empty")
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0 for value in self.probabilities
        ):
            raise ValueError("prediction probabilities must be finite values in [0,1]")
        probability_sum = math.fsum(self.probabilities)
        absolute_tolerance = math.ulp(1.0) * max(1, len(self.probabilities))
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=absolute_tolerance):
            raise ValueError("prediction probabilities must sum to one")
        if not math.isfinite(self.loss) or self.loss < 0.0:
            raise ValueError("prediction loss must be finite and nonnegative")
        _require_sha256(
            self.dependency_fingerprint_sha256,
            FieldDescription("prediction dependency fingerprint"),
        )
        return self


class MetricRecord(FrozenRecord):
    experiment: ExperimentName
    pair: DirectedPairName
    method: TransferMethod
    condition: EvaluationConditionName
    seed: RandomSeed
    metric_name: MetricId
    metric_value: Estimate | None
    metric_unit: MetricUnit
    direction: MetricDirection
    evaluation_class_set_sha256: Sha256Digest
    input_artifact_ids: tuple[ArtifactIdentifier, ...]
    dependency_fingerprint_sha256: Sha256Digest
    valid: bool
    invalid_reason: InvalidReason | None

    @model_validator(mode="after")
    def validate_record(self) -> MetricRecord:
        if not self.pair or not self.condition or not self.metric_unit:
            raise ValueError("metric identity/unit fields must be non-empty")
        _require_sha256(
            self.evaluation_class_set_sha256,
            FieldDescription("evaluation class-set SHA-256"),
        )
        _require_sha256(
            self.dependency_fingerprint_sha256,
            FieldDescription("metric dependency fingerprint"),
        )
        if self.valid:
            if self.metric_value is None or not math.isfinite(self.metric_value):
                raise ValueError("valid metric requires a finite metric value")
            if self.invalid_reason is not None:
                raise ValueError("valid metric must not have an invalid reason")
        elif not self.invalid_reason:
            raise ValueError("invalid metric requires an invalid reason")
        elif self.metric_value is not None and not math.isfinite(self.metric_value):
            raise ValueError("invalid metric value must be finite when present")
        if not self.input_artifact_ids or any(not value for value in self.input_artifact_ids):
            raise ValueError("metric requires non-empty input artifact identities")
        return self


class PairedComparisonRecord(FrozenRecord):
    contrast_name: ContrastName
    family: MultiplicityFamily
    pair: DirectedPairName
    method_a: TransferMethod
    method_b: TransferMethod
    metric: MetricId
    paired_seed_count: Index
    mean_difference: RelativeGain | None
    median_difference: RelativeGain | None
    bca_ci_low: RelativeGain | None
    bca_ci_high: RelativeGain | None
    raw_p: SignificanceLevel | None
    holm_p: SignificanceLevel | None
    materiality_threshold: RelativeGain | None
    equivalence_margin_low: RelativeGain | None
    equivalence_margin_high: RelativeGain | None
    input_metric_artifact_ids: tuple[ArtifactIdentifier, ...]
    dependency_fingerprint_sha256: Sha256Digest
    decision: ComparisonDecision

    @model_validator(mode="after")
    def validate_record(self) -> PairedComparisonRecord:
        if not self.contrast_name or not self.pair:
            raise ValueError("comparison identity fields must be non-empty")
        if self.method_a == self.method_b:
            raise ValueError("paired comparison methods must differ")
        if self.paired_seed_count < 0:
            raise ValueError("paired seed count must be nonnegative")
        for name, value in (
            ("mean_difference", self.mean_difference),
            ("median_difference", self.median_difference),
            ("bca_ci_low", self.bca_ci_low),
            ("bca_ci_high", self.bca_ci_high),
            ("materiality_threshold", self.materiality_threshold),
            ("equivalence_margin_low", self.equivalence_margin_low),
            ("equivalence_margin_high", self.equivalence_margin_high),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when present")
        for name, value in (("raw_p", self.raw_p), ("holm_p", self.holm_p)):
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0,1]")
        if (self.bca_ci_low is None) != (self.bca_ci_high is None):
            raise ValueError("BCa interval endpoints must both be present or absent")
        if (
            self.bca_ci_low is not None
            and self.bca_ci_high is not None
            and self.bca_ci_low > self.bca_ci_high
        ):
            raise ValueError("BCa interval endpoints are reversed")
        if (self.equivalence_margin_low is None) != (self.equivalence_margin_high is None):
            raise ValueError("equivalence margins must both be present or absent")
        if (
            self.equivalence_margin_low is not None
            and self.equivalence_margin_high is not None
            and self.equivalence_margin_low >= self.equivalence_margin_high
        ):
            raise ValueError("equivalence margins are not strictly ordered")
        if not self.input_metric_artifact_ids or any(
            not value for value in self.input_metric_artifact_ids
        ):
            raise ValueError("comparison requires non-empty input metric artifact identities")
        _require_sha256(
            self.dependency_fingerprint_sha256,
            FieldDescription("comparison dependency fingerprint"),
        )
        return self


class StatisticalMetadataRecord(FrozenRecord):
    test_name: StatisticalTestName
    exact_or_asymptotic: StatisticalExactness
    alternative: StatisticalAlternative
    zero_difference_count: Index
    bootstrap_resamples: ResampleCount
    bootstrap_seed: RandomSeed | None
    holm_rank: Index | None
    family_size: SampleCount
    statistical_code_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_record(self) -> StatisticalMetadataRecord:
        if not self.test_name:
            raise ValueError("statistical test name must be non-empty")
        if self.zero_difference_count < 0:
            raise ValueError("zero-difference count must be nonnegative")
        if self.bootstrap_resamples < 0:
            raise ValueError("bootstrap resample count must be nonnegative")
        if self.bootstrap_seed is not None and self.bootstrap_seed < 0:
            raise ValueError("bootstrap seed must be nonnegative")
        if self.family_size <= 0:
            raise ValueError("multiplicity family size must be positive")
        if self.holm_rank is not None and not 1 <= self.holm_rank <= self.family_size:
            raise ValueError("Holm rank must lie within the multiplicity family")
        _require_sha256(
            self.statistical_code_sha256,
            FieldDescription("statistical code SHA-256"),
        )
        return self


def _require_sha256(
    value: Sha256Digest,
    field_name: FieldDescription,
) -> None:
    if not is_sha256_digest(value):
        raise ValueError(f"{field_name} must be lowercase SHA-256 hex")


class EvaluationValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PredictionRecordCollection:
    records: tuple[PredictionRecord, ...]


@dataclass(frozen=True, slots=True)
class MetricRecordCollection:
    records: tuple[MetricRecord, ...]


def validate_prediction_records(
    records: PredictionRecordCollection,
) -> PredictionRecordCollection:
    materialized = records.records
    _require_unique_semantic_identities(
        tuple(
            (
                record.experiment,
                record.pair,
                record.method,
                record.condition,
                record.seed,
                record.split,
                record.row_hash,
            )
            for record in materialized
        ),
        "duplicate prediction semantic identity",
    )
    return PredictionRecordCollection(materialized)


def validate_metric_records(records: MetricRecordCollection) -> MetricRecordCollection:
    materialized = records.records
    _require_unique_semantic_identities(
        tuple(
            (
                record.experiment,
                record.pair,
                record.method,
                record.condition,
                record.seed,
                record.metric_name,
            )
            for record in materialized
        ),
        "duplicate metric semantic identity",
    )
    return MetricRecordCollection(materialized)


def _require_unique_semantic_identities(
    identities: tuple[EvaluationSemanticIdentity, ...],
    error_message: str,
) -> None:
    if len(set(identities)) != len(identities):
        raise EvaluationValidationError(error_message)


def validate_comparison_metadata(
    comparison: PairedComparisonRecord,
    metadata: StatisticalMetadataRecord,
) -> None:
    if comparison.holm_p is not None and metadata.holm_rank is None:
        raise EvaluationValidationError("Holm-adjusted comparison requires Holm rank metadata")
    if comparison.holm_p is None and metadata.holm_rank is not None:
        raise EvaluationValidationError(
            "Holm rank metadata requires an adjusted comparison p-value"
        )
    if metadata.bootstrap_resamples > 0 and metadata.bootstrap_seed is None:
        raise EvaluationValidationError("bootstrap procedure requires a derived bootstrap seed")
    if metadata.bootstrap_resamples == 0 and metadata.bootstrap_seed is not None:
        raise EvaluationValidationError("bootstrap seed is invalid when no bootstrap was performed")
    if comparison.raw_p is None and comparison.holm_p is not None:
        raise EvaluationValidationError("Holm adjustment requires a raw p-value")
