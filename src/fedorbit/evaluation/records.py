from __future__ import annotations

import math
import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from fedorbit.domain.enums import (
    ExperimentName,
    MetricId,
    MultiplicityFamily,
    Split,
    TransferMethod,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
    pair: str
    method: TransferMethod
    condition: str
    seed: int
    row_hash: str
    split: Split
    true_local_class_id: str
    predicted_local_class_id: str
    probabilities: tuple[float, ...]
    loss: float
    checkpoint_artifact_id: str
    processed_split_artifact_id: str
    dependency_fingerprint_sha256: str

    @model_validator(mode="after")
    def validate_record(self) -> PredictionRecord:
        if not self.pair or not self.condition or not self.row_hash:
            raise ValueError("prediction identity fields must be non-empty")
        if not self.probabilities:
            raise ValueError("prediction probability vector must be non-empty")
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0 for value in self.probabilities
        ):
            raise ValueError("prediction probabilities must be finite values in [0,1]")
        if not math.isclose(sum(self.probabilities), 1.0):
            raise ValueError("prediction probabilities must sum to one")
        if not math.isfinite(self.loss) or self.loss < 0.0:
            raise ValueError("prediction loss must be finite and nonnegative")
        _require_sha256(self.dependency_fingerprint_sha256, "prediction dependency fingerprint")
        return self


class MetricRecord(FrozenRecord):
    experiment: ExperimentName
    pair: str
    method: TransferMethod
    condition: str
    seed: int
    metric_name: MetricId
    metric_value: float | None
    metric_unit: str
    direction: MetricDirection
    evaluation_class_set_sha256: str
    input_artifact_ids: tuple[str, ...]
    dependency_fingerprint_sha256: str
    valid: bool
    invalid_reason: str | None

    @model_validator(mode="after")
    def validate_record(self) -> MetricRecord:
        _require_sha256(self.evaluation_class_set_sha256, "evaluation class-set SHA-256")
        _require_sha256(self.dependency_fingerprint_sha256, "metric dependency fingerprint")
        if self.valid:
            if self.metric_value is None or not math.isfinite(self.metric_value):
                raise ValueError("valid metric requires a finite metric value")
            if self.invalid_reason is not None:
                raise ValueError("valid metric must not have an invalid reason")
        elif not self.invalid_reason:
            raise ValueError("invalid metric requires an invalid reason")
        if not self.input_artifact_ids:
            raise ValueError("metric requires at least one input artifact")
        return self


class PairedComparisonRecord(FrozenRecord):
    contrast_name: str
    family: MultiplicityFamily
    pair: str
    method_a: TransferMethod
    method_b: TransferMethod
    metric: MetricId
    paired_seed_count: int
    mean_difference: float | None
    median_difference: float | None
    bca_ci_low: float | None
    bca_ci_high: float | None
    raw_p: float | None
    holm_p: float | None
    materiality_threshold: float | None
    equivalence_margin_low: float | None
    equivalence_margin_high: float | None
    input_metric_artifact_ids: tuple[str, ...]
    dependency_fingerprint_sha256: str
    decision: ComparisonDecision

    @model_validator(mode="after")
    def validate_record(self) -> PairedComparisonRecord:
        if self.paired_seed_count < 0:
            raise ValueError("paired seed count must be nonnegative")
        for name, value in (("raw_p", self.raw_p), ("holm_p", self.holm_p)):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if (
            self.bca_ci_low is not None
            and self.bca_ci_high is not None
            and self.bca_ci_low > self.bca_ci_high
        ):
            raise ValueError("BCa interval endpoints are reversed")
        _require_sha256(self.dependency_fingerprint_sha256, "comparison dependency fingerprint")
        return self


class StatisticalMetadataRecord(FrozenRecord):
    test_name: str
    exact_or_asymptotic: StatisticalExactness
    alternative: StatisticalAlternative
    zero_difference_count: int
    bootstrap_resamples: int
    bootstrap_seed: int | None
    holm_rank: int | None
    family_size: int
    statistical_code_sha256: str

    @model_validator(mode="after")
    def validate_record(self) -> StatisticalMetadataRecord:
        if self.zero_difference_count < 0:
            raise ValueError("zero-difference count must be nonnegative")
        if self.bootstrap_resamples < 0:
            raise ValueError("bootstrap resample count must be nonnegative")
        if self.family_size <= 0:
            raise ValueError("multiplicity family size must be positive")
        if self.holm_rank is not None and not 1 <= self.holm_rank <= self.family_size:
            raise ValueError("Holm rank must lie within the multiplicity family")
        _require_sha256(self.statistical_code_sha256, "statistical code SHA-256")
        return self


def _require_sha256(value: str, field_name: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be lowercase SHA-256 hex")
