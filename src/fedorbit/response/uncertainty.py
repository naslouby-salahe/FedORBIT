from __future__ import annotations

import math
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.runtime import (
    RandomSeed,
    RngNamespace,
    SeedDerivationRequest,
    derive_seed32,
)
from fedorbit.learning.training import BaseCheckpoint
from fedorbit.response.estimation import (
    ShadowData,
    ShadowSettings,
    paired_shadow_derivative,
    run_shadow_pair,
    standard_error,
)
from fedorbit.response.pilot import PilotData
from fedorbit.types import (
    ClassIndex,
    Coefficient,
    ConceptCount,
    ConfidenceLevel,
    Estimate,
    Floor,
    Index,
    ReplicateCount,
    ResampleCount,
    ResponseSeedStage,
    StableJsonPayload,
    StandardError,
)

type NativeClassSet = tuple[ClassIndex, ...]
type NativeClassSets = tuple[NativeClassSet, ...]
type DerivativeSeries = tuple[Estimate, ...]
type EntryDerivatives = tuple[DerivativeSeries, ...]


class ResponseUncertaintyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FinalResponseEntry:
    outcome_index: Index
    intervention_index: Index
    a_hat: Estimate
    standard_error: StandardError
    lower: Estimate
    upper: Estimate
    useful: bool


@dataclass(frozen=True, slots=True)
class FinalResponseEstimate:
    entries: tuple[FinalResponseEntry, ...]
    critical_value: Estimate
    useful_intervention_columns: ConceptCount
    median_band_width_ratio: Coefficient
    stability_rule_passed: bool


def max_t_critical_value(
    entry_derivatives: EntryDerivatives,
    seed: RandomSeed,
    resamples: ResampleCount | None = None,
    confidence_level: ConfidenceLevel | None = None,
    standard_error_floor: Floor | None = None,
) -> Estimate:
    final = active_config().scientific.source_response_final
    resample_count: ResampleCount = (
        resamples if resamples is not None else final.max_t_bootstrap_resamples
    )
    level: ConfidenceLevel = (
        confidence_level if confidence_level is not None else final.simultaneous_confidence_level
    )
    se_floor = (
        standard_error_floor
        if standard_error_floor is not None
        else final.response_standard_error_floor
    )
    if resample_count <= 0:
        raise ResponseUncertaintyError("bootstrap resample count must be positive")
    if not 0.0 < level < 1.0:
        raise ResponseUncertaintyError("bootstrap confidence level must be in (0, 1)")
    if se_floor <= 0.0:
        raise ResponseUncertaintyError("bootstrap standard-error floor must be positive")
    if not entry_derivatives:
        raise ResponseUncertaintyError("no response entries for bootstrap")
    replicate_count = len(entry_derivatives[0])
    if replicate_count < 2 or any(len(values) != replicate_count for values in entry_derivatives):
        raise ResponseUncertaintyError(
            "response bootstrap requires equal paired replicate counts of at least two"
        )
    means = tuple(statistics.fmean(values) for values in entry_derivatives)
    bootstrap_seed = derive_seed32(
        SeedDerivationRequest(
            seed,
            RngNamespace.RESPONSE_BOOTSTRAP,
            cast(
                StableJsonPayload,
                OrderedDict(
                    entries=len(entry_derivatives),
                    replicates=replicate_count,
                    resamples=resample_count,
                    confidence=level,
                ),
            ),
        )
    )
    rng = torch.Generator().manual_seed(bootstrap_seed)
    derivatives = np.asarray(entry_derivatives, dtype=np.float64)
    bootstrap_indices = torch.randint(
        0,
        replicate_count,
        (resample_count, replicate_count),
        generator=rng,
    ).numpy()
    resampled = derivatives[:, bootstrap_indices]
    bootstrap_means = np.mean(resampled, axis=2)
    bootstrap_standard_errors = np.std(resampled, axis=2, ddof=1) / math.sqrt(replicate_count)
    studentized = np.abs(bootstrap_means - np.asarray(means)[:, None]) / np.maximum(
        bootstrap_standard_errors,
        se_floor,
    )
    maxima = np.max(studentized, axis=0)
    quantile_probability = float(level)
    critical_value: Estimate = float(np.quantile(maxima, quantile_probability, method="higher"))
    return critical_value


def estimate_final_response(
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: NativeClassSets,
    settings: ShadowSettings,
    seed: RandomSeed,
) -> FinalResponseEstimate:
    final = active_config().scientific.source_response_final
    return estimate_response_bands(
        model,
        checkpoint,
        data,
        intervention_classes,
        settings,
        seed,
        replicate_count=final.paired_replicates_per_intervention,
        bootstrap_resamples=final.max_t_bootstrap_resamples,
        confidence_level=final.simultaneous_confidence_level,
        seed_stage=ResponseSeedStage("final-source-response"),
    )


def estimate_response_bands(
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: NativeClassSets,
    settings: ShadowSettings,
    seed: RandomSeed,
    *,
    replicate_count: ReplicateCount,
    bootstrap_resamples: ResampleCount,
    confidence_level: ConfidenceLevel,
    seed_stage: ResponseSeedStage,
) -> FinalResponseEstimate:
    final = active_config().scientific.source_response_final
    outcome_count = len(data.outcome_native_class_sets)
    intervention_count = len(intervention_classes)
    if outcome_count == 0 or intervention_count == 0:
        raise ResponseUncertaintyError("response matrix must have non-empty axes")
    if replicate_count < 2:
        raise ResponseUncertaintyError(
            "response estimation requires at least two paired replicates"
        )
    accumulated = np.empty(
        (outcome_count * intervention_count, replicate_count),
        dtype=np.float64,
    )
    for replicate in range(replicate_count):
        for intervention_index, concept_classes in enumerate(intervention_classes):
            shadow_data = ShadowData(
                data.train_features,
                data.train_targets,
                data.meta_features,
                data.meta_targets,
                concept_classes,
                data.outcome_native_class_sets,
                data.base_class_weights,
            )
            schedule_seed = derive_seed32(
                SeedDerivationRequest(
                    seed,
                    RngNamespace.RESPONSE_SCHEDULE,
                    cast(
                        StableJsonPayload,
                        OrderedDict(
                            stage=seed_stage,
                            replicate=replicate,
                            intervention=intervention_index,
                        ),
                    ),
                )
            )
            risks = run_shadow_pair(
                model,
                checkpoint.state_dict,
                checkpoint.optimizer_state,
                checkpoint.rng_state,
                shadow_data,
                settings,
                schedule_seed,
            )
            for outcome_index, (positive, negative, baseline) in enumerate(risks):
                derivative = paired_shadow_derivative(
                    positive,
                    negative,
                    baseline,
                    settings.epsilon,
                    final.response_risk_denominator_floor,
                )
                if not all(
                    math.isfinite(value) for value in (positive, negative, baseline, derivative)
                ):
                    raise ResponseUncertaintyError(
                        "non-finite shadow state or loss in response estimation"
                    )
                entry_index = outcome_index * intervention_count + intervention_index
                accumulated[entry_index, replicate] = derivative
    entry_derivatives = tuple(tuple(values) for values in accumulated)
    means = tuple(statistics.fmean(values) for values in entry_derivatives)
    standard_errors = tuple(standard_error(values) for values in entry_derivatives)
    critical = max_t_critical_value(
        entry_derivatives,
        seed,
        resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        standard_error_floor=final.response_standard_error_floor,
    )
    entries, useful_columns = _build_final_entries(
        outcome_count,
        intervention_count,
        means,
        standard_errors,
        critical,
    )
    useful_entries = tuple(entry for entry in entries if entry.useful)
    if not useful_entries:
        return FinalResponseEstimate(tuple(entries), critical, len(useful_columns), math.nan, False)
    band_widths = tuple(entry.upper - entry.lower for entry in useful_entries)
    absolute_means = tuple(abs(entry.a_hat) for entry in useful_entries)
    ratio = statistics.median(band_widths) / max(
        statistics.median(absolute_means),
        final.useful_response_magnitude_threshold,
    )
    stable = (
        len(useful_columns) >= final.minimum_useful_intervention_columns
        and ratio <= final.median_band_width_to_median_absolute_mean_response_maximum
    )
    return FinalResponseEstimate(tuple(entries), critical, len(useful_columns), ratio, stable)


def _build_final_entries(
    outcome_count: ConceptCount,
    intervention_count: ConceptCount,
    means: DerivativeSeries,
    standard_errors: tuple[StandardError, ...],
    critical: Estimate,
) -> tuple[list[FinalResponseEntry], set[Index]]:
    final = active_config().scientific.source_response_final
    entries: list[FinalResponseEntry] = []
    useful_columns: set[Index] = set()
    for outcome in range(outcome_count):
        for intervention in range(intervention_count):
            entry_index = outcome * intervention_count + intervention
            a_hat = means[entry_index]
            se = standard_errors[entry_index]
            lower = a_hat - critical * se
            upper = a_hat + critical * se
            useful = abs(a_hat) >= final.useful_response_magnitude_threshold and (
                lower > 0.0 or upper < 0.0
            )
            if useful:
                useful_columns.add(intervention)
            entries.append(
                FinalResponseEntry(
                    outcome,
                    intervention,
                    a_hat,
                    se,
                    lower,
                    upper,
                    useful,
                )
            )
    return entries, useful_columns
