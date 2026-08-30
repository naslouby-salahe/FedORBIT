from __future__ import annotations

import math
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from fedorbit.config.context import active_config
from fedorbit.config.models import SourceResponseFinalConfig
from fedorbit.domain.serialization import StableJsonPayload
from fedorbit.response.estimation import (
    ShadowData,
    ShadowSettings,
    paired_shadow_derivative,
    run_shadow_pair,
)
from fedorbit.response.pilot import PilotData
from fedorbit.runtime.seeds import RandomSeed, RngNamespace, SeedDerivationRequest, derive_seed32
from fedorbit.training.trainer import BaseCheckpoint


class ResponseUncertaintyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FinalResponseEntry:
    outcome_index: int
    intervention_index: int
    a_hat: float
    standard_error: float
    lower: float
    upper: float
    useful: bool


@dataclass(frozen=True, slots=True)
class FinalResponseEstimate:
    entries: tuple[FinalResponseEntry, ...]
    critical_value: float
    useful_intervention_columns: int
    median_band_width_ratio: float
    stability_rule_passed: bool


def max_t_critical_value(
    entry_derivatives: tuple[tuple[float, ...], ...],
    seed: int,
    resamples: int | None = None,
    confidence_level: float | None = None,
    standard_error_floor: float | None = None,
) -> float:
    final = active_config().scientific.source_response_final
    resample_count = resamples if resamples is not None else final.max_t_bootstrap_resamples
    level = (
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
            RandomSeed(seed),
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
    ).value
    rng = torch.Generator().manual_seed(bootstrap_seed)
    maxima: list[float] = []
    for _ in range(resample_count):
        indices = tuple(
            int(torch.randint(0, replicate_count, (1,), generator=rng)[0])
            for _ in range(replicate_count)
        )
        studentized: list[float] = []
        for entry_index, values in enumerate(entry_derivatives):
            resampled = tuple(values[index] for index in indices)
            bootstrap_mean = statistics.fmean(resampled)
            bootstrap_se = standard_error(resampled)
            studentized.append(
                abs(bootstrap_mean - means[entry_index]) / max(bootstrap_se, se_floor)
            )
        maxima.append(max(studentized))
    return float(
        np.quantile(
            np.asarray(maxima, dtype=np.float64),
            level,
            method="higher",
        )
    )


def estimate_final_response(
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: tuple[tuple[int, ...], ...],
    settings: ShadowSettings,
    seed: int,
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
        seed_stage="final-source-response",
    )


def estimate_response_bands(
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: tuple[tuple[int, ...], ...],
    settings: ShadowSettings,
    seed: int,
    *,
    replicate_count: int,
    bootstrap_resamples: int,
    confidence_level: float,
    seed_stage: str,
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
    accumulated: list[list[float]] = [[] for _ in range(outcome_count * intervention_count)]
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
                    RandomSeed(seed),
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
            ).value
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
                accumulated[entry_index].append(derivative)
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
        final,
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
    final: SourceResponseFinalConfig,
    outcome_count: int,
    intervention_count: int,
    means: tuple[float, ...],
    standard_errors: tuple[float, ...],
    critical: float,
) -> tuple[list[FinalResponseEntry], set[int]]:
    entries: list[FinalResponseEntry] = []
    useful_columns: set[int] = set()
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


def standard_error(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return math.nan
    return statistics.stdev(values) / math.sqrt(len(values))
