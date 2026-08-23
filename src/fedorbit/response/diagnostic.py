from __future__ import annotations

import math
import statistics

import torch

from fedorbit.config.models import FedorbitConfig
from fedorbit.models.training import BaseCheckpoint
from fedorbit.response.bootstrap import max_t_critical_value
from fedorbit.response.final import FinalResponseEntry, FinalResponseEstimate
from fedorbit.response.pilot import PilotData
from fedorbit.response.shadows import (
    ShadowData,
    ShadowSettings,
    paired_shadow_derivative,
    run_shadow_pair,
)


class DiagnosticError(ValueError):
    pass


def estimate_target_response_diagnostic(
    config: FedorbitConfig,
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: tuple[int, ...],
    seed: int,
) -> FinalResponseEstimate:
    diagnostic = config.scientific.target_response_diagnostic
    final = config.scientific.source_response_final
    outcome_count = len(data.outcome_native_class_sets)
    accumulated: list[list[float]] = [[] for _ in range(outcome_count)]
    all_finite = True
    settings = ShadowSettings(
        diagnostic.intervention_magnitude,
        diagnostic.shadow_optimizer_steps,
        data.learning_rate,
        data.weight_decay,
    )
    shadow_data = ShadowData(
        data.train_features,
        data.train_targets,
        data.meta_features,
        data.meta_targets,
        intervention_classes,
        data.outcome_native_class_sets,
        data.base_class_weights,
    )
    for replicate in range(diagnostic.paired_replicates):
        risks = run_shadow_pair(
            config,
            model,
            checkpoint.state_dict,
            checkpoint.optimizer_state,
            checkpoint.rng_state,
            shadow_data,
            settings,
            seed + replicate,
        )
        for outcome_index in range(outcome_count):
            positive, negative, baseline = risks[outcome_index]
            derivative = paired_shadow_derivative(
                positive,
                negative,
                baseline,
                diagnostic.intervention_magnitude,
                final.response_risk_denominator_floor,
            )
            if not all(
                math.isfinite(value) for value in (positive, negative, baseline, derivative)
            ):
                all_finite = False
            accumulated[outcome_index].append(derivative)
    if not all_finite:
        raise DiagnosticError("non-finite shadow state or loss in target-local diagnostic")
    entry_derivatives = tuple(tuple(values) for values in accumulated)
    means = tuple(statistics.fmean(values) for values in entry_derivatives)
    standard_errors = tuple(_standard_error(values) for values in entry_derivatives)
    critical = max_t_critical_value(
        config,
        entry_derivatives,
        seed,
        resamples=diagnostic.simultaneous_bootstrap_resamples,
        confidence_level=diagnostic.confidence_level,
        standard_error_floor=final.response_standard_error_floor,
    )
    entries = tuple(
        FinalResponseEntry(
            outcome_index,
            0,
            means[outcome_index],
            standard_errors[outcome_index],
            means[outcome_index] - critical * standard_errors[outcome_index],
            means[outcome_index] + critical * standard_errors[outcome_index],
            abs(means[outcome_index]) >= final.useful_response_magnitude_threshold,
        )
        for outcome_index in range(outcome_count)
    )
    return FinalResponseEstimate(
        entries,
        critical,
        len(entries),
        math.nan,
        True,
    )


def _standard_error(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return math.nan
    return statistics.stdev(values) / math.sqrt(len(values))
