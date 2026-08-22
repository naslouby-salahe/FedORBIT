from __future__ import annotations

import math
import statistics

import torch

from fedorbit.config.models import FedorbitConfig
from fedorbit.models.training import BaseCheckpoint
from fedorbit.response.bootstrap import max_t_critical_value
from fedorbit.response.final import FinalResponseEntry, FinalResponseEstimate
from fedorbit.response.pilot import DerivativeSeries
from fedorbit.response.shadows import paired_shadow_derivative, run_shadow_pair


class DiagnosticError(ValueError):
    pass


def estimate_target_response_diagnostic(
    config: FedorbitConfig,
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    meta_features: torch.Tensor,
    meta_targets: torch.Tensor,
    intervention_classes: tuple[int, ...],
    outcome_native_class_sets: tuple[tuple[int, ...], ...],
    base_class_weights: torch.Tensor,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> FinalResponseEstimate:
    diagnostic = config.scientific.target_response_diagnostic
    final = config.scientific.source_response_final
    outcome_count = len(outcome_native_class_sets)
    series = [DerivativeSeries(outcome, 0, []) for outcome in range(outcome_count)]
    all_finite = True
    for replicate in range(diagnostic.paired_replicates):
        risks = run_shadow_pair(
            config,
            model,
            checkpoint.state_dict,
            checkpoint.optimizer_state,
            checkpoint.rng_state,
            train_features,
            train_targets,
            meta_features,
            meta_targets,
            intervention_classes,
            outcome_native_class_sets,
            base_class_weights,
            diagnostic.intervention_magnitude,
            diagnostic.shadow_optimizer_steps,
            learning_rate,
            weight_decay,
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
            series[outcome_index].values.append(derivative)
    if not all_finite:
        raise DiagnosticError("non-finite shadow state or loss in target-local diagnostic")
    entry_derivatives = tuple(tuple(entry.values) for entry in series)
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
