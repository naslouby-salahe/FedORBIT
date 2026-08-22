from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import torch

from fedorbit.config.models import FedorbitConfig, SourceResponsePilotConfig
from fedorbit.models.training import BaseCheckpoint
from fedorbit.response.shadows import (
    ShadowData,
    ShadowSettings,
    paired_shadow_derivative,
    run_shadow_pair,
)


@dataclass(frozen=True, slots=True)
class ResponseCandidate:
    intervention_magnitude: float
    optimizer_step_horizon: int


@dataclass(frozen=True, slots=True)
class PilotData:
    train_features: torch.Tensor
    train_targets: torch.Tensor
    meta_features: torch.Tensor
    meta_targets: torch.Tensor
    outcome_native_class_sets: tuple[tuple[int, ...], ...]
    base_class_weights: torch.Tensor
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class DerivativeSeries:
    outcome_index: int
    intervention_index: int
    values: list[float]


@dataclass(frozen=True, slots=True)
class PilotEntry:
    outcome_index: int
    intervention_index: int
    a_hat_full: float
    se_full: float
    a_hat_half: float
    se_half: float
    derivative_discrepancy: float
    useful: bool
    sign_agreement: float


@dataclass(frozen=True, slots=True)
class CandidateResult:
    candidate: ResponseCandidate
    entries: tuple[PilotEntry, ...]
    eligible: bool
    ineligibility_reasons: tuple[str, ...]
    pilot_score: float


class ResponsePilotError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PairDerivatives:
    full_values: tuple[float, ...]
    half_values: tuple[float, ...]
    derivatives_finite: bool


def _pair_derivatives(
    config: FedorbitConfig,
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    shadow_data: ShadowData,
    full_settings: ShadowSettings,
    half_settings: ShadowSettings,
    pair_seed: int,
    outcome_count: int,
    pilot: SourceResponsePilotConfig,
) -> PairDerivatives:
    full_risks = run_shadow_pair(
        config,
        model,
        checkpoint.state_dict,
        checkpoint.optimizer_state,
        checkpoint.rng_state,
        shadow_data,
        full_settings,
        pair_seed,
    )
    half_risks = run_shadow_pair(
        config,
        model,
        checkpoint.state_dict,
        checkpoint.optimizer_state,
        checkpoint.rng_state,
        shadow_data,
        half_settings,
        pair_seed,
    )
    full_values: list[float] = []
    half_values: list[float] = []
    all_finite = True
    for outcome_index in range(outcome_count):
        full, half, finite = _outcome_derivatives(
            full_risks[outcome_index],
            half_risks[outcome_index],
            full_settings.epsilon,
            half_settings.epsilon,
            pilot.numerical_floor,
        )
        if not finite:
            all_finite = False
        full_values.append(full)
        half_values.append(half)
    return PairDerivatives(tuple(full_values), tuple(half_values), all_finite)


def _outcome_derivatives(
    full_risks: tuple[float, float, float],
    half_risks: tuple[float, float, float],
    full_epsilon: float,
    half_epsilon: float,
    numerical_floor: float,
) -> tuple[float, float, bool]:
    positive, negative, baseline = full_risks
    positive_half, negative_half, baseline_half = half_risks
    full = paired_shadow_derivative(positive, negative, baseline, full_epsilon, numerical_floor)
    half = paired_shadow_derivative(
        positive_half, negative_half, baseline_half, half_epsilon, numerical_floor
    )
    finite = all(
        math.isfinite(value)
        for value in (
            positive,
            negative,
            baseline,
            positive_half,
            negative_half,
            baseline_half,
            full,
            half,
        )
    )
    return full, half, finite


def run_source_response_pilot(
    config: FedorbitConfig,
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: tuple[tuple[int, ...], ...],
    seed: int,
) -> tuple[CandidateResult, ...]:
    pilot = config.scientific.source_response_pilot
    replicate_count = pilot.paired_schedules_per_candidate
    results: list[CandidateResult] = []
    for magnitude in pilot.intervention_magnitudes:
        for horizon in pilot.optimizer_step_horizons:
            candidate = ResponseCandidate(magnitude, horizon)
            results.append(
                _evaluate_candidate(
                    config,
                    model,
                    checkpoint,
                    data,
                    intervention_classes,
                    candidate,
                    replicate_count,
                    seed,
                )
            )
    return tuple(results)


def select_response_configuration(
    results: tuple[CandidateResult, ...],
) -> ResponseCandidate:
    eligible = tuple(result for result in results if result.eligible)
    if not eligible:
        raise ResponsePilotError("no eligible source-response pilot candidate")
    ordered = tuple(
        sorted(
            eligible,
            key=lambda result: (
                -result.pilot_score,
                result.candidate.optimizer_step_horizon,
                result.candidate.intervention_magnitude,
            ),
        )
    )
    return ordered[0].candidate


def _evaluate_candidate(
    config: FedorbitConfig,
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: tuple[tuple[int, ...], ...],
    candidate: ResponseCandidate,
    replicate_count: int,
    seed: int,
) -> CandidateResult:
    pilot = config.scientific.source_response_pilot
    outcome_count = len(data.outcome_native_class_sets)
    intervention_count = len(intervention_classes)
    full_series = [
        DerivativeSeries(outcome, intervention, [])
        for outcome in range(outcome_count)
        for intervention in range(intervention_count)
    ]
    half_series = [
        DerivativeSeries(outcome, intervention, [])
        for outcome in range(outcome_count)
        for intervention in range(intervention_count)
    ]
    all_finite = True
    for replicate in range(replicate_count):
        for intervention_index, concept_classes in enumerate(intervention_classes):
            pair_seed = seed + replicate * (intervention_count + 1) + intervention_index
            shadow_data = ShadowData(
                data.train_features,
                data.train_targets,
                data.meta_features,
                data.meta_targets,
                concept_classes,
                data.outcome_native_class_sets,
                data.base_class_weights,
            )
            full_settings = ShadowSettings(
                candidate.intervention_magnitude,
                candidate.optimizer_step_horizon,
                data.learning_rate,
                data.weight_decay,
            )
            half_settings = ShadowSettings(
                candidate.intervention_magnitude / 2,
                candidate.optimizer_step_horizon,
                data.learning_rate,
                data.weight_decay,
            )
            pair_derivatives = _pair_derivatives(
                config,
                model,
                checkpoint,
                shadow_data,
                full_settings,
                half_settings,
                pair_seed,
                outcome_count,
                pilot,
            )
            if not pair_derivatives.derivatives_finite:
                all_finite = False
            for outcome_index in range(outcome_count):
                entry_index = outcome_index * intervention_count + intervention_index
                full_series[entry_index].values.append(pair_derivatives.full_values[outcome_index])
                half_series[entry_index].values.append(pair_derivatives.half_values[outcome_index])
    entries: list[PilotEntry] = []
    useful_columns: set[int] = set()
    for outcome_index in range(outcome_count):
        for intervention_index in range(intervention_count):
            entry_index = outcome_index * intervention_count + intervention_index
            full_values = tuple(full_series[entry_index].values)
            half_values = tuple(half_series[entry_index].values)
            a_hat_full = statistics.fmean(full_values)
            se_full = standard_error(full_values)
            a_hat_half = statistics.fmean(half_values)
            se_half = standard_error(half_values)
            discrepancy = abs(a_hat_full - a_hat_half) / max(
                abs(a_hat_half), pilot.useful_response_magnitude_threshold
            )
            useful = (
                max(abs(a_hat_full), abs(a_hat_half)) >= pilot.useful_response_magnitude_threshold
            )
            if useful:
                useful_columns.add(intervention_index)
            sign_agreement_value = sign_agreement(full_values)
            entries.append(
                PilotEntry(
                    outcome_index,
                    intervention_index,
                    a_hat_full,
                    se_full,
                    a_hat_half,
                    se_half,
                    discrepancy,
                    useful,
                    sign_agreement_value,
                )
            )
    reasons: list[str] = []
    if not all_finite:
        reasons.append("non-finite shadow state or loss")
    useful_entries = tuple(entry for entry in entries if entry.useful)
    if not useful_entries:
        reasons.append("no useful entries")
    else:
        if (
            statistics.median(tuple(entry.derivative_discrepancy for entry in useful_entries))
            > pilot.relative_derivative_discrepancy_ceiling
        ):
            reasons.append("median derivative discrepancy above ceiling")
        if (
            statistics.median(tuple(entry.sign_agreement for entry in useful_entries))
            < pilot.sign_agreement_minimum
        ):
            reasons.append("median sign agreement below minimum")
        if len(useful_columns) < pilot.minimum_useful_intervention_columns:
            reasons.append("too few useful intervention columns")
    if useful_entries:
        score = statistics.median(
            tuple(
                abs(entry.a_hat_full) / (entry.se_full + pilot.numerical_floor)
                for entry in useful_entries
            )
        ) - pilot.curvature_penalty_coefficient * statistics.median(
            tuple(entry.derivative_discrepancy for entry in useful_entries)
        )
    else:
        score = math.nan
    return CandidateResult(
        candidate,
        tuple(entries),
        eligible=not reasons,
        ineligibility_reasons=tuple(reasons),
        pilot_score=score,
    )


def standard_error(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return math.nan
    return statistics.stdev(values) / math.sqrt(len(values))


def sign_agreement(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    positive = sum(1 for value in values if value > 0)
    negative = sum(1 for value in values if value < 0)
    return max(positive, negative) / len(values)
