from __future__ import annotations

import math
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from typing import cast

import torch

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.runtime import (
    RandomSeed,
    RngNamespace,
    SeedDerivationRequest,
    derive_seed32,
)
from fedorbit.learning.training import BaseCheckpoint, ClassWeights
from fedorbit.response.estimation import (
    ShadowData,
    ShadowSettings,
    paired_shadow_derivative,
    run_shadow_pair,
)
from fedorbit.types import StableJsonPayload


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
    base_class_weights: ClassWeights
    learning_rate: float
    weight_decay: float


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


def run_source_response_pilot(
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: tuple[tuple[int, ...], ...],
    seed: int,
) -> tuple[CandidateResult, ...]:
    pilot = active_config().scientific.source_response_pilot
    return tuple(
        _evaluate_candidate(
            model,
            checkpoint,
            data,
            intervention_classes,
            ResponseCandidate(magnitude, horizon),
            pilot.paired_schedules_per_candidate,
            seed,
        )
        for magnitude in pilot.intervention_magnitudes
        for horizon in pilot.optimizer_step_horizons
    )


def select_response_configuration(results: tuple[CandidateResult, ...]) -> ResponseCandidate:
    eligible = tuple(result for result in results if result.eligible)
    if not eligible:
        raise ResponsePilotError("no eligible source-response pilot candidate")
    return min(
        eligible,
        key=lambda result: (
            -result.pilot_score,
            result.candidate.optimizer_step_horizon,
            result.candidate.intervention_magnitude,
        ),
    ).candidate


def _evaluate_candidate(
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: tuple[tuple[int, ...], ...],
    candidate: ResponseCandidate,
    replicate_count: int,
    seed: int,
) -> CandidateResult:
    outcome_count = len(data.outcome_native_class_sets)
    intervention_count = len(intervention_classes)
    full_values: list[list[float]] = [[] for _ in range(outcome_count * intervention_count)]
    half_values: list[list[float]] = [[] for _ in range(outcome_count * intervention_count)]
    all_finite = True
    full_settings = ShadowSettings(
        candidate.intervention_magnitude,
        candidate.optimizer_step_horizon,
        data.learning_rate,
        data.weight_decay,
    )
    half_settings = ShadowSettings(
        candidate.intervention_magnitude / 2.0,
        candidate.optimizer_step_horizon,
        data.learning_rate,
        data.weight_decay,
    )
    for replicate in range(replicate_count):
        for intervention_index, concept_classes in enumerate(intervention_classes):
            schedule_seed = derive_seed32(
                SeedDerivationRequest(
                    RandomSeed(seed),
                    RngNamespace.RESPONSE_SCHEDULE,
                    cast(
                        StableJsonPayload,
                        OrderedDict(
                            stage="pilot",
                            magnitude=candidate.intervention_magnitude,
                            horizon=candidate.optimizer_step_horizon,
                            replicate=replicate,
                            intervention=intervention_index,
                        ),
                    ),
                )
            ).value
            shadow_data = ShadowData(
                data.train_features,
                data.train_targets,
                data.meta_features,
                data.meta_targets,
                concept_classes,
                data.outcome_native_class_sets,
                data.base_class_weights,
            )
            full_risks = run_shadow_pair(
                model,
                checkpoint.state_dict,
                checkpoint.optimizer_state,
                checkpoint.rng_state,
                shadow_data,
                full_settings,
                schedule_seed,
            )
            half_risks = run_shadow_pair(
                model,
                checkpoint.state_dict,
                checkpoint.optimizer_state,
                checkpoint.rng_state,
                shadow_data,
                half_settings,
                schedule_seed,
            )
            for outcome_index in range(outcome_count):
                full = _derivative(full_risks[outcome_index], full_settings.epsilon)
                half = _derivative(half_risks[outcome_index], half_settings.epsilon)
                if not math.isfinite(full) or not math.isfinite(half):
                    all_finite = False
                entry_index = outcome_index * intervention_count + intervention_index
                full_values[entry_index].append(full)
                half_values[entry_index].append(half)
    entries: list[PilotEntry] = []
    useful_columns: set[int] = set()
    for outcome_index in range(outcome_count):
        for intervention_index in range(intervention_count):
            entry_index = outcome_index * intervention_count + intervention_index
            entry = _build_pilot_entry(
                outcome_index,
                intervention_index,
                tuple(full_values[entry_index]),
                tuple(half_values[entry_index]),
            )
            if entry.useful:
                useful_columns.add(intervention_index)
            entries.append(entry)
    reasons = _eligibility_reasons(all_finite, entries, useful_columns)
    useful_entries = tuple(entry for entry in entries if entry.useful)
    score = _pilot_score(useful_entries)
    return CandidateResult(
        candidate,
        tuple(entries),
        eligible=not reasons,
        ineligibility_reasons=tuple(reasons),
        pilot_score=score,
    )


def _derivative(
    risks: tuple[float, float, float],
    epsilon: float,
) -> float:
    positive, negative, baseline = risks
    if not all(math.isfinite(value) for value in risks):
        return math.nan
    return paired_shadow_derivative(
        positive,
        negative,
        baseline,
        epsilon,
        active_config().scientific.source_response_pilot.numerical_floor,
    )


def _build_pilot_entry(
    outcome_index: int,
    intervention_index: int,
    full_values: tuple[float, ...],
    half_values: tuple[float, ...],
) -> PilotEntry:
    pilot = active_config().scientific.source_response_pilot
    a_hat_full = statistics.fmean(full_values)
    a_hat_half = statistics.fmean(half_values)
    discrepancy = abs(a_hat_full - a_hat_half) / max(
        abs(a_hat_half),
        pilot.useful_response_magnitude_threshold,
    )
    useful = max(abs(a_hat_full), abs(a_hat_half)) >= pilot.useful_response_magnitude_threshold
    return PilotEntry(
        outcome_index,
        intervention_index,
        a_hat_full,
        standard_error(full_values),
        a_hat_half,
        standard_error(half_values),
        discrepancy,
        useful,
        sign_agreement(full_values),
    )


def _eligibility_reasons(
    all_finite: bool,
    entries: list[PilotEntry],
    useful_columns: set[int],
) -> list[str]:
    pilot = active_config().scientific.source_response_pilot
    reasons: list[str] = []
    if not all_finite:
        reasons.append("non-finite shadow state or loss")
    useful_entries = tuple(entry for entry in entries if entry.useful)
    if not useful_entries:
        reasons.append("no useful entries")
        return reasons
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
    return reasons


def _pilot_score(
    useful_entries: tuple[PilotEntry, ...],
) -> float:
    pilot = active_config().scientific.source_response_pilot
    if not useful_entries:
        return math.nan
    signal = statistics.median(
        tuple(
            abs(entry.a_hat_full) / (entry.se_full + pilot.numerical_floor)
            for entry in useful_entries
        )
    )
    curvature = statistics.median(tuple(entry.derivative_discrepancy for entry in useful_entries))
    return signal - pilot.curvature_penalty_coefficient * curvature


def standard_error(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return math.nan
    return statistics.stdev(values) / math.sqrt(len(values))


def sign_agreement(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    positive = sum(1 for value in values if value > 0.0)
    negative = sum(1 for value in values if value < 0.0)
    return max(positive, negative) / len(values)
