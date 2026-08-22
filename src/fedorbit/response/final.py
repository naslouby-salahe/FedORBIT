from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import torch

from fedorbit.config.models import FedorbitConfig, SourceResponseFinalConfig
from fedorbit.models.training import BaseCheckpoint
from fedorbit.response.bootstrap import max_t_critical_value
from fedorbit.response.pilot import DerivativeSeries, PilotData
from fedorbit.response.shadows import (
    ShadowData,
    ShadowSettings,
    paired_shadow_derivative,
    run_shadow_pair,
)
from fedorbit.strict_interface.packet import SourcePacket

RESPONSE_PACKET_SCHEMA = "source-response-packet/v1"


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


class FinalResponseError(ValueError):
    pass


def estimate_final_response(
    config: FedorbitConfig,
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: tuple[tuple[int, ...], ...],
    settings: ShadowSettings,
    seed: int,
) -> FinalResponseEstimate:
    final = config.scientific.source_response_final
    replicate_count = final.paired_replicates_per_intervention
    outcome_count = len(data.outcome_native_class_sets)
    intervention_count = len(intervention_classes)
    series = [
        DerivativeSeries(outcome, intervention, [])
        for outcome in range(outcome_count)
        for intervention in range(intervention_count)
    ]
    all_finite = _collect_final_derivatives(
        config,
        model,
        checkpoint,
        data,
        intervention_classes,
        settings,
        seed,
        replicate_count,
        outcome_count,
        intervention_count,
        final,
        series,
    )
    if not all_finite:
        raise FinalResponseError("non-finite shadow state or loss in final response estimation")
    entry_derivatives = tuple(tuple(entry.values) for entry in series)
    means = tuple(statistics.fmean(values) for values in entry_derivatives)
    standard_errors = tuple(_standard_error(values) for values in entry_derivatives)
    critical = max_t_critical_value(config, entry_derivatives, seed)
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
                lower > 0 or upper < 0
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
    useful_entries = tuple(entry for entry in entries if entry.useful)
    if not useful_entries:
        return FinalResponseEstimate(tuple(entries), critical, len(useful_columns), math.nan, False)
    band_widths = tuple(entry.upper - entry.lower for entry in useful_entries)
    absolute_means = tuple(abs(entry.a_hat) for entry in useful_entries)
    ratio = statistics.median(band_widths) / max(
        statistics.median(absolute_means), final.useful_response_magnitude_threshold
    )
    stable = (
        len(useful_columns) >= final.minimum_useful_intervention_columns
        and ratio <= final.median_band_width_to_median_absolute_mean_response_maximum
    )
    return FinalResponseEstimate(tuple(entries), critical, len(useful_columns), ratio, stable)


def _collect_final_derivatives(
    config: FedorbitConfig,
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: tuple[tuple[int, ...], ...],
    settings: ShadowSettings,
    seed: int,
    replicate_count: int,
    outcome_count: int,
    intervention_count: int,
    final: SourceResponseFinalConfig,
    series: list[DerivativeSeries],
) -> bool:
    all_finite = True
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
            risks = run_shadow_pair(
                config,
                model,
                checkpoint.state_dict,
                checkpoint.optimizer_state,
                checkpoint.rng_state,
                shadow_data,
                settings,
                seed + replicate * (intervention_count + 1) + intervention_index,
            )
            for outcome_index in range(outcome_count):
                positive, negative, baseline = risks[outcome_index]
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
                    all_finite = False
                entry_index = outcome_index * intervention_count + intervention_index
                series[entry_index].values.append(derivative)
    return all_finite


def build_source_packet(
    estimate: FinalResponseEstimate,
    anonymous_fine_node_ids: tuple[str, ...],
    exposed_coarse_group_id: str,
    per_node_train_support: tuple[int, ...],
    per_node_meta_support: tuple[int, ...],
    per_node_effective_replicate_count: tuple[int, ...],
    source_checkpoint_sha256: str,
    response_configuration_sha256: str,
    creation_timestamp: str,
    preprocessing_state_sha256: str = "",
    transfer_node_manifest_sha256: str = "",
    response_seed: int = 0,
) -> SourcePacket:
    def build(integrity: str) -> SourcePacket:
        return SourcePacket(
            anonymous_fine_node_ids=anonymous_fine_node_ids,
            exposed_coarse_group_id=exposed_coarse_group_id,
            L=tuple(entry.lower for entry in estimate.entries),
            U=tuple(entry.upper for entry in estimate.entries),
            per_node_train_support=per_node_train_support,
            per_node_meta_support=per_node_meta_support,
            per_node_effective_replicate_count=per_node_effective_replicate_count,
            packet_schema_metadata=RESPONSE_PACKET_SCHEMA,
            source_checkpoint_sha256=source_checkpoint_sha256,
            response_configuration_sha256=response_configuration_sha256,
            packet_integrity_sha256=integrity,
            packet_validity_state=_validity_state(estimate),
            preprocessing_state_sha256=preprocessing_state_sha256,
            transfer_node_manifest_sha256=transfer_node_manifest_sha256,
            response_seed=response_seed,
            technical_creation_timestamp=creation_timestamp,
        )

    return build(build("").compute_integrity_sha256())


def _validity_state(estimate: FinalResponseEstimate) -> str:
    if not estimate.stability_rule_passed:
        return "unstable"
    return "stable"


def _standard_error(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return math.nan
    return statistics.stdev(values) / math.sqrt(len(values))
