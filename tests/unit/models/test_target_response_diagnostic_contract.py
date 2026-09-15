from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
from torch import nn

import fedorbit.response.uncertainty as response_uncertainty
from fedorbit.config.loading import active_config
from fedorbit.learning.training import (
    BaseCheckpoint,
    ClassWeights,
    ModelParameterState,
    OptimizerState,
    RngState,
    SelectedHyperparameters,
)
from fedorbit.methods.target import (
    OptimizerBudgetError,
    TargetOptimizerBudgetCategory,
    TargetOptimizerReservations,
    TargetOptimizerStepLedger,
    assert_target_diagnostic_reserve,
    target_diagnostic_reserve,
)
from fedorbit.response.estimation import ShadowData, ShadowSettings
from fedorbit.response.pilot import PilotData
from fedorbit.response.uncertainty import (
    FinalResponseEstimate,
    estimate_final_response,
    estimate_response_bands,
)
from fedorbit.types import (
    ClassIndex,
    DirectedPairName,
    ResponseSeedStage,
    TransferMethod,
)

type NativeClassSet = tuple[ClassIndex, ...]
type NativeClassSets = tuple[NativeClassSet, ...]
type DerivativeSeries = tuple[float, ...]
type EntryDerivatives = tuple[DerivativeSeries, ...]
type ShadowRiskTriple = tuple[float, float, float]
type ShadowRiskTriples = tuple[ShadowRiskTriple, ...]

OUTCOME_NATIVE_CLASS_SETS: NativeClassSets = ((ClassIndex(0),), (ClassIndex(1),))
INTERVENTION_CLASSES: NativeClassSets = ((ClassIndex(0),), (ClassIndex(1),))
SHADOW_PAIR_DIRECTIONS = 2

BAND_CRITICAL_VALUE = 2.0
BAND_STANDARD_ERROR = 0.01
BAND_DERIVATIVE = 0.05
SHADOW_POSITIVE_RISK = 1.0
SHADOW_NEGATIVE_RISK = 0.5
SHADOW_BASELINE_RISK = 1.0

ROADMAP_DIAGNOSTIC_MAGNITUDE = 0.10
ROADMAP_DIAGNOSTIC_SHADOW_OPTIMIZER_STEPS = 25
ROADMAP_DIAGNOSTIC_PAIRED_REPLICATES = 8
ROADMAP_DIAGNOSTIC_BOOTSTRAP_RESAMPLES = 1000
ROADMAP_DIAGNOSTIC_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True, slots=True)
class RecordedShadowPair:
    settings: ShadowSettings
    schedule_seed: int


@dataclass(frozen=True, slots=True)
class RecordedDerivative:
    epsilon: float
    denominator_floor: float


@dataclass(frozen=True, slots=True)
class RecordedCriticalValue:
    entry_count: int
    seed: int
    resamples: int | None
    confidence_level: float | None
    standard_error_floor: float | None


@dataclass(frozen=True, slots=True)
class RecordedBandRoute:
    data: PilotData
    settings: ShadowSettings
    seed: int
    replicate_count: int
    bootstrap_resamples: int
    confidence_level: float
    seed_stage: ResponseSeedStage


class RecordingShadowPair:
    def __init__(self) -> None:
        self.calls: list[RecordedShadowPair] = []

    def __call__(
        self,
        _model: nn.Module,
        _base_state: ModelParameterState,
        _base_optimizer_state: OptimizerState,
        _base_rng_state: RngState,
        data: ShadowData,
        settings: ShadowSettings,
        schedule_seed: int,
    ) -> ShadowRiskTriples:
        self.calls.append(RecordedShadowPair(settings, schedule_seed))
        return tuple(
            (SHADOW_POSITIVE_RISK, SHADOW_NEGATIVE_RISK, SHADOW_BASELINE_RISK)
            for _ in range(len(data.outcome_native_class_sets))
        )


class RecordingPairedShadowDerivative:
    def __init__(self) -> None:
        self.calls: list[RecordedDerivative] = []

    def __call__(
        self,
        _positive_risk: float,
        _negative_risk: float,
        _baseline_risk: float,
        epsilon: float,
        denominator_floor: float,
    ) -> float:
        self.calls.append(RecordedDerivative(epsilon, denominator_floor))
        return BAND_DERIVATIVE


class RecordingStandardError:
    def __init__(self) -> None:
        self.calls: list[DerivativeSeries] = []

    def __call__(self, values: DerivativeSeries) -> float:
        self.calls.append(values)
        return BAND_STANDARD_ERROR


class RecordingCriticalValue:
    def __init__(self) -> None:
        self.calls: list[RecordedCriticalValue] = []

    def __call__(
        self,
        entry_derivatives: EntryDerivatives,
        seed: int,
        resamples: int | None = None,
        confidence_level: float | None = None,
        standard_error_floor: float | None = None,
    ) -> float:
        self.calls.append(
            RecordedCriticalValue(
                len(entry_derivatives),
                seed,
                resamples,
                confidence_level,
                standard_error_floor,
            )
        )
        return BAND_CRITICAL_VALUE


class RecordingBandRoute:
    def __init__(self) -> None:
        self.calls: list[RecordedBandRoute] = []

    def __call__(
        self,
        _model: nn.Module,
        _checkpoint: BaseCheckpoint,
        data: PilotData,
        _intervention_classes: NativeClassSets,
        settings: ShadowSettings,
        seed: int,
        *,
        replicate_count: int,
        bootstrap_resamples: int,
        confidence_level: float,
        seed_stage: ResponseSeedStage,
    ) -> FinalResponseEstimate:
        self.calls.append(
            RecordedBandRoute(
                data,
                settings,
                seed,
                replicate_count,
                bootstrap_resamples,
                confidence_level,
                seed_stage,
            )
        )
        return FinalResponseEstimate((), BAND_CRITICAL_VALUE, 0, math.nan, False)


def _toy_model() -> nn.Module:
    return nn.Linear(3, 2)


def _toy_checkpoint() -> BaseCheckpoint:
    config = active_config()
    return BaseCheckpoint(
        epoch=0,
        valid_macro_cross_entropy=1.0,
        state_dict=ModelParameterState(()),
        optimizer_state=OptimizerState(b""),
        rng_state=RngState(torch.get_rng_state().clone(), ()),
        selected_hyperparameters=SelectedHyperparameters(
            config.scientific.base_model_pilot.reference_learning_rate,
            0.0,
            0.0,
        ),
        train_class_weights=ClassWeights(torch.ones(2)),
    )


def _toy_pilot_data() -> PilotData:
    config = active_config()
    return PilotData(
        train_features=torch.zeros((4, 3)),
        train_targets=torch.tensor((0, 1, 0, 1)),
        meta_features=torch.zeros((2, 3)),
        meta_targets=torch.tensor((0, 1)),
        outcome_native_class_sets=OUTCOME_NATIVE_CLASS_SETS,
        base_class_weights=ClassWeights(torch.ones(2)),
        learning_rate=config.scientific.base_model_pilot.reference_learning_rate,
        weight_decay=0.0,
    )


def _shadow_settings(epsilon: float, horizon: int) -> ShadowSettings:
    return ShadowSettings(
        epsilon,
        horizon,
        active_config().scientific.base_model_pilot.reference_learning_rate,
        0.0,
    )


def _diagnostic_settings() -> ShadowSettings:
    diagnostic = active_config().scientific.target_response_diagnostic
    return _shadow_settings(diagnostic.intervention_magnitude, diagnostic.shadow_optimizer_steps)


def _diagnostic_intervention_class_count() -> int:
    diagnostic = active_config().scientific.target_response_diagnostic
    budget = active_config().scientific.target_optimizer_budget
    steps_per_column = (
        diagnostic.paired_replicates * SHADOW_PAIR_DIRECTIONS * diagnostic.shadow_optimizer_steps
    )
    reserved = budget.reserved.target_response_diagnostic
    assert reserved % steps_per_column == 0
    return reserved // steps_per_column


def test_registered_target_response_diagnostic_settings_match_the_roadmap() -> None:
    diagnostic = active_config().scientific.target_response_diagnostic
    assert diagnostic.intervention_magnitude == ROADMAP_DIAGNOSTIC_MAGNITUDE
    assert diagnostic.shadow_optimizer_steps == ROADMAP_DIAGNOSTIC_SHADOW_OPTIMIZER_STEPS
    assert diagnostic.paired_replicates == ROADMAP_DIAGNOSTIC_PAIRED_REPLICATES
    assert diagnostic.simultaneous_bootstrap_resamples == ROADMAP_DIAGNOSTIC_BOOTSTRAP_RESAMPLES
    assert diagnostic.confidence_level == ROADMAP_DIAGNOSTIC_CONFIDENCE_LEVEL


def test_final_response_estimator_route_passes_the_registered_final_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = active_config().scientific.source_response_final
    settings = _shadow_settings(
        active_config().scientific.source_response_pilot.intervention_magnitudes[0],
        active_config().scientific.source_response_pilot.optimizer_step_horizons[0],
    )
    data = _toy_pilot_data()
    checkpoint = _toy_checkpoint()
    model = _toy_model()
    seed = active_config().scientific.randomness.pilot_seeds[0]
    route = RecordingBandRoute()
    monkeypatch.setattr(response_uncertainty, "estimate_response_bands", route)
    estimate = estimate_final_response(
        model,
        checkpoint,
        data,
        INTERVENTION_CLASSES,
        settings,
        seed,
    )
    assert len(route.calls) == 1
    call = route.calls[0]
    assert call.data is data
    assert call.settings == settings
    assert call.seed == seed
    assert call.replicate_count == final.paired_replicates_per_intervention
    assert call.bootstrap_resamples == final.max_t_bootstrap_resamples
    assert call.confidence_level == final.simultaneous_confidence_level
    assert call.seed_stage is ResponseSeedStage.FINAL_SOURCE_RESPONSE
    assert estimate.critical_value == BAND_CRITICAL_VALUE


def test_target_local_diagnostic_reuses_the_final_estimator_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = active_config()
    diagnostic = config.scientific.target_response_diagnostic
    final = config.scientific.source_response_final
    settings = _diagnostic_settings()
    shadows = RecordingShadowPair()
    derivatives = RecordingPairedShadowDerivative()
    standard_errors = RecordingStandardError()
    critical_values = RecordingCriticalValue()
    monkeypatch.setattr(response_uncertainty, "run_shadow_pair", shadows)
    monkeypatch.setattr(response_uncertainty, "paired_shadow_derivative", derivatives)
    monkeypatch.setattr(response_uncertainty, "standard_error", standard_errors)
    monkeypatch.setattr(response_uncertainty, "max_t_critical_value", critical_values)
    seed = config.scientific.randomness.pilot_seeds[0]
    estimate = estimate_response_bands(
        _toy_model(),
        _toy_checkpoint(),
        _toy_pilot_data(),
        INTERVENTION_CLASSES,
        settings,
        seed,
        replicate_count=diagnostic.paired_replicates,
        bootstrap_resamples=diagnostic.simultaneous_bootstrap_resamples,
        confidence_level=diagnostic.confidence_level,
        seed_stage=ResponseSeedStage.TARGET_LOCAL_DIAGNOSTIC,
    )
    paired_schedules = diagnostic.paired_replicates * len(INTERVENTION_CLASSES)
    assert len(shadows.calls) == paired_schedules
    assert all(call.settings == settings for call in shadows.calls)
    assert all(call.settings.epsilon == diagnostic.intervention_magnitude for call in shadows.calls)
    assert all(call.settings.horizon == diagnostic.shadow_optimizer_steps for call in shadows.calls)
    assert len({call.schedule_seed for call in shadows.calls}) == paired_schedules
    assert len(derivatives.calls) == paired_schedules * len(OUTCOME_NATIVE_CLASS_SETS)
    assert all(call.epsilon == diagnostic.intervention_magnitude for call in derivatives.calls)
    assert all(
        call.denominator_floor == final.response_risk_denominator_floor
        for call in derivatives.calls
    )
    assert len(standard_errors.calls) == len(OUTCOME_NATIVE_CLASS_SETS) * len(INTERVENTION_CLASSES)
    assert all(len(values) == diagnostic.paired_replicates for values in standard_errors.calls)
    assert len(critical_values.calls) == 1
    critical = critical_values.calls[0]
    assert critical.entry_count == len(OUTCOME_NATIVE_CLASS_SETS) * len(INTERVENTION_CLASSES)
    assert critical.resamples == diagnostic.simultaneous_bootstrap_resamples
    assert critical.confidence_level == diagnostic.confidence_level
    assert critical.standard_error_floor == final.response_standard_error_floor
    assert estimate.critical_value == BAND_CRITICAL_VALUE
    assert len(estimate.entries) == len(OUTCOME_NATIVE_CLASS_SETS) * len(INTERVENTION_CLASSES)


def test_target_diagnostic_reserve_matches_the_registered_reservation() -> None:
    diagnostic = active_config().scientific.target_response_diagnostic
    reserved = (
        active_config().scientific.target_optimizer_budget.reserved.target_response_diagnostic
    )
    intervention_class_count = _diagnostic_intervention_class_count()
    assert (
        target_diagnostic_reserve(intervention_class_count)
        == intervention_class_count
        * diagnostic.paired_replicates
        * SHADOW_PAIR_DIRECTIONS
        * diagnostic.shadow_optimizer_steps
    )
    assert target_diagnostic_reserve(intervention_class_count) == reserved
    assert_target_diagnostic_reserve(intervention_class_count)
    with pytest.raises(OptimizerBudgetError):
        assert_target_diagnostic_reserve(intervention_class_count + 1)
    with pytest.raises(OptimizerBudgetError):
        assert_target_diagnostic_reserve(intervention_class_count - 1)


def test_target_diagnostic_reserve_is_a_subset_of_the_registered_budget() -> None:
    config = active_config()
    budget = config.scientific.target_optimizer_budget
    reserved = budget.reserved
    reservations = TargetOptimizerReservations(
        reserved.target_response_diagnostic,
        reserved.confirmation_candidates,
        reserved.live_assimilation,
        reserved.nontransferable_safety_reserve,
    )
    assert reservations.total_steps <= budget.maximum_steps_per_method_pair_seed_before_test
    assert (
        reserved.target_response_diagnostic <= budget.maximum_steps_per_method_pair_seed_before_test
    )
    ledger = TargetOptimizerStepLedger(
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        DirectedPairName("source -> target"),
        config.scientific.randomness.pilot_seeds[0],
    )
    category = TargetOptimizerBudgetCategory.TARGET_RESPONSE_DIAGNOSTIC
    assert ledger.remaining(category) == reserved.target_response_diagnostic
    ledger.consume(category, reserved.target_response_diagnostic)
    assert ledger.remaining(category) == 0
    with pytest.raises(OptimizerBudgetError):
        ledger.consume(
            category,
            config.scientific.target_response_diagnostic.shadow_optimizer_steps,
        )


def test_target_diagnostic_reserve_decomposes_into_registered_per_class_units() -> None:
    config = active_config()
    diagnostic = config.scientific.target_response_diagnostic
    reserved = config.scientific.target_optimizer_budget.reserved.target_response_diagnostic
    units = (
        diagnostic.paired_replicates * SHADOW_PAIR_DIRECTIONS * diagnostic.shadow_optimizer_steps
    )
    intervention_class_count = _diagnostic_intervention_class_count()
    assert reserved == units * intervention_class_count
    ledger = TargetOptimizerStepLedger(
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        DirectedPairName("source -> target"),
        config.scientific.randomness.pilot_seeds[0],
    )
    category = TargetOptimizerBudgetCategory.TARGET_RESPONSE_DIAGNOSTIC
    for _ in range(intervention_class_count):
        ledger.consume(category, units)
    assert ledger.remaining(category) == 0
