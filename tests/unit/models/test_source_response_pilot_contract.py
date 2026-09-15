from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, replace

import pytest
import torch
from torch import nn

import fedorbit.response.pilot as response_pilot
from fedorbit.config.loading import active_config
from fedorbit.learning.training import (
    BaseCheckpoint,
    ClassWeights,
    ModelParameterState,
    OptimizerState,
    RngState,
    SelectedHyperparameters,
)
from fedorbit.response.estimation import NonFiniteShadowLossError, ShadowData, ShadowSettings
from fedorbit.response.pilot import (
    CandidateResult,
    PilotCheckpoint,
    PilotData,
    ResponseCandidate,
    ResponsePilotError,
    run_pooled_source_response_pilot,
    select_response_configuration,
)
from fedorbit.types import ClassIndex, IneligibilityReason

type NativeClassSet = tuple[ClassIndex, ...]
type NativeClassSets = tuple[NativeClassSet, ...]
type ShadowRiskTriple = tuple[float, float, float]
type ShadowRiskTriples = tuple[ShadowRiskTriple, ...]
type DerivativeSchedule = Callable[[float, int, int, int, int], float]

OUTCOME_NATIVE_CLASS_SETS: NativeClassSets = ((ClassIndex(0),), (ClassIndex(1),))
INTERVENTION_CLASSES: NativeClassSets = ((ClassIndex(0),), (ClassIndex(1),))

ENTRY_COUNT = len(OUTCOME_NATIVE_CLASS_SETS) * len(INTERVENTION_CLASSES)

SCORE_MEANS = (0.1, 0.2, 0.3, 0.7)
SCORE_SPREADS = (0.02, 0.005, 0.001, 0.05)
SCORE_SLOPES = (0.2, 0.5, 0.8, 0.2)
SCORE_SPREAD_MAGNITUDE_SLOPE = 0.5

SIGN_POSITIVE_REPLICATES = 5
SIGN_POSITIVE_VALUE = 0.06
SIGN_NEGATIVE_VALUE = -0.02

LINEARITY_CURVED_FLOOR = 0.05
LINEARITY_CURVED_SLOPE = 1.0
LINEARITY_STRAIGHT_FLOOR = 0.1
LINEARITY_STRAIGHT_SLOPE = 0.1

SELECTION_MAGNITUDE_VALUE = 0.01


@dataclass(frozen=True, slots=True)
class RecordedShadowPair:
    epsilon: float
    horizon: int
    schedule_seed: int


class RecordingShadowPair:
    def __init__(self, schedule: DerivativeSchedule, intervention_classes: NativeClassSets) -> None:
        self._schedule = schedule
        self._intervention_classes = intervention_classes
        self._calls_per_column: dict[tuple[float, int], int] = {}
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
        self.calls.append(RecordedShadowPair(settings.epsilon, settings.horizon, schedule_seed))
        intervention_index = self._intervention_classes.index(data.intervention_classes)
        column = (settings.epsilon, intervention_index)
        position = self._calls_per_column.get(column, 0)
        self._calls_per_column[column] = position + 1
        replicate_index = (
            position
            % active_config().scientific.source_response_pilot.paired_schedules_per_candidate
        )
        return tuple(
            self._risk_triple(
                settings.epsilon,
                settings.horizon,
                intervention_index,
                outcome_index,
                replicate_index,
            )
            for outcome_index in range(len(data.outcome_native_class_sets))
        )

    def _risk_triple(
        self,
        epsilon: float,
        horizon: int,
        intervention_index: int,
        outcome_index: int,
        replicate_index: int,
    ) -> ShadowRiskTriple:
        derivative = self._schedule(
            epsilon,
            horizon,
            intervention_index,
            outcome_index,
            replicate_index,
        )
        return (0.0, 2.0 * derivative * epsilon, 1.0)


def _constant_schedule(value: float) -> DerivativeSchedule:
    return lambda _epsilon, _horizon, _intervention, _outcome, _replicate: value


def _epsilon_schedule(values: dict[float, float]) -> DerivativeSchedule:
    return lambda epsilon, _horizon, _intervention, _outcome, _replicate: values[epsilon]


def _linearity_schedule(floor: float, slope: float) -> DerivativeSchedule:
    return lambda epsilon, _horizon, _intervention, _outcome, _replicate: floor + slope * epsilon


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


def _toy_checkpoints() -> tuple[PilotCheckpoint, ...]:
    config = active_config()
    reference_learning_rate = config.scientific.base_model_pilot.reference_learning_rate
    return tuple(
        PilotCheckpoint(
            model=nn.Linear(3, 2),
            checkpoint=BaseCheckpoint(
                epoch=0,
                valid_macro_cross_entropy=1.0,
                state_dict=ModelParameterState(()),
                optimizer_state=OptimizerState(b""),
                rng_state=RngState(torch.get_rng_state().clone(), ()),
                selected_hyperparameters=SelectedHyperparameters(
                    reference_learning_rate,
                    0.0,
                    0.0,
                ),
                train_class_weights=ClassWeights(torch.ones(2)),
            ),
            seed=seed,
        )
        for seed in config.scientific.randomness.pilot_seeds
    )


def _run_pilot(
    monkeypatch: pytest.MonkeyPatch,
    schedule: DerivativeSchedule,
) -> tuple[tuple[CandidateResult, ...], RecordingShadowPair]:
    recorder = RecordingShadowPair(schedule, INTERVENTION_CLASSES)
    monkeypatch.setattr(response_pilot, "run_shadow_pair", recorder)
    results = run_pooled_source_response_pilot(
        _toy_checkpoints(),
        _toy_pilot_data(),
        INTERVENTION_CLASSES,
    )
    return results, recorder


def _result_for_magnitude(
    results: tuple[CandidateResult, ...], magnitude: float
) -> CandidateResult:
    return next(
        result for result in results if result.candidate.intervention_magnitude == magnitude
    )


def _useful_columns(result: CandidateResult) -> set[int]:
    return {entry.intervention_index for entry in result.entries if entry.useful}


def _score_entry_value(epsilon: float, entry_index: int, replicate_index: int) -> float:
    offset = (
        active_config().scientific.source_response_pilot.paired_schedules_per_candidate - 1
    ) / 2.0
    mean = SCORE_MEANS[entry_index] * (1.0 + SCORE_SLOPES[entry_index] * epsilon)
    spread = SCORE_SPREADS[entry_index] * (1.0 + SCORE_SPREAD_MAGNITUDE_SLOPE * epsilon)
    return mean + spread * (replicate_index - offset)


def _score_schedule(
    epsilon: float,
    _horizon: int,
    intervention_index: int,
    outcome_index: int,
    replicate_index: int,
) -> float:
    entry_index = outcome_index * len(INTERVENTION_CLASSES) + intervention_index
    return _score_entry_value(epsilon, entry_index, replicate_index)


def _score_entry_series(epsilon: float, entry_index: int) -> tuple[float, ...]:
    config = active_config()
    return tuple(
        _score_entry_value(epsilon, entry_index, replicate_index)
        for _ in range(len(config.scientific.randomness.pilot_seeds))
        for replicate_index in range(
            config.scientific.source_response_pilot.paired_schedules_per_candidate
        )
    )


def _expected_pilot_score(magnitude: float) -> float:
    pilot = active_config().scientific.source_response_pilot
    signals: list[float] = []
    discrepancies: list[float] = []
    for entry_index in range(ENTRY_COUNT):
        full_series = _score_entry_series(magnitude, entry_index)
        half_series = _score_entry_series(magnitude / 2.0, entry_index)
        full_mean = statistics.fmean(full_series)
        half_mean = statistics.fmean(half_series)
        error = statistics.stdev(full_series) / math.sqrt(len(full_series))
        signals.append(abs(full_mean) / (error + pilot.numerical_floor))
        discrepancies.append(
            abs(full_mean - half_mean)
            / max(abs(half_mean), pilot.useful_response_magnitude_threshold)
        )
    return statistics.median(signals) - pilot.curvature_penalty_coefficient * statistics.median(
        discrepancies
    )


def _candidate_result(candidate: ResponseCandidate, pilot_score: float) -> CandidateResult:
    return CandidateResult(candidate, (), True, (), pilot_score)


def _ineligible_candidate_result(
    candidate: ResponseCandidate,
    reason: IneligibilityReason,
) -> CandidateResult:
    return CandidateResult(candidate, (), False, (reason,), math.nan)


def test_candidate_expansion_is_the_registered_cartesian_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = active_config().scientific.source_response_pilot
    results, _recorder = _run_pilot(monkeypatch, _constant_schedule(0.05))
    expected = tuple(
        ResponseCandidate(magnitude, horizon)
        for magnitude in pilot.intervention_magnitudes
        for horizon in pilot.optimizer_step_horizons
    )
    assert len(results) == len(pilot.intervention_magnitudes) * len(pilot.optimizer_step_horizons)
    assert tuple(result.candidate for result in results) == expected
    assert len(set(expected)) == len(results)
    assert all(len(result.entries) == ENTRY_COUNT for result in results)


def test_pooled_replicate_structure_pairs_full_and_half_magnitudes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = active_config()
    pilot = config.scientific.source_response_pilot
    pilot_seeds = config.scientific.randomness.pilot_seeds
    results, recorder = _run_pilot(monkeypatch, _constant_schedule(0.05))
    schedules_per_candidate = (
        len(pilot_seeds) * pilot.paired_schedules_per_candidate * len(INTERVENTION_CLASSES)
    )
    assert len(recorder.calls) == len(results) * 2 * schedules_per_candidate
    calls_per_schedule: dict[int, list[RecordedShadowPair]] = {}
    for call in recorder.calls:
        calls_per_schedule.setdefault(call.schedule_seed, []).append(call)
    assert len(calls_per_schedule) == len(results) * schedules_per_candidate
    pairs_per_candidate: dict[ResponseCandidate, int] = {}
    epsilons_per_candidate: dict[ResponseCandidate, set[float]] = {}
    for grouped in calls_per_schedule.values():
        assert len(grouped) == 2
        assert grouped[0].horizon == grouped[1].horizon
        epsilons = {grouped[0].epsilon, grouped[1].epsilon}
        assert max(epsilons) == 2.0 * min(epsilons)
        candidate = ResponseCandidate(max(epsilons), grouped[0].horizon)
        pairs_per_candidate[candidate] = pairs_per_candidate.get(candidate, 0) + 1
        epsilons_per_candidate.setdefault(candidate, set()).update(epsilons)
    assert set(pairs_per_candidate) == {result.candidate for result in results}
    assert set(pairs_per_candidate.values()) == {schedules_per_candidate}
    for candidate, epsilons in epsilons_per_candidate.items():
        assert candidate.intervention_magnitude in pilot.intervention_magnitudes
        assert epsilons == {
            candidate.intervention_magnitude,
            candidate.intervention_magnitude / 2.0,
        }


def test_registered_source_response_pilot_plan_has_108_cells() -> None:
    config = active_config()
    pilot = config.scientific.source_response_pilot
    candidates = len(pilot.intervention_magnitudes) * len(pilot.optimizer_step_horizons)
    clients = len(config.scientific.datasets.clients)
    pilot_seeds = len(config.scientific.randomness.pilot_seeds)
    assert clients * pilot_seeds * candidates == 108


def test_usefulness_uses_the_registered_maximum_of_full_and_half_magnitudes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = active_config().scientific.source_response_pilot
    magnitudes = pilot.intervention_magnitudes
    threshold = pilot.useful_response_magnitude_threshold
    just_below = math.nextafter(threshold, 0.0)
    schedule = _epsilon_schedule(
        {
            magnitudes[0]: 0.0,
            magnitudes[0] / 2.0: threshold,
            magnitudes[1]: just_below,
            magnitudes[2]: 0.0,
        }
    )
    results, _recorder = _run_pilot(monkeypatch, schedule)
    for result in results:
        for entry in result.entries:
            assert entry.useful is (max(abs(entry.a_hat_full), abs(entry.a_hat_half)) >= threshold)
    at_threshold = _result_for_magnitude(results, magnitudes[0])
    for entry in at_threshold.entries:
        assert entry.a_hat_full == 0.0
        assert entry.a_hat_half == threshold
        assert entry.useful is True
    below_threshold = _result_for_magnitude(results, magnitudes[2])
    for entry in below_threshold.entries:
        assert entry.a_hat_half == pytest.approx(just_below, rel=1.0e-15)
        assert max(abs(entry.a_hat_full), abs(entry.a_hat_half)) < threshold
        assert entry.useful is False


def test_linearity_ceiling_rejects_a_median_discrepancy_above_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = active_config().scientific.source_response_pilot
    results, _recorder = _run_pilot(
        monkeypatch,
        _linearity_schedule(LINEARITY_CURVED_FLOOR, LINEARITY_CURVED_SLOPE),
    )
    for result in results:
        useful = tuple(entry for entry in result.entries if entry.useful)
        assert len(useful) == ENTRY_COUNT
        assert (
            statistics.median(tuple(entry.derivative_discrepancy for entry in useful))
            > pilot.relative_derivative_discrepancy_ceiling
        )
        assert result.ineligibility_reasons == (
            IneligibilityReason.DERIVATIVE_DISCREPANCY_ABOVE_CEILING,
        )
        assert result.eligible is False


def test_linearity_ceiling_accepts_a_median_discrepancy_below_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = active_config().scientific.source_response_pilot
    results, _recorder = _run_pilot(
        monkeypatch,
        _linearity_schedule(LINEARITY_STRAIGHT_FLOOR, LINEARITY_STRAIGHT_SLOPE),
    )
    for result in results:
        useful = tuple(entry for entry in result.entries if entry.useful)
        assert len(useful) == ENTRY_COUNT
        assert (
            statistics.median(tuple(entry.derivative_discrepancy for entry in useful))
            <= pilot.relative_derivative_discrepancy_ceiling
        )
        assert IneligibilityReason.DERIVATIVE_DISCREPANCY_ABOVE_CEILING not in (
            result.ineligibility_reasons
        )
        assert result.eligible is True


def test_sign_agreement_minimum_rejects_a_median_below_the_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = active_config()
    pilot = config.scientific.source_response_pilot
    replicates = pilot.paired_schedules_per_candidate
    series = (SIGN_POSITIVE_VALUE,) * SIGN_POSITIVE_REPLICATES + (SIGN_NEGATIVE_VALUE,) * (
        replicates - SIGN_POSITIVE_REPLICATES
    )

    def schedule(
        _epsilon: float,
        _horizon: int,
        _intervention: int,
        _outcome: int,
        replicate: int,
    ) -> float:
        return series[replicate]

    results, _recorder = _run_pilot(monkeypatch, schedule)
    expected_agreement = max(
        sum(1 for value in series if value > 0.0),
        sum(1 for value in series if value < 0.0),
    ) / len(series)
    assert expected_agreement < pilot.sign_agreement_minimum
    for result in results:
        useful = tuple(entry for entry in result.entries if entry.useful)
        assert len(useful) == ENTRY_COUNT
        assert (
            statistics.median(tuple(entry.sign_agreement for entry in useful))
            < pilot.sign_agreement_minimum
        )
        assert result.ineligibility_reasons == (IneligibilityReason.SIGN_AGREEMENT_BELOW_MINIMUM,)
        assert result.eligible is False


def test_sign_agreement_minimum_accepts_a_median_at_or_above_the_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = active_config().scientific.source_response_pilot
    results, _recorder = _run_pilot(monkeypatch, _constant_schedule(0.05))
    for result in results:
        useful = tuple(entry for entry in result.entries if entry.useful)
        assert len(useful) == ENTRY_COUNT
        assert all(entry.sign_agreement >= pilot.sign_agreement_minimum for entry in useful)
        assert IneligibilityReason.SIGN_AGREEMENT_BELOW_MINIMUM not in result.ineligibility_reasons


def test_too_few_useful_intervention_columns_rejects_a_single_useful_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = active_config().scientific.source_response_pilot
    low_value = pilot.useful_response_magnitude_threshold / 10.0
    high_value = pilot.useful_response_magnitude_threshold * 10.0

    def schedule(
        _epsilon: float,
        _horizon: int,
        intervention: int,
        _outcome: int,
        _replicate: int,
    ) -> float:
        return high_value if intervention == 1 else low_value

    results, _recorder = _run_pilot(monkeypatch, schedule)
    for result in results:
        useful_columns = _useful_columns(result)
        assert useful_columns == {1}
        assert len(useful_columns) < pilot.minimum_useful_intervention_columns
        assert result.ineligibility_reasons == (
            IneligibilityReason.TOO_FEW_USEFUL_INTERVENTION_COLUMNS,
        )
        assert result.eligible is False


def test_non_finite_shadow_loss_marks_the_candidate_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def schedule(
        _epsilon: float,
        _horizon: int,
        _intervention: int,
        _outcome: int,
        _replicate: int,
    ) -> float:
        raise NonFiniteShadowLossError("non-finite shadow loss")

    results, _recorder = _run_pilot(monkeypatch, schedule)
    for result in results:
        assert result.ineligibility_reasons == (
            IneligibilityReason.NON_FINITE_SHADOW_STATE,
            IneligibilityReason.NO_USEFUL_ENTRIES,
        )
        assert result.eligible is False


def test_no_useful_entry_rejects_the_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    pilot = active_config().scientific.source_response_pilot
    results, _recorder = _run_pilot(
        monkeypatch,
        _constant_schedule(pilot.useful_response_magnitude_threshold / 10.0),
    )
    for result in results:
        assert result.ineligibility_reasons == (IneligibilityReason.NO_USEFUL_ENTRIES,)
        assert result.eligible is False
        assert math.isnan(result.pilot_score)


def test_pilot_score_is_median_signal_less_the_registered_curvature_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = active_config().scientific.source_response_pilot
    results, _recorder = _run_pilot(monkeypatch, _score_schedule)
    expected_by_magnitude = {
        magnitude: _expected_pilot_score(magnitude) for magnitude in pilot.intervention_magnitudes
    }
    for result in results:
        assert result.ineligibility_reasons == ()
        assert result.eligible is True
        assert len(result.entries) == ENTRY_COUNT
        assert result.pilot_score == pytest.approx(
            expected_by_magnitude[result.candidate.intervention_magnitude],
            rel=1.0e-12,
        )


def test_selection_prefers_the_highest_registered_pilot_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = active_config().scientific.source_response_pilot

    def schedule(
        _epsilon: float,
        horizon: int,
        _intervention: int,
        _outcome: int,
        _replicate: int,
    ) -> float:
        return SELECTION_MAGNITUDE_VALUE * horizon

    results, _recorder = _run_pilot(monkeypatch, schedule)
    selected = select_response_configuration(results)
    assert selected == ResponseCandidate(
        min(pilot.intervention_magnitudes),
        max(pilot.optimizer_step_horizons),
    )
    scores = {result.candidate: result.pilot_score for result in results}
    assert scores[selected] == max(scores.values())


def test_selection_breaks_score_ties_by_smallest_horizon_then_magnitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = active_config().scientific.source_response_pilot
    results, _recorder = _run_pilot(monkeypatch, _constant_schedule(0.05))
    assert len({result.pilot_score for result in results}) == 1
    assert select_response_configuration(results) == ResponseCandidate(
        min(pilot.intervention_magnitudes),
        min(pilot.optimizer_step_horizons),
    )


def test_selection_orders_equivalent_scores_by_horizon_then_magnitude() -> None:
    pilot = active_config().scientific.source_response_pilot
    magnitudes = pilot.intervention_magnitudes
    horizons = pilot.optimizer_step_horizons
    results = (
        _candidate_result(ResponseCandidate(magnitudes[0], horizons[1]), 5.0),
        _candidate_result(ResponseCandidate(magnitudes[2], horizons[0]), 5.0),
        _candidate_result(ResponseCandidate(magnitudes[1], horizons[0]), 5.0),
    )
    assert select_response_configuration(results) == ResponseCandidate(
        magnitudes[1],
        horizons[0],
    )


def test_selection_raises_when_no_candidate_is_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    pilot = active_config().scientific.source_response_pilot
    results, _recorder = _run_pilot(monkeypatch, _constant_schedule(0.0))
    assert all(result.eligible is False for result in results)
    with pytest.raises(ResponsePilotError):
        select_response_configuration(results)
    with pytest.raises(ResponsePilotError):
        select_response_configuration(
            tuple(
                _ineligible_candidate_result(
                    ResponseCandidate(magnitude, horizon),
                    IneligibilityReason.NO_USEFUL_ENTRIES,
                )
                for magnitude in pilot.intervention_magnitudes
                for horizon in pilot.optimizer_step_horizons
            )
        )


def test_pilot_requires_one_checkpoint_per_registered_pilot_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = active_config()
    pilot_seeds = config.scientific.randomness.pilot_seeds
    monkeypatch.setattr(
        response_pilot,
        "run_shadow_pair",
        RecordingShadowPair(_constant_schedule(0.05), INTERVENTION_CLASSES),
    )
    checkpoints = _toy_checkpoints()
    assert len(checkpoints) == len(pilot_seeds)
    with pytest.raises(ResponsePilotError):
        run_pooled_source_response_pilot(checkpoints[:-1], _toy_pilot_data(), INTERVENTION_CLASSES)
    with pytest.raises(ResponsePilotError):
        run_pooled_source_response_pilot(
            (*checkpoints, checkpoints[0]),
            _toy_pilot_data(),
            INTERVENTION_CLASSES,
        )
    duplicated = tuple(replace(checkpoint, seed=pilot_seeds[0]) for checkpoint in checkpoints)
    with pytest.raises(ResponsePilotError):
        run_pooled_source_response_pilot(
            duplicated,
            _toy_pilot_data(),
            INTERVENTION_CLASSES,
        )
    registered = run_pooled_source_response_pilot(
        checkpoints,
        _toy_pilot_data(),
        INTERVENTION_CLASSES,
    )
    pilot = config.scientific.source_response_pilot
    assert len(registered) == len(pilot.intervention_magnitudes) * len(
        pilot.optimizer_step_horizons
    )
