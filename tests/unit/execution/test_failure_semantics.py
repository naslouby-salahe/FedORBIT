from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.analysis.records import MetricDirection
from fedorbit.config.loading import active_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.experiments import solvers as experiment_solvers
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.experiments.scoring import persist_ineligible_transfer_cell
from fedorbit.experiments.solvers import (
    execute_exact_sparse_solver_benchmark,
    execute_map_dependent_action_boundary,
)
from fedorbit.experiments.synthesis import completed_experiment_metric_records
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.failures import (
    FilesystemFailureError,
    LeakageError,
    RetryPolicy,
    classify_failure,
    scientific_algorithmic_failure_outcome,
    validation_failure_outcome,
)
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.optimization.exact_qap import QapSeparatorResult, QapUncertifiedError
from fedorbit.optimization.objective import CurriculumAction, RobustActionProblem
from fedorbit.types import (
    ArtifactState,
    DatasetId,
    DirectedPairName,
    EvaluationConditionName,
    ExperimentName,
    FailureCategory,
    FailureReason,
    MetricId,
    OverwritePolicy,
    PairSeedIneligibilityReason,
    ScalabilityBlockPattern,
    ScientificAlgorithmicFailureError,
    StrictResourceViolationError,
    TerminalState,
    TransferMethod,
)


def _request(experiment: ExperimentName, overwrite: OverwritePolicy) -> ExperimentExecutionRequest:
    catalogue = build_catalogue()
    return ExperimentExecutionRequest(
        experiment=experiment,
        definition=catalogue.definition(experiment),
        overwrite_policy=overwrite,
    )


def _with_single_seed(config: FedorbitConfig) -> FedorbitConfig:
    return config.model_copy(
        update={
            "scientific": config.scientific.model_copy(
                update={
                    "randomness": config.scientific.randomness.model_copy(
                        update={"confirmatory_seeds": (1103,)}
                    )
                }
            )
        }
    )


def _single_fixture_boundary_config() -> FedorbitConfig:
    config = _with_single_seed(active_config())
    return config.model_copy(
        update={
            "experiments": config.experiments.model_copy(
                update={
                    "map_dependent_action_boundary": (
                        config.experiments.map_dependent_action_boundary.model_copy(
                            update={"fixtures_per_seed": 1}
                        )
                    )
                }
            )
        }
    )


def _single_cell_benchmark_config() -> FedorbitConfig:
    config = _with_single_seed(active_config())
    benchmark = config.experiments.exact_sparse_solver_benchmark
    return config.model_copy(
        update={
            "experiments": config.experiments.model_copy(
                update={
                    "exact_sparse_solver_benchmark": benchmark.model_copy(
                        update={
                            "synthetic_k": benchmark.synthetic_k.model_copy(
                                update={"minimum": 4, "maximum": 4}
                            ),
                            "block_patterns": (ScalabilityBlockPattern.BALANCED,),
                            "supports": (1,),
                            "methods": (
                                TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                                TransferMethod.GENERIC_EXACT_QAP,
                            ),
                        }
                    )
                }
            )
        }
    )


def test_infrastructure_retry_is_exactly_two_retries_and_three_attempts() -> None:
    retries = active_config().runtime.failure_handling.retries_after_initial_infrastructure_failure
    assert retries == 2
    policy = RetryPolicy(retries)
    classification = classify_failure(FilesystemFailureError("fixture infrastructure failure"))
    assert classification.category is FailureCategory.INFRASTRUCTURE
    decisions = tuple(policy.decide(attempt, classification) for attempt in range(3))
    assert [decision.retry for decision in decisions] == [True, True, False]
    assert decisions[0].remaining_attempts == 2
    assert decisions[1].remaining_attempts == 1
    assert decisions[2].terminal_state is TerminalState.FAILED_INFRASTRUCTURE


def test_validation_failure_distinguishes_invalid_from_failed() -> None:
    reason = FailureReason("fixture validation reason")
    invalid = validation_failure_outcome(reason, True)
    failed = validation_failure_outcome(reason, False)
    assert invalid.terminal_state is TerminalState.INVALID
    assert failed.terminal_state is TerminalState.FAILED_VALIDATION
    assert invalid.failure_category is FailureCategory.VALIDATION
    for error in (StrictResourceViolationError("fixture"), LeakageError("fixture")):
        classification = classify_failure(error)
        assert classification.category is FailureCategory.VALIDATION
        assert classification.retryable is False
        decision = RetryPolicy(2).decide(0, classification)
        assert decision.retry is False
        assert decision.remaining_attempts == 0


def test_scientific_algorithmic_failure_is_terminal_without_adaptive_retry() -> None:
    error = ScientificAlgorithmicFailureError("sparse master hit its cut cap")
    classification = classify_failure(error)
    assert classification.category is FailureCategory.SCIENTIFIC_ALGORITHMIC
    decision = RetryPolicy(2).decide(0, classification)
    assert decision.retry is False
    assert decision.terminal_state is TerminalState.FAILED_SCIENTIFIC_ALGORITHMIC
    outcome = scientific_algorithmic_failure_outcome(FailureReason(str(error)), False, (), ())
    assert outcome.terminal_state is TerminalState.FAILED_SCIENTIFIC_ALGORITHMIC
    assert outcome.proceeded_to_confirmation is False
    assert outcome.proceeded_to_assimilation is False
    assert outcome.proceeded_to_test_scoring is False
    assert outcome.certified_action_available is False


def test_boundary_and_abstention_outcomes_complete_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = _request(ExperimentName.MAP_DEPENDENT_ACTION_BOUNDARY, OverwritePolicy.REPLACE)
    monkeypatch.setattr(experiment_solvers, "active_config", _single_fixture_boundary_config)
    execute_map_dependent_action_boundary(store, layout, request)
    boundary_records = completed_experiment_metric_records(store, request.experiment)
    assert boundary_records
    assert all(record.valid for record in boundary_records)
    assert all(record.invalid_reason is None for record in boundary_records)
    assert all(record.metric_value is not None for record in boundary_records)

    abstention_experiment = ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER
    persist_ineligible_transfer_cell(
        store,
        layout,
        abstention_experiment,
        DirectedPairName("ton_iot_windows10_host -> ton_iot_network"),
        DatasetId.TON_IOT_WINDOWS10_HOST,
        DatasetId.TON_IOT_NETWORK,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        1103,
        OverwritePolicy.REPLACE,
        pair_seed_ineligibility_reason=PairSeedIneligibilityReason.NO_NONTRIVIAL_TARGET_BLOCK,
    )
    abstentions = tuple(
        record
        for record in completed_experiment_metric_records(store, abstention_experiment)
        if record.metric_name is MetricId.ABSTENTION_INDICATOR
    )
    assert len(abstentions) == 1
    assert abstentions[0].valid is False
    assert abstentions[0].metric_value is None
    assert abstentions[0].invalid_reason == (
        PairSeedIneligibilityReason.NO_NONTRIVIAL_TARGET_BLOCK.value
    )
    assert abstentions[0].direction is MetricDirection.DESCRIPTIVE
    for manifest in store.all_manifests():
        assert store.resolve(manifest.artifact_id).state == ArtifactState.COMPLETED


@pytest.mark.parametrize("terminal_state", (TerminalState.TIME_LIMIT, TerminalState.RESOURCE_LIMIT))
def test_completed_cell_carries_a_solver_limit_method_outcome(
    terminal_state: TerminalState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = _request(ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK, OverwritePolicy.REPLACE)
    uncertified = QapSeparatorResult(
        correspondence=None,
        objective_value=None,
        certified=False,
        terminal_state=terminal_state,
    )
    monkeypatch.setattr(experiment_solvers, "active_config", _single_cell_benchmark_config)

    def uncertified_separator(
        problem: RobustActionProblem, action: CurriculumAction
    ) -> QapSeparatorResult:
        del problem, action
        return uncertified

    monkeypatch.setattr(
        experiment_solvers, "fixed_action_worst_correspondence_qap", uncertified_separator
    )
    with pytest.raises(QapUncertifiedError):
        uncertified.require_certified()
    execute_exact_sparse_solver_benchmark(store, layout, request)
    records = completed_experiment_metric_records(
        store, ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK
    )
    qap_records = tuple(
        record for record in records if record.method is TransferMethod.GENERIC_EXACT_QAP
    )
    assert qap_records
    assert all(record.valid for record in qap_records)
    timeouts = tuple(
        record for record in qap_records if record.metric_name is MetricId.TIMEOUT_INDICATOR
    )
    assert len(timeouts) == 1
    assert timeouts[0].metric_value == (1.0 if terminal_state is TerminalState.TIME_LIMIT else 0.0)
    assert not any(
        record.metric_name in {MetricId.ABSOLUTE_OBJECTIVE_ERROR, MetricId.RELATIVE_OBJECTIVE_ERROR}
        for record in qap_records
    )
    exact_sparse_records = tuple(
        record for record in records if record.method is TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
    )
    assert any(
        record.metric_name is MetricId.CORRESPONDENCE_CERTIFICATE_VALIDITY
        and record.valid
        and record.metric_value == 1.0
        for record in exact_sparse_records
    )
    assert EvaluationConditionName("k4-balanced") in {record.condition for record in records}
    for manifest in store.all_manifests():
        assert store.resolve(manifest.artifact_id).state == ArtifactState.COMPLETED
