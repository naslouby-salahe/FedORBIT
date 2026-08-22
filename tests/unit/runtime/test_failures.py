from __future__ import annotations

import pytest

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import FailureCategory, TerminalState
from fedorbit.runtime.failures import (
    CertificateNotProducedError,
    ConfigurationMismatchError,
    ConflictingDuplicatesError,
    CudaRuntimeError,
    FilesystemFailureError,
    LeakageError,
    MalformedTemporarySerializationError,
    ProcessCrashError,
    RetryPolicy,
    SchemaFailureError,
    SparseMasterNonConvergenceError,
    SplitOverlapError,
    StrictResourceViolationError,
    TheoremPrimitiveMismatchError,
    classify_failure,
    failure_boundary_outcome,
    infrastructure_exhausted_outcome,
    scientific_algorithmic_failure_outcome,
    scientific_null_outcome,
    solver_limit_outcome,
    validation_failure_outcome,
)


def test_infrastructure_failures_classified() -> None:
    for error in (
        ProcessCrashError(),
        FilesystemFailureError(),
        CudaRuntimeError(),
        MalformedTemporarySerializationError(),
    ):
        classification = classify_failure(error)
        assert classification.category == FailureCategory.INFRASTRUCTURE
        assert classification.terminal_state == TerminalState.FAILED_INFRASTRUCTURE
        assert classification.retryable


def test_validation_failures_classified() -> None:
    for error in (
        LeakageError(),
        SplitOverlapError(),
        ConflictingDuplicatesError(),
        StrictResourceViolationError(),
        ConfigurationMismatchError(),
        TheoremPrimitiveMismatchError(),
        SchemaFailureError(),
    ):
        classification = classify_failure(error)
        assert classification.category == FailureCategory.VALIDATION
        assert classification.terminal_state == TerminalState.FAILED_VALIDATION
        assert not classification.retryable


def test_algorithmic_failures_classified() -> None:
    for error in (
        SparseMasterNonConvergenceError(),
        CertificateNotProducedError(),
    ):
        classification = classify_failure(error)
        assert classification.category == FailureCategory.SCIENTIFIC_ALGORITHMIC
        assert classification.terminal_state == TerminalState.FAILED_SCIENTIFIC_ALGORITHMIC
        assert not classification.retryable


def test_unregistered_failure_rejected() -> None:
    unregistered = ValueError("unregistered")
    with pytest.raises(TypeError):
        classify_failure(unregistered)


def test_retry_exactly_twice_after_initial_attempt(
    fedorbit_config: FedorbitConfig,
) -> None:
    retries = fedorbit_config.runtime.failure_handling.retries_after_initial_infrastructure_failure
    assert retries == 2
    policy = RetryPolicy(retries)

    first = policy.decide(0, classify_failure(ProcessCrashError()))
    assert first.retry
    assert first.remaining_attempts == 2

    second = policy.decide(1, classify_failure(ProcessCrashError()))
    assert second.retry
    assert second.remaining_attempts == 1

    third = policy.decide(2, classify_failure(ProcessCrashError()))
    assert not third.retry
    assert third.terminal_state == TerminalState.FAILED_INFRASTRUCTURE


def test_validation_failure_is_not_retried() -> None:
    policy = RetryPolicy(2)
    decision = policy.decide(0, classify_failure(LeakageError()))
    assert not decision.retry
    assert decision.terminal_state == TerminalState.FAILED_VALIDATION


def test_algorithmic_failure_is_not_retried() -> None:
    policy = RetryPolicy(2)
    classification = classify_failure(SparseMasterNonConvergenceError())
    decision = policy.decide(0, classification)
    assert not decision.retry
    assert decision.terminal_state == TerminalState.FAILED_SCIENTIFIC_ALGORITHMIC


def test_negative_retry_count_rejected() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(-1)


def test_infrastructure_exhaustion_blocks_dependents() -> None:
    outcome = infrastructure_exhausted_outcome()
    assert outcome.terminal_state == TerminalState.FAILED_INFRASTRUCTURE
    assert outcome.failure_category == FailureCategory.INFRASTRUCTURE
    assert outcome.dependent_cells_blocked


def test_validation_invalid_vs_failed() -> None:
    invalid = validation_failure_outcome("leakage detected", invalid=True)
    assert invalid.terminal_state == TerminalState.INVALID
    assert invalid.failure_category == FailureCategory.VALIDATION
    failed = validation_failure_outcome("schema mismatch in implementation", invalid=False)
    assert failed.terminal_state == TerminalState.FAILED_VALIDATION


def test_scientific_null_remains_completed() -> None:
    outcome = scientific_null_outcome()
    assert outcome.terminal_state == TerminalState.COMPLETED
    assert outcome.failure_category == FailureCategory.SCIENTIFIC_NULL


def test_failure_boundary_remains_completed() -> None:
    outcome = failure_boundary_outcome()
    assert outcome.terminal_state == TerminalState.COMPLETED
    assert outcome.failure_category == FailureCategory.SCIENTIFIC_BOUNDARY


def test_solver_limit_is_completed_with_method_outcome() -> None:
    outcome = solver_limit_outcome(time_limit=True, resource_limit=False)
    assert outcome.terminal_state == TerminalState.COMPLETED
    assert outcome.method_outcome.time_limit
    assert not outcome.method_outcome.resource_limit
    resource = solver_limit_outcome(time_limit=False, resource_limit=True)
    assert resource.method_outcome.resource_limit


def test_algorithmic_failure_contract() -> None:
    outcome = scientific_algorithmic_failure_outcome(
        reason="sparse master non-convergence",
        diagnostics=True,
        completed_support_records=("support-1", "support-2"),
        cut_master_counters=(("scenario_cuts", 7), ("master_iterations", 9)),
    )
    assert outcome.terminal_state == TerminalState.FAILED_SCIENTIFIC_ALGORITHMIC
    assert outcome.failure_reason == "sparse master non-convergence"
    assert outcome.diagnostics_produced
    assert outcome.completed_support_records == ("support-1", "support-2")
    assert ("scenario_cuts", 7) in outcome.cut_master_counters
    assert not outcome.certified_action_available
    assert not outcome.proceeded_to_confirmation
    assert not outcome.proceeded_to_assimilation
    assert not outcome.proceeded_to_test_scoring
