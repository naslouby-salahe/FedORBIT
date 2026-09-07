from __future__ import annotations

from dataclasses import dataclass, field

from fedorbit.types import FailureCategory, Index, TerminalState


class InfrastructureFailureError(RuntimeError):
    pass


class ProcessCrashError(InfrastructureFailureError):
    pass


class FilesystemFailureError(InfrastructureFailureError):
    pass


class CudaRuntimeError(InfrastructureFailureError):
    pass


class MalformedTemporarySerializationError(InfrastructureFailureError):
    pass


class FedorbitValidationError(RuntimeError):
    pass


class LeakageError(FedorbitValidationError):
    pass


class SplitOverlapError(FedorbitValidationError):
    pass


class ConflictingDuplicatesError(FedorbitValidationError):
    pass


class StrictResourceViolationError(FedorbitValidationError):
    pass


class ConfigurationMismatchError(FedorbitValidationError):
    pass


class TheoremPrimitiveMismatchError(FedorbitValidationError):
    pass


class SchemaFailureError(FedorbitValidationError):
    pass


class ScientificAlgorithmicFailureError(RuntimeError):
    pass


class SparseMasterNonConvergenceError(ScientificAlgorithmicFailureError):
    pass


class CertificateNotProducedError(ScientificAlgorithmicFailureError):
    pass


@dataclass(frozen=True, slots=True)
class FailureClassification:
    category: FailureCategory
    terminal_state: TerminalState
    retryable: bool


_INFRASTRUCTURE = FailureClassification(
    FailureCategory.INFRASTRUCTURE, TerminalState.FAILED_INFRASTRUCTURE, True
)
_VALIDATION = FailureClassification(
    FailureCategory.VALIDATION, TerminalState.FAILED_VALIDATION, False
)
_ALGORITHMIC = FailureClassification(
    FailureCategory.SCIENTIFIC_ALGORITHMIC, TerminalState.FAILED_SCIENTIFIC_ALGORITHMIC, False
)

_INFRASTRUCTURE_TYPES = (
    ProcessCrashError,
    FilesystemFailureError,
    CudaRuntimeError,
    MalformedTemporarySerializationError,
)

_VALIDATION_TYPES = (
    LeakageError,
    SplitOverlapError,
    ConflictingDuplicatesError,
    StrictResourceViolationError,
    ConfigurationMismatchError,
    TheoremPrimitiveMismatchError,
    SchemaFailureError,
)

_ALGORITHMIC_TYPES = (SparseMasterNonConvergenceError, CertificateNotProducedError)


def classify_failure(error: BaseException) -> FailureClassification:
    if isinstance(error, _INFRASTRUCTURE_TYPES):
        return _INFRASTRUCTURE
    if isinstance(error, _VALIDATION_TYPES):
        return _VALIDATION
    if isinstance(error, _ALGORITHMIC_TYPES):
        return _ALGORITHMIC
    raise TypeError(f"unregistered failure type: {type(error).__name__}")


@dataclass(frozen=True, slots=True)
class RetryDecision:
    retry: bool
    remaining_attempts: int #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    terminal_state: TerminalState | None = None


class RetryPolicy:
    def __init__(self, retries_after_initial: int) -> None: #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: retries_after_initial)
        if retries_after_initial < 0: #TODO: remove this. Such checks are not done in the code. They are values we get from the configuration and are tested by unit tests
            raise ValueError("retry count must be nonnegative")
        self._maximum_total_attempts = retries_after_initial + 1

    def decide(self, attempt_index: Index, classification: FailureClassification) -> RetryDecision:
        if not classification.retryable:
            return RetryDecision(
                retry=False,
                remaining_attempts=0,
                terminal_state=classification.terminal_state,
            )
        remaining = self._maximum_total_attempts - attempt_index - 1
        if remaining <= 0:
            return RetryDecision(
                retry=False,
                remaining_attempts=0,
                terminal_state=TerminalState.FAILED_INFRASTRUCTURE,
            )
        return RetryDecision(retry=True, remaining_attempts=remaining)


@dataclass(frozen=True, slots=True)
class MethodOutcome:
    time_limit: bool = False
    resource_limit: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    terminal_state: TerminalState
    failure_category: FailureCategory | None = None
    method_outcome: MethodOutcome = field(default_factory=MethodOutcome)
    failure_reason: str | None = None #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    diagnostics_produced: bool = False
    completed_support_records: tuple[str, ...] = () #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    cut_master_counters: tuple[tuple[str, int], ...] = () #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
    certified_action_available: bool = False
    proceeded_to_confirmation: bool = False
    proceeded_to_assimilation: bool = False
    proceeded_to_test_scoring: bool = False
    dependent_cells_blocked: bool = False


def infrastructure_exhausted_outcome() -> ExecutionOutcome:
    return ExecutionOutcome(
        terminal_state=TerminalState.FAILED_INFRASTRUCTURE,
        failure_category=FailureCategory.INFRASTRUCTURE,
        dependent_cells_blocked=True,
    )


def validation_failure_outcome(reason: str, invalid: bool) -> ExecutionOutcome: #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: reason)
    return ExecutionOutcome(
        terminal_state=TerminalState.INVALID if invalid else TerminalState.FAILED_VALIDATION,
        failure_category=FailureCategory.VALIDATION,
        failure_reason=reason,
    )


def scientific_algorithmic_failure_outcome(
    reason: str, #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: reason)
    diagnostics: bool,
    completed_support_records: tuple[str, ...], #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: completed_support_records)
    cut_master_counters: tuple[tuple[str, int], ...], #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: cut_master_counters)
) -> ExecutionOutcome:
    return ExecutionOutcome(
        terminal_state=TerminalState.FAILED_SCIENTIFIC_ALGORITHMIC,
        failure_category=FailureCategory.SCIENTIFIC_ALGORITHMIC,
        failure_reason=reason,
        diagnostics_produced=diagnostics,
        completed_support_records=completed_support_records,
        cut_master_counters=cut_master_counters,
        certified_action_available=False,
        proceeded_to_confirmation=False,
        proceeded_to_assimilation=False,
        proceeded_to_test_scoring=False,
    )


def scientific_null_outcome() -> ExecutionOutcome:
    return ExecutionOutcome(
        terminal_state=TerminalState.COMPLETED,
        failure_category=FailureCategory.SCIENTIFIC_NULL,
    )


def failure_boundary_outcome() -> ExecutionOutcome:
    return ExecutionOutcome(
        terminal_state=TerminalState.COMPLETED,
        failure_category=FailureCategory.SCIENTIFIC_BOUNDARY,
    )


def solver_limit_outcome(time_limit: bool, resource_limit: bool) -> ExecutionOutcome:
    return ExecutionOutcome(
        terminal_state=TerminalState.COMPLETED,
        method_outcome=MethodOutcome(time_limit=time_limit, resource_limit=resource_limit),
    )
