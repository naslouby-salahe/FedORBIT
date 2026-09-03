from __future__ import annotations

from fedorbit.infrastructure.failures import LeakageError, classify_failure
from fedorbit.types import FailureCategory


def test_failure_classification_preserves_validation_boundary() -> None:
    assert classify_failure(LeakageError()).category == FailureCategory.VALIDATION
