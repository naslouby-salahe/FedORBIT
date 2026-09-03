from __future__ import annotations

import pytest

from fedorbit.interface import (
    AccessLogger,
    ResourceKind,
    StrictResourceViolationError,
    validate_rfc3339_utc,
)
from fedorbit.types import ClientRole


def test_test_access_fails_closed_before_finalization() -> None:
    logger = AccessLogger()
    with pytest.raises(StrictResourceViolationError):
        logger.record(ClientRole.TARGET, ResourceKind.TEST, transfer_finalized=False)
    logger.record(ClientRole.TARGET, ResourceKind.TEST, transfer_finalized=True)
    logger.validate()


def test_source_cannot_access_target_confirmation_or_test() -> None:
    logger = AccessLogger()
    with pytest.raises(StrictResourceViolationError):
        logger.record(ClientRole.SOURCE, ResourceKind.CONFIRM)
    with pytest.raises(StrictResourceViolationError):
        logger.record(ClientRole.SOURCE, ResourceKind.TEST, transfer_finalized=True)


def test_timestamp_requires_rfc3339_utc() -> None:
    validate_rfc3339_utc("2026-08-23T21:00:00Z")
    with pytest.raises(StrictResourceViolationError):
        validate_rfc3339_utc("2026-08-23T21:00:00+01:00")
    with pytest.raises(StrictResourceViolationError):
        validate_rfc3339_utc("not-a-time")
