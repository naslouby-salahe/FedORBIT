from __future__ import annotations

import pytest

from fedorbit.interface import (
    AccessEvent,
    AccessTrace,
    ResourceKind,
    StrictResourceViolationError,
    static_leakage_scan,
    validate_disjoint_feature_namespaces,
    validate_dynamic_access_log_scan,
    validate_no_cross_client_entity_ids,
    validate_no_cross_client_timestamp_pairing,
    validate_oracle_acl_isolation,
    validate_resource_manifest_equality,
    validate_static_leakage_scan,
)
from fedorbit.types import ClientRole


def test_disjoint_feature_namespaces() -> None:
    validate_disjoint_feature_namespaces(frozenset({"a", "b"}), frozenset({"c", "d"}))
    with pytest.raises(StrictResourceViolationError):
        validate_disjoint_feature_namespaces(frozenset({"a", "b"}), frozenset({"b", "c"}))


def test_no_cross_client_entity_ids() -> None:
    validate_no_cross_client_entity_ids(frozenset({"e1"}), frozenset({"e2"}))
    with pytest.raises(StrictResourceViolationError):
        validate_no_cross_client_entity_ids(frozenset({"e1"}), frozenset({"e1"}))


def test_no_cross_client_timestamp_pairing() -> None:
    validate_no_cross_client_timestamp_pairing(
        frozenset({"2024-01-01T00:00:00Z"}), frozenset({"2024-01-02T00:00:00Z"})
    )
    with pytest.raises(StrictResourceViolationError):
        validate_no_cross_client_timestamp_pairing(
            frozenset({"2024-01-01T00:00:00Z"}), frozenset({"2024-01-01T00:00:00Z"})
        )


def test_oracle_acl_isolation() -> None:
    validate_oracle_acl_isolation(
        cell_is_oracle_validation_context=True, oracle_information_accessed=True
    )
    validate_oracle_acl_isolation(
        cell_is_oracle_validation_context=False, oracle_information_accessed=False
    )
    with pytest.raises(StrictResourceViolationError):
        validate_oracle_acl_isolation(
            cell_is_oracle_validation_context=False, oracle_information_accessed=True
        )


def test_resource_manifest_equality() -> None:
    validate_resource_manifest_equality(
        frozenset({ResourceKind.TRAIN, ResourceKind.META}),
        frozenset({ResourceKind.TRAIN, ResourceKind.META}),
    )
    with pytest.raises(StrictResourceViolationError):
        validate_resource_manifest_equality(
            frozenset({ResourceKind.TRAIN}),
            frozenset({ResourceKind.TRAIN, ResourceKind.META}),
        )


def test_static_leakage_scan_finds_forbidden_terms() -> None:
    payload = b'{"label": "DDoS attack detected"}'
    findings = static_leakage_scan(payload, frozenset({"ddos", "ransomware"}))
    assert findings == ("ddos",)
    validate_static_leakage_scan(b'{"label": "benign"}', frozenset({"ddos", "ransomware"}))
    with pytest.raises(StrictResourceViolationError):
        validate_static_leakage_scan(payload, frozenset({"ddos"}))


def test_dynamic_access_log_scan_reuses_the_resource_policy() -> None:
    valid_trace = AccessTrace(
        (AccessEvent(ClientRole.SOURCE, ResourceKind.TRAIN, transfer_finalized=False),)
    )
    validate_dynamic_access_log_scan(valid_trace)
    invalid_trace = AccessTrace(
        (AccessEvent(ClientRole.SOURCE, ResourceKind.CONFIRM, transfer_finalized=False),)
    )
    with pytest.raises(StrictResourceViolationError):
        validate_dynamic_access_log_scan(invalid_trace)
