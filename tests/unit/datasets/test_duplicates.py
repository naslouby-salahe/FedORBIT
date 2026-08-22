from __future__ import annotations

import math

import pytest

from fedorbit.datasets.adapters import (
    FORBIDDEN_IDENTITY_ROLE,
    FORBIDDEN_PAYLOAD_ROLE,
    FORBIDDEN_PROVENANCE_ROLE,
    AdapterSchema,
    edge_iiotset_adapter,
)
from fedorbit.datasets.duplicates import (
    CanonicalRow,
    DuplicateError,
    canonical_row_bytes,
    deduplicate_rows,
    exact_duplicate_hash,
    normalize_value,
    validate_duplicate_groups,
)

EDGE_COLUMNS = (
    "frame.time",
    "ip.src_host",
    "http.request.method",
    "tcp.ack",
    "http.file_data",
    "Attack_label",
    "Attack_type",
)


def _edge_schema() -> AdapterSchema:
    return edge_iiotset_adapter().resolve_schema(
        observed_columns=EDGE_COLUMNS,
        timestamp_parse_success_fraction=1.0,
        timestamp_alias_minimum=0.999,
        observed_value_samples={
            "tcp.ack": ("1", "2", "3.5"),
        },
    )


def test_forbidden_fields_excluded_before_hashing() -> None:
    schema = _edge_schema()
    assert schema.role_of("ip.src_host") == FORBIDDEN_IDENTITY_ROLE
    assert schema.role_of("http.file_data") == FORBIDDEN_PAYLOAD_ROLE
    assert schema.role_of("http.request.method") == FORBIDDEN_PROVENANCE_ROLE
    behavioral = schema.behavioral_features()
    assert "ip.src_host" not in behavioral
    assert "http.request.method" not in behavioral
    assert "http.file_data" not in behavioral
    assert "tcp.ack" in behavioral


def test_canonical_bytes_numeric_float64_little_endian() -> None:
    schema = _edge_schema()
    features: dict[str, str | int | float | None] = {
        "frame.time": "0",
        "ip.src_host": "x",
        "http.request.method": "GET",
        "tcp.ack": 3.5,
        "http.file_data": "data",
        "Attack_label": 1,
        "Attack_type": "ddos",
    }
    payload = canonical_row_bytes(features, schema.canonical_feature_order, schema.roles)
    assert b"\x00\x00\x00\x00\x00\x00\x0c@" in payload
    assert len(payload) >= 8


def test_missing_numeric_is_canonical_quiet_nan() -> None:
    schema = _edge_schema()
    missing_nan: dict[str, str | int | float | None] = {
        "frame.time": "0",
        "ip.src_host": "x",
        "http.request.method": "GET",
        "tcp.ack": float("nan"),
        "http.file_data": "d",
        "Attack_label": 1,
        "Attack_type": "ddos",
    }
    missing_none: dict[str, str | int | float | None] = {
        "frame.time": "0",
        "ip.src_host": "x",
        "http.request.method": "GET",
        "tcp.ack": None,
        "http.file_data": "d",
        "Attack_label": 1,
        "Attack_type": "ddos",
    }
    first = canonical_row_bytes(missing_nan, schema.canonical_feature_order, schema.roles)
    second = canonical_row_bytes(missing_none, schema.canonical_feature_order, schema.roles)
    assert first == second
    assert b"\x00\x00\x00\x00\x00\x00\xf8\x7f" in first
    assert len(first) >= 9


def test_numeric_zero_is_distinct_from_missing() -> None:
    schema = _edge_schema()
    zero_features: dict[str, str | int | float | None] = {
        "frame.time": "0",
        "ip.src_host": "x",
        "http.request.method": "GET",
        "tcp.ack": 0.0,
        "http.file_data": "d",
        "Attack_label": 1,
        "Attack_type": "ddos",
    }
    missing_features: dict[str, str | int | float | None] = {
        "frame.time": "0",
        "ip.src_host": "x",
        "http.request.method": "GET",
        "tcp.ack": None,
        "http.file_data": "d",
        "Attack_label": 1,
        "Attack_type": "ddos",
    }
    zero = canonical_row_bytes(zero_features, schema.canonical_feature_order, schema.roles)
    missing = canonical_row_bytes(missing_features, schema.canonical_feature_order, schema.roles)
    assert zero != missing


def test_categorical_nfc_normalization_before_hashing() -> None:
    schema = _edge_schema()
    assert schema.role_of("http.request.method") == FORBIDDEN_PROVENANCE_ROLE
    nfc_features: dict[str, str | int | float | None] = {
        "frame.time": "0",
        "ip.src_host": "x",
        "http.request.method": "GET",
        "tcp.ack": 1.0,
        "http.file_data": "d",
        "Attack_label": 1,
        "Attack_type": "ddos",
    }
    decomposed = canonical_row_bytes(nfc_features, schema.canonical_feature_order, schema.roles)
    composed = canonical_row_bytes(nfc_features, schema.canonical_feature_order, schema.roles)
    assert decomposed == composed


def test_exact_duplicate_hash_deterministic() -> None:
    schema = _edge_schema()
    features: dict[str, str | int | float | None] = {
        "frame.time": "0",
        "ip.src_host": "x",
        "http.request.method": "GET",
        "tcp.ack": 1.0,
        "http.file_data": "d",
        "Attack_label": 1,
        "Attack_type": "ddos",
    }
    assert exact_duplicate_hash(features, schema.canonical_feature_order, schema.roles) == (
        exact_duplicate_hash(features, schema.canonical_feature_order, schema.roles)
    )
    assert len(exact_duplicate_hash(features, schema.canonical_feature_order, schema.roles)) == 64


def test_duplicate_hashing_ignores_forbidden_identity() -> None:
    schema = _edge_schema()
    base: dict[str, str | int | float | None] = {
        "frame.time": "0",
        "ip.src_host": "a",
        "http.request.method": "GET",
        "tcp.ack": 1.0,
        "http.file_data": "d",
        "Attack_label": 1,
        "Attack_type": "ddos",
    }
    different_identity: dict[str, str | int | float | None] = dict(base)
    different_identity["ip.src_host"] = "different-host"
    assert exact_duplicate_hash(base, schema.canonical_feature_order, schema.roles) == (
        exact_duplicate_hash(different_identity, schema.canonical_feature_order, schema.roles)
    )


def _row(schema: AdapterSchema, tcp_ack: float, label: str, fraction: float) -> CanonicalRow:
    features: dict[str, str | int | float | None] = {
        "frame.time": "0",
        "ip.src_host": "x",
        "http.request.method": "GET",
        "tcp.ack": tcp_ack,
        "http.file_data": "d",
        "Attack_label": 1,
        "Attack_type": label,
    }
    return CanonicalRow(
        features=features,
        label=label,
        timestamp_fraction=fraction,
        group_id="",
    )


def test_deduplicate_groups_exact_duplicates() -> None:
    schema = _edge_schema()
    rows = (
        _row(schema, 1.0, "ddos", 0.1),
        _row(schema, 1.0, "ddos", 0.2),
        _row(schema, 2.0, "normal", 0.9),
    )
    groups = deduplicate_rows(schema, rows)
    assert len(groups) == 2
    validate_duplicate_groups(groups)


def test_conflicting_duplicates_invalid_data() -> None:
    schema = _edge_schema()
    rows = (_row(schema, 1.0, "ddos", 0.1), _row(schema, 1.0, "normal", 0.1))
    groups = deduplicate_rows(schema, rows)
    with pytest.raises(DuplicateError):
        validate_duplicate_groups(groups)


def test_normalize_value_missing_spelling_canonicalization() -> None:
    assert normalize_value("0", is_categorical=True) == ""
    assert normalize_value("0.0", is_categorical=True) == ""
    assert normalize_value("1", is_categorical=True) == "1"
    numeric_missing = normalize_value("NAN", is_categorical=False)
    assert isinstance(numeric_missing, float)
    assert math.isnan(numeric_missing)
