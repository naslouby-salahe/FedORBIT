from __future__ import annotations

import math
import unicodedata

import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.datasets.canonicalization import (
    CanonicalFeatureVector,
    CanonicalizationError,
    CanonicalRow,
    canonical_row_bytes,
    deduplicate_rows,
    exact_duplicate_hash,
    normalize_value,
    validate_duplicate_groups,
)
from fedorbit.datasets.common import ObservedColumnSamples
from fedorbit.datasets.edge_iiotset.schema import edge_iiotset_adapter

EDGE_COLUMNS = (
    "frame.time",
    "ip.src_host",
    "service_state",
    "tcp.ack",
    "http.file_data",
    "Attack_label",
    "Attack_type",
)


def _schema():
    config = load_fedorbit_config()
    return edge_iiotset_adapter(config).resolve_schema(
        EDGE_COLUMNS,
        1.0,
        config.scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum,
        ObservedColumnSamples(
            {
                "tcp.ack": ("1", "2", "3.5"),
                "service_state": ("OPEN", "CLOSED"),
            }
        ),
    )


def _features(
    tcp_ack: float | None = 1.0,
    service_state: str = "OPEN",
    identity: str = "host-a",
) -> CanonicalFeatureVector:
    return CanonicalFeatureVector(
        {
            "frame.time": "2024-01-01T00:00:00Z",
            "ip.src_host": identity,
            "service_state": service_state,
            "tcp.ack": tcp_ack,
            "http.file_data": "payload",
            "Attack_label": 1,
            "Attack_type": "ddos",
        }
    )


def test_numeric_float64_serialization_is_little_endian() -> None:
    payload = canonical_row_bytes(_features(tcp_ack=3.5), _schema())
    assert b"\x00\x00\x00\x00\x00\x00\x0c@" in payload


def test_missing_numeric_uses_canonical_quiet_nan_and_zero_remains_observed() -> None:
    missing = canonical_row_bytes(_features(tcp_ack=None), _schema())
    zero = canonical_row_bytes(_features(tcp_ack=0.0), _schema())
    assert b"\x00\x00\x00\x00\x00\x00\xf8\x7f" in missing
    assert zero != missing


def test_behavioral_categorical_strings_are_nfc_normalized() -> None:
    decomposed = unicodedata.normalize("NFD", "é")
    composed = unicodedata.normalize("NFC", "é")
    first = canonical_row_bytes(_features(service_state=decomposed), _schema())
    second = canonical_row_bytes(_features(service_state=composed), _schema())
    assert first == second


def test_duplicate_hash_ignores_forbidden_identity_fields() -> None:
    schema = _schema()
    assert exact_duplicate_hash(_features(identity="host-a"), schema) == exact_duplicate_hash(
        _features(identity="host-b"), schema
    )


def test_exact_duplicate_grouping_rejects_conflicting_labels() -> None:
    schema = _schema()
    rows = (
        CanonicalRow(_features(), "ddos", 0.1, ""),
        CanonicalRow(_features(), "normal", 0.2, ""),
    )
    groups = deduplicate_rows(schema, rows)
    assert groups.group_count == 1
    with pytest.raises(CanonicalizationError):
        validate_duplicate_groups(groups)


def test_normalize_value_applies_missing_vocabulary_without_erasing_numeric_zero() -> None:
    assert normalize_value("0", is_categorical=True) == ""
    assert normalize_value("0.0", is_categorical=True) == ""
    numeric_zero = normalize_value(0.0, is_categorical=False)
    assert numeric_zero == 0.0
    numeric_missing = normalize_value("nan", is_categorical=False)
    assert isinstance(numeric_missing, float)
    assert math.isnan(float(numeric_missing))
