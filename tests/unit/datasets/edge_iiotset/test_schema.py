from __future__ import annotations

import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.datasets.common import DatasetSchemaError, FieldRole, ObservedColumnSamples
from fedorbit.datasets.edge_iiotset.schema import (
    EDGE_EXCLUSIONS,
    EDGE_LEAKAGE_SAFEGUARD_EXCLUSIONS,
    edge_iiotset_adapter,
)

EDGE_COLUMNS = (
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "http.request.method",
    "tcp.ack",
    "mqtt.msg",
    "Attack_label",
    "Attack_type",
)


def test_edge_schema_resolves_timestamp_labels_and_feature_roles() -> None:
    config = load_fedorbit_config()
    schema = edge_iiotset_adapter(config).resolve_schema(
        EDGE_COLUMNS,
        1.0,
        config.scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum,
        ObservedColumnSamples({"tcp.ack": ("1", "2", "3.5")}),
    )
    assert schema.timestamp_column == "frame.time"
    assert schema.multiclass_label_column == "Attack_type"
    assert schema.binary_label_column == "Attack_label"
    assert schema.role_of("frame.time") == FieldRole.TIMESTAMP
    assert schema.role_of("tcp.ack") == FieldRole.BEHAVIORAL_NUMERIC
    assert schema.role_of("ip.src_host") == FieldRole.FORBIDDEN_IDENTITY
    assert schema.role_of("mqtt.msg") == FieldRole.FORBIDDEN_PAYLOAD
    assert schema.role_of("http.request.method") == FieldRole.FORBIDDEN_PROVENANCE


def test_edge_exclusion_contract_contains_all_registered_safeguards() -> None:
    assert {
        "frame.time",
        "ip.src_host",
        "ip.dst_host",
        "tcp.srcport",
        "tcp.dstport",
        "udp.port",
        "mqtt.msg",
    } <= EDGE_EXCLUSIONS
    assert EDGE_LEAKAGE_SAFEGUARD_EXCLUSIONS == frozenset(
        {
            "http.request.method",
            "http.referer",
            "http.request.version",
            "dns.qry.name.len",
            "mqtt.conack.flags",
            "mqtt.protoname",
            "mqtt.topic",
        }
    )


def test_edge_timestamp_parse_failure_is_data_invalid() -> None:
    config = load_fedorbit_config()
    with pytest.raises(DatasetSchemaError):
        edge_iiotset_adapter(config).resolve_schema(
            EDGE_COLUMNS,
            0.5,
            config.scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum,
        )


def test_duplicate_or_missing_semantic_columns_fail_closed() -> None:
    config = load_fedorbit_config()
    adapter = edge_iiotset_adapter(config)
    threshold = (
        config.scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum
    )
    with pytest.raises(DatasetSchemaError):
        adapter.resolve_schema(EDGE_COLUMNS + ("frame.time",), 1.0, threshold)
    with pytest.raises(DatasetSchemaError):
        adapter.resolve_schema(tuple(c for c in EDGE_COLUMNS if c != "Attack_type"), 1.0, threshold)
