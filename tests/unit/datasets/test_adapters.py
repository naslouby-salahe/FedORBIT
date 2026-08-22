from __future__ import annotations

import pytest

from fedorbit.datasets.adapters import (
    BEHAVIORAL_CATEGORICAL_ROLE,
    BEHAVIORAL_NUMERIC_ROLE,
    BINARY_LABEL_ROLE,
    EDGE_EXCLUSIONS,
    EDGE_LEAKAGE_SAFEGUARD_EXCLUSIONS,
    FORBIDDEN_IDENTITY_ROLE,
    FORBIDDEN_PAYLOAD_ROLE,
    FORBIDDEN_PROVENANCE_ROLE,
    MULTICLASS_LABEL_ROLE,
    TIMESTAMP_ROLE,
    SchemaError,
    edge_iiotset_adapter,
    role_for_field,
    ton_iot_adapter,
)
from fedorbit.datasets.adapters.registry import adapter_for, registered_adapters
from fedorbit.datasets.ontology import (
    TRANSFER_ONTOLOGY,
    canonicalize_label,
    coarse_group_for,
    native_labels_for,
    transfer_concept_for,
)
from fedorbit.domain.enums import CoarseGroup, DatasetId


def test_label_canonicalization_steps() -> None:
    assert canonicalize_label("DDoS TCP") == "ddos_tcp"
    assert canonicalize_label(" SQL/Injection ") == "sql_injection"
    assert canonicalize_label("Normal") == "normal"
    assert canonicalize_label("Port Scanning") == "port_scanning"
    assert canonicalize_label("XSS") == "xss"
    assert canonicalize_label("MITM") == "mitm"
    assert canonicalize_label("DDoS_UDP") == "ddos_udp"


def test_transfer_ontology_exact() -> None:
    assert set(TRANSFER_ONTOLOGY) == {
        "DDoS",
        "Ransomware",
        "Backdoor",
        "Injection",
        "XSS",
        "Password attack",
        "Scanning",
        "MITM",
    }
    assert TRANSFER_ONTOLOGY["DDoS"] == (
        CoarseGroup.DISRUPTION,
        ("ddos_udp", "ddos_icmp", "ddos_tcp", "ddos_http"),
        ("ddos",),
    )
    assert TRANSFER_ONTOLOGY["Scanning"][0] == CoarseGroup.ACCESS_AND_DISCOVERY
    assert TRANSFER_ONTOLOGY["Injection"][0] == CoarseGroup.EXPLOITATION


def test_transfer_concept_lookup_per_dataset() -> None:
    assert transfer_concept_for(DatasetId.EDGE_IIOTSET_NETWORK, "ddos_tcp") == "DDoS"
    assert transfer_concept_for(DatasetId.TON_IOT_NETWORK, "ddos") == "DDoS"
    assert transfer_concept_for(DatasetId.EDGE_IIOTSET_NETWORK, "sql_injection") == "Injection"
    assert transfer_concept_for(DatasetId.TON_IOT_NETWORK, "injection") == "Injection"
    assert transfer_concept_for(DatasetId.EDGE_IIOTSET_NETWORK, "uploading") is None
    assert transfer_concept_for(DatasetId.TON_IOT_NETWORK, "dos") is None


def test_coarse_group_mapping() -> None:
    assert coarse_group_for(DatasetId.EDGE_IIOTSET_NETWORK, "ransomware") == CoarseGroup.DISRUPTION
    assert coarse_group_for(DatasetId.EDGE_IIOTSET_NETWORK, "backdoor") == CoarseGroup.EXPLOITATION
    assert (
        coarse_group_for(DatasetId.TON_IOT_NETWORK, "scanning") == CoarseGroup.ACCESS_AND_DISCOVERY
    )


def test_native_labels_include_eligible_local_classes() -> None:
    edge_labels = native_labels_for(DatasetId.EDGE_IIOTSET_NETWORK)
    assert "uploading" in edge_labels
    assert "ddos_tcp" in edge_labels
    ton_labels = native_labels_for(DatasetId.TON_IOT_NETWORK)
    assert "dos" in ton_labels
    assert "ddos" in ton_labels
    assert "uploading" not in ton_labels


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


def test_edge_adapter_resolves_real_schema() -> None:
    adapter = edge_iiotset_adapter()
    schema = adapter.resolve_schema(
        observed_columns=EDGE_COLUMNS,
        timestamp_parse_success_fraction=1.0,
        timestamp_alias_minimum=0.999,
        observed_value_samples={
            "tcp.ack": ("1", "2", "3.5"),
            "http.request.method": ("GET", "POST", "PUT"),
        },
    )
    assert schema.role_of("tcp.ack") == BEHAVIORAL_NUMERIC_ROLE
    assert schema.timestamp_column == "frame.time"
    assert schema.multiclass_label_column == "Attack_type"
    assert schema.binary_label_column == "Attack_label"
    assert schema.role_of("frame.time") == TIMESTAMP_ROLE
    assert schema.role_of("Attack_type") == MULTICLASS_LABEL_ROLE
    assert schema.role_of("Attack_label") == BINARY_LABEL_ROLE
    assert schema.role_of("http.request.method") == FORBIDDEN_PROVENANCE_ROLE
    assert schema.role_of("ip.src_host") == FORBIDDEN_IDENTITY_ROLE
    assert schema.role_of("mqtt.msg") == FORBIDDEN_PAYLOAD_ROLE


def test_edge_exclusion_sets_exact() -> None:
    assert "frame.time" in EDGE_EXCLUSIONS
    assert "tcp.srcport" in EDGE_EXCLUSIONS
    assert "udp.port" in EDGE_EXCLUSIONS
    assert "http.request.method" in EDGE_LEAKAGE_SAFEGUARD_EXCLUSIONS
    assert "mqtt.topic" in EDGE_LEAKAGE_SAFEGUARD_EXCLUSIONS
    assert len(EDGE_LEAKAGE_SAFEGUARD_EXCLUSIONS) == 7


def test_ton_adapter_resolves_real_schema() -> None:
    adapter = ton_iot_adapter(DatasetId.TON_IOT_WINDOWS10_HOST)
    schema = adapter.resolve_schema(
        observed_columns=("ts", "src_ip", "label", "type", "Processor_pct_User_Time"),
        timestamp_parse_success_fraction=1.0,
        timestamp_alias_minimum=0.999,
    )
    assert schema.timestamp_column == "ts"
    assert schema.multiclass_label_column == "type"
    assert schema.binary_label_column == "label"
    assert schema.role_of("src_ip") == FORBIDDEN_IDENTITY_ROLE


def test_timestamp_alias_requires_parse_success() -> None:
    adapter = edge_iiotset_adapter()
    with pytest.raises(SchemaError):
        adapter.resolve_schema(
            observed_columns=EDGE_COLUMNS,
            timestamp_parse_success_fraction=0.5,
            timestamp_alias_minimum=0.999,
        )


def test_ambiguous_timestamp_rejected() -> None:
    from fedorbit.datasets.adapters.schema import exactly_one_candidate

    with pytest.raises(SchemaError):
        exactly_one_candidate(("ts", "ts"), ("ts",), "timestamp")
    with pytest.raises(SchemaError):
        exactly_one_candidate(("ts", "ts_alias"), ("ts", "ts_alias"), "timestamp")
    with pytest.raises(SchemaError):
        exactly_one_candidate(("label", "type"), ("ts",), "timestamp")


def test_missing_label_field_rejected() -> None:
    adapter = ton_iot_adapter(DatasetId.TON_IOT_NETWORK)
    with pytest.raises(SchemaError):
        adapter.resolve_schema(
            observed_columns=("ts", "label", "type_missing"),
            timestamp_parse_success_fraction=1.0,
            timestamp_alias_minimum=0.999,
        )


def test_registry_covers_all_clients() -> None:
    adapters = registered_adapters()
    assert set(adapters) == set(DatasetId)
    assert adapter_for(DatasetId.EDGE_IIOTSET_NETWORK).dataset_id == DatasetId.EDGE_IIOTSET_NETWORK
    assert adapter_for(DatasetId.TON_IOT_NETWORK).dataset_id == DatasetId.TON_IOT_NETWORK


def test_role_for_field_markers() -> None:
    assert role_for_field("src_ip") == FORBIDDEN_IDENTITY_ROLE
    assert role_for_field("http.file_data") == FORBIDDEN_PAYLOAD_ROLE
    assert role_for_field("source_file_name") == FORBIDDEN_PROVENANCE_ROLE
    assert role_for_field("duration") == BEHAVIORAL_CATEGORICAL_ROLE
