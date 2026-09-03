from __future__ import annotations

import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.datasets.ton_iot.components import component_for, ton_iot_adapter
from fedorbit.datasets.ton_iot.validation import (
    TonIotLabelObservation,
    TonIotValidationError,
    validate_ton_iot_label_consistency,
    validate_ton_iot_schema,
)
from fedorbit.types import DatasetId


def test_ton_schema_validation_matches_selected_component() -> None:
    config = load_fedorbit_config()
    component = component_for(DatasetId.TON_IOT_NETWORK)
    schema = ton_iot_adapter(component.dataset_id).resolve_schema(
        ("ts", "src_ip", "label", "type", "duration"),
        1.0,
        config.scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum,
    )
    validate_ton_iot_schema(schema, component)


def test_ton_schema_validation_rejects_component_identity_mismatch() -> None:
    config = load_fedorbit_config()
    network = component_for(DatasetId.TON_IOT_NETWORK)
    windows = component_for(DatasetId.TON_IOT_WINDOWS10_HOST)
    schema = ton_iot_adapter(network.dataset_id).resolve_schema(
        ("ts", "label", "type"),
        1.0,
        config.scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum,
    )
    with pytest.raises(TonIotValidationError):
        validate_ton_iot_schema(schema, windows)


def test_ton_binary_and_multiclass_labels_must_describe_same_partition() -> None:
    validate_ton_iot_label_consistency(
        (TonIotLabelObservation("normal", 0), TonIotLabelObservation("ddos", 1))
    )
    with pytest.raises(TonIotValidationError):
        validate_ton_iot_label_consistency((TonIotLabelObservation("normal", 1),))
