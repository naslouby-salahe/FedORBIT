from __future__ import annotations

import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.datasets.common import DatasetSchemaError, FieldRole
from fedorbit.datasets.ton_iot.components import TON_COMPONENTS, component_for, ton_iot_adapter
from fedorbit.domain.enums import DatasetId


def test_ton_iot_component_registry_is_exact() -> None:
    assert tuple(component.dataset_id for component in TON_COMPONENTS) == (
        DatasetId.TON_IOT_WINDOWS10_HOST,
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        DatasetId.TON_IOT_NETWORK,
    )
    linux = component_for(DatasetId.TON_IOT_LINUX_PROCESS_HOST)
    assert linux.relative_path == (
        "Train_Test_datasets/Train_Test_Linux_dataset/Train_Test_Linux_process.csv"
    )


def test_ton_windows_adapter_resolves_timestamp_labels_and_identity_exclusion() -> None:
    config = load_fedorbit_config()
    schema = ton_iot_adapter(DatasetId.TON_IOT_WINDOWS10_HOST, config).resolve_schema(
        ("ts", "src_ip", "label", "type", "Processor_pct_User_Time"),
        1.0,
        config.scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum,
    )
    assert schema.timestamp_column == "ts"
    assert schema.multiclass_label_column == "type"
    assert schema.binary_label_column == "label"
    assert schema.role_of("src_ip") == FieldRole.FORBIDDEN_IDENTITY


def test_ton_adapter_requires_exact_label_semantics() -> None:
    config = load_fedorbit_config()
    threshold = (
        config.scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum
    )
    with pytest.raises(DatasetSchemaError):
        ton_iot_adapter(DatasetId.TON_IOT_NETWORK, config).resolve_schema(
            ("ts", "label", "attack_type"), 1.0, threshold
        )


def test_non_ton_component_lookup_fails_closed() -> None:
    with pytest.raises(ValueError):
        component_for(DatasetId.EDGE_IIOTSET_NETWORK)
