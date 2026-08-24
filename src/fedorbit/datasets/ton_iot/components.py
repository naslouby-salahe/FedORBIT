from __future__ import annotations

from dataclasses import dataclass

from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.common import AdapterContract, DatasetAdapter
from fedorbit.domain.enums import DatasetId


@dataclass(frozen=True, slots=True)
class TonIotComponent:
    dataset_id: DatasetId
    component_name: str
    required_path_tokens: tuple[str, ...]
    forbidden_path_tokens: tuple[str, ...]


TON_COMPONENTS = (
    TonIotComponent(
        DatasetId.TON_IOT_WINDOWS10_HOST,
        "windows10_host",
        ("windows", "10"),
        ("windows7", "windows_7", "win7"),
    ),
    TonIotComponent(
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        "linux_process",
        ("linux", "process"),
        ("disk", "memory"),
    ),
    TonIotComponent(
        DatasetId.TON_IOT_NETWORK,
        "network",
        ("network",),
        (),
    ),
)


def component_for(dataset_id: DatasetId) -> TonIotComponent:
    for component in TON_COMPONENTS:
        if component.dataset_id == dataset_id:
            return component
    raise ValueError(f"dataset is not a ToN-IoT client: {dataset_id.value}")


def ton_iot_adapter(dataset_id: DatasetId, config: FedorbitConfig) -> DatasetAdapter:
    component_for(dataset_id)
    expected_timestamp = config.scientific.datasets.clients[dataset_id].expected_timestamp_field
    return DatasetAdapter(
        AdapterContract(
            dataset_id,
            (expected_timestamp,),
            ("type",),
            ("label",),
        )
    )
