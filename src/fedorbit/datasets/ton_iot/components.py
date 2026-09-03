from __future__ import annotations

from dataclasses import dataclass

from fedorbit.config.loading import active_config
from fedorbit.datasets.common import AdapterContract, DatasetAdapter
from fedorbit.types import DatasetId


@dataclass(frozen=True, slots=True)
class TonIotComponent:
    dataset_id: DatasetId
    component_name: str
    relative_paths: tuple[str, ...]


TON_COMPONENTS = (
    TonIotComponent(
        DatasetId.TON_IOT_WINDOWS10_HOST,
        "windows10_host",
        ("Processed_datasets/Processed_Windows_dataset/windows10_dataset.csv",),
    ),
    TonIotComponent(
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        "linux_process",
        (
            "Processed_datasets/Processed_Linux_dataset/Linux_process_1.csv",
            "Processed_datasets/Processed_Linux_dataset/Linux_process_2.csv",
        ),
    ),
    TonIotComponent(
        DatasetId.TON_IOT_NETWORK,
        "network",
        tuple(
            f"Processed_datasets/Processed_Network_dataset/Network_dataset_{index}.csv"
            for index in range(1, 24)
        ),
    ),
)


def component_for(dataset_id: DatasetId) -> TonIotComponent:
    for component in TON_COMPONENTS:
        if component.dataset_id == dataset_id:
            return component
    raise ValueError(f"dataset is not a ToN-IoT client: {dataset_id.value}")


def ton_iot_adapter(dataset_id: DatasetId) -> DatasetAdapter:
    component_for(dataset_id)
    config = active_config()
    expected_timestamp = config.scientific.datasets.clients[dataset_id].expected_timestamp_field
    return DatasetAdapter(
        AdapterContract(
            dataset_id,
            (expected_timestamp,),
            ("type",),
            ("label",),
        )
    )
