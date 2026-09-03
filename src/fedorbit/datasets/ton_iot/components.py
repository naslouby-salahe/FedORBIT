from __future__ import annotations

from dataclasses import dataclass

from fedorbit.config.loading import active_config
from fedorbit.datasets.common import AdapterContract, DatasetAdapter
from fedorbit.types import DatasetId


@dataclass(frozen=True, slots=True)
class TonIotComponent:
    dataset_id: DatasetId
    component_name: str
    relative_path: str


TON_COMPONENTS = (
    TonIotComponent(
        DatasetId.TON_IOT_WINDOWS10_HOST,
        "windows10_host",
        "Train_Test_datasets/Train_Test_Windows_dataset/Train_Test_Windows_10.csv",
    ),
    TonIotComponent(
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        "linux_process",
        "Train_Test_datasets/Train_Test_Linux_dataset/Train_Test_Linux_process.csv",
    ),
    TonIotComponent(
        DatasetId.TON_IOT_NETWORK,
        "network",
        "Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv",
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
