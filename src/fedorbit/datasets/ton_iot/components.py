from __future__ import annotations

from dataclasses import dataclass

from fedorbit.config.loading import active_config
from fedorbit.datasets.common import AdapterContract, DatasetAdapter
from fedorbit.types import ClientComponentName, DatasetId, DatasetRelativePath, TabularColumnName


@dataclass(frozen=True, slots=True)
class TonIotComponent:
    dataset_id: DatasetId
    component_name: ClientComponentName
    relative_paths: tuple[DatasetRelativePath, ...]


def ton_iot_components() -> tuple[TonIotComponent, ...]:
    return tuple(
        TonIotComponent(dataset_id, component.component_name, component.relative_paths)
        for dataset_id, component in active_config().scientific.datasets.ton_iot_components.items()
    )


def component_for(dataset_id: DatasetId) -> TonIotComponent:
    component = active_config().scientific.datasets.ton_iot_components.get(dataset_id)
    if component is None:
        raise ValueError(f"dataset is not a ToN-IoT client: {dataset_id.value}")
    return TonIotComponent(dataset_id, component.component_name, component.relative_paths)


def ton_iot_adapter(dataset_id: DatasetId) -> DatasetAdapter:
    component_for(dataset_id)
    config = active_config()
    expected_timestamp = config.scientific.datasets.clients[dataset_id].expected_timestamp_field
    return DatasetAdapter(
        AdapterContract(
            dataset_id,
            (TabularColumnName(expected_timestamp),),
            (TabularColumnName("type"),),
            (TabularColumnName("label"),),
        )
    )
