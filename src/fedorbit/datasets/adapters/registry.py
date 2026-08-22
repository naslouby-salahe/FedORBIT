from __future__ import annotations

from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.adapters.contract import (
    DatasetAdapter,
    edge_iiotset_adapter,
    ton_iot_adapter,
)
from fedorbit.domain.enums import DatasetId


class AdapterError(ValueError):
    pass


def adapter_for(dataset_id: DatasetId, config: FedorbitConfig) -> DatasetAdapter:
    if dataset_id == DatasetId.EDGE_IIOTSET_NETWORK:
        return edge_iiotset_adapter(config)
    if dataset_id in (
        DatasetId.TON_IOT_WINDOWS10_HOST,
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        DatasetId.TON_IOT_NETWORK,
    ):
        return ton_iot_adapter(dataset_id, config)
    raise AdapterError(f"no adapter registered for dataset {dataset_id.value}")


def registered_adapters(config: FedorbitConfig) -> dict[DatasetId, DatasetAdapter]:
    return {dataset_id: adapter_for(dataset_id, config) for dataset_id in DatasetId}
