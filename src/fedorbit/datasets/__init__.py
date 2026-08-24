from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.canonicalization import (
    CanonicalFeatureVector,
    CanonicalRow,
    CanonicalizationError,
    DuplicateGroupMembers,
    DuplicateGroups,
    PartitionedFeatureValues,
    canonical_row_bytes,
    deduplicate_rows,
    exact_duplicate_hash,
    normalize_value,
    partition_features,
    validate_duplicate_groups,
)
from fedorbit.datasets.common import (
    AdapterSchema,
    DatasetAdapter,
    DatasetSchemaError,
    FieldRole,
    ObservedColumnSamples,
    role_for_field,
)
from fedorbit.datasets.edge_iiotset.schema import edge_iiotset_adapter
from fedorbit.datasets.ton_iot.components import ton_iot_adapter
from fedorbit.domain.enums import DatasetId


class DatasetRegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetAdapterRegistry:
    adapters_by_dataset: Mapping[DatasetId, DatasetAdapter]

    def adapter_for(self, dataset_id: DatasetId) -> DatasetAdapter:
        adapter = self.adapters_by_dataset.get(dataset_id)
        if adapter is None:
            raise DatasetRegistryError(f"no adapter registered for dataset {dataset_id.value}")
        return adapter

    def registered_datasets(self) -> tuple[DatasetId, ...]:
        return tuple(self.adapters_by_dataset)


def adapter_for(dataset_id: DatasetId, config: FedorbitConfig) -> DatasetAdapter:
    if dataset_id == DatasetId.EDGE_IIOTSET_NETWORK:
        return edge_iiotset_adapter(config)
    if dataset_id in (
        DatasetId.TON_IOT_WINDOWS10_HOST,
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        DatasetId.TON_IOT_NETWORK,
    ):
        return ton_iot_adapter(dataset_id, config)
    raise DatasetRegistryError(f"no adapter registered for dataset {dataset_id.value}")


def registered_adapters(config: FedorbitConfig) -> DatasetAdapterRegistry:
    return DatasetAdapterRegistry(
        {dataset_id: adapter_for(dataset_id, config) for dataset_id in DatasetId}
    )


__all__ = [
    "AdapterSchema",
    "CanonicalFeatureVector",
    "CanonicalRow",
    "CanonicalizationError",
    "DatasetAdapter",
    "DatasetAdapterRegistry",
    "DatasetRegistryError",
    "DatasetSchemaError",
    "DuplicateGroupMembers",
    "DuplicateGroups",
    "FieldRole",
    "ObservedColumnSamples",
    "PartitionedFeatureValues",
    "adapter_for",
    "canonical_row_bytes",
    "deduplicate_rows",
    "exact_duplicate_hash",
    "normalize_value",
    "partition_features",
    "registered_adapters",
    "role_for_field",
    "validate_duplicate_groups",
]
