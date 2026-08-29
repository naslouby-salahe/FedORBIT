from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from fedorbit.datasets.edge_iiotset.loader import inspect_edge_tabular_files
from fedorbit.datasets.ton_iot.components import component_for
from fedorbit.datasets.ton_iot.loader import inspect_ton_iot_component_files
from fedorbit.domain.enums import DatasetId
from fedorbit.domain.serialization import StableJsonPayload, stable_json


class RawInventoryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RawInventoryRequest:
    dataset: DatasetId
    raw_root: Path


@dataclass(frozen=True, slots=True)
class RawFileInventory:
    relative_path: str
    byte_size: int
    sha256: str
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RawDatasetInventory:
    dataset: DatasetId
    files: tuple[RawFileInventory, ...]

    def __post_init__(self) -> None:
        if not self.files:
            raise RawInventoryError("raw dataset inventory requires at least one file")

    def fingerprint(self) -> str:
        return hashlib.sha256(stable_json(self.serialization_payload()).encode("utf-8")).hexdigest()

    def serialization_payload(self) -> StableJsonPayload:
        file_entries: list[StableJsonPayload] = []
        for file in self.files:
            file_entries.append(
                cast(
                    StableJsonPayload,
                    OrderedDict[str, StableJsonPayload](
                        relative_path=file.relative_path,
                        byte_size=file.byte_size,
                        sha256=file.sha256,
                        columns=list(file.columns),
                    ),
                )
            )
        return cast(
            StableJsonPayload,
            OrderedDict[str, StableJsonPayload](
                dataset=self.dataset.value,
                files=file_entries,
            ),
        )


def inspect_raw_inventory(request: RawInventoryRequest) -> RawDatasetInventory:
    if request.dataset == DatasetId.EDGE_IIOTSET_NETWORK:
        inspected = inspect_edge_tabular_files(request.raw_root)
    else:
        inspected = inspect_ton_iot_component_files(
            request.raw_root, component_for(request.dataset)
        )
    return RawDatasetInventory(
        dataset=request.dataset,
        files=tuple(
            RawFileInventory(
                relative_path=entry.relative_path,
                byte_size=entry.byte_size,
                sha256=entry.sha256,
                columns=entry.columns,
            )
            for entry in inspected
        ),
    )
