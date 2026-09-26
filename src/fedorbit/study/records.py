from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from fedorbit.types import (
    ConditionName,
    DatasetId,
    DeviceName,
    ExperimentId,
    FrozenModel,
    ReplicateIndex,
    SupportSize,
)


class RecordKind(StrEnum):
    CELL = "cell"
    INFEASIBLE_SUPPORT = "infeasible-support"
    INELIGIBLE_DEVICE = "ineligible-device"


class CellRecord(FrozenModel):
    kind: Literal[RecordKind.CELL] = RecordKind.CELL
    experiment: ExperimentId
    dataset: DatasetId
    device: DeviceName
    support_size: SupportSize
    replicate: ReplicateIndex
    condition: ConditionName
    auroc: float = Field(ge=0.0, le=1.0)
    false_positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    true_positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    scale_mismatch: float = Field(ge=0.0)
    split_half_instability: float = Field(ge=0.0)


class InfeasibleSupportRecord(FrozenModel):
    kind: Literal[RecordKind.INFEASIBLE_SUPPORT] = RecordKind.INFEASIBLE_SUPPORT
    experiment: ExperimentId
    dataset: DatasetId
    device: DeviceName
    support_size: SupportSize
    available_rows: int = Field(ge=0)


class IneligibleDeviceRecord(FrozenModel):
    kind: Literal[RecordKind.INELIGIBLE_DEVICE] = RecordKind.INELIGIBLE_DEVICE
    experiment: ExperimentId
    dataset: DatasetId
    device: DeviceName
    benign_rows: int = Field(ge=0)
    attack_rows: int = Field(ge=0)
    worst_case_standard_error: float = Field(ge=0.0)


AnyRecord = CellRecord | InfeasibleSupportRecord | IneligibleDeviceRecord
EvidenceRecord = Annotated[AnyRecord, Field(discriminator="kind")]
_ADAPTER: TypeAdapter[EvidenceRecord] = TypeAdapter(EvidenceRecord)


def serialise_records(records: list[AnyRecord]) -> bytes:
    lines = (_ADAPTER.dump_json(record).decode("utf-8") for record in records)
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_records(payload: bytes) -> list[AnyRecord]:
    return [_ADAPTER.validate_json(line) for line in payload.decode("utf-8").splitlines() if line]
