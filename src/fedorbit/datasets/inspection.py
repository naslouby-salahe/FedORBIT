from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from fedorbit.artifacts.storage import atomic_write_json
from fedorbit.config.context import active_config
from fedorbit.datasets.edge_iiotset.loader import discover_edge_tabular_files
from fedorbit.datasets.ontology import normalize_label
from fedorbit.datasets.ton_iot.components import component_for
from fedorbit.datasets.ton_iot.loader import discover_ton_iot_component_files
from fedorbit.domain.enums import DatasetId, RawDatasetDirectory
from fedorbit.domain.serialization import StableJsonPayload, stable_json


class DatasetInspectionError(ValueError):
    pass


class ChronologyValidationState(StrEnum):
    VALID = "valid"
    MISSING_FIELD = "missing_field"
    AMBIGUOUS_EVENT_TIME = "ambiguous_event_time"
    UNPARSEABLE_EVENT_TIME = "unparseable_event_time"
    LABEL_INCONSISTENCY = "label_inconsistency"


@dataclass(frozen=True, slots=True)
class DatasetInspectionRequest:
    dataset: DatasetId
    raw_root: Path


@dataclass(frozen=True, slots=True)
class LabelCount:
    label: str
    row_count: int


@dataclass(frozen=True, slots=True)
class EventTimeInspection:
    field: str
    observed_row_count: int
    timestamp_pattern_row_count: int
    unusable_row_count: int
    state: ChronologyValidationState
    reason: str


@dataclass(frozen=True, slots=True)
class DatasetObservation:
    dataset: DatasetId
    row_count: int
    observed_columns: tuple[str, ...]
    local_class_counts: tuple[LabelCount, ...]
    binary_label_counts: tuple[LabelCount, ...]
    inconsistent_binary_label_rows: int
    event_time: EventTimeInspection

    @property
    def valid_for_chronological_preprocessing(self) -> bool:
        return self.event_time.state == ChronologyValidationState.VALID

    def fingerprint(self) -> str:
        payload = stable_json(cast(StableJsonPayload, self))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetObservationPersistenceRequest:
    observation: DatasetObservation
    preprocessing_root: Path


_AMBIGUOUS_EDGE_TIME = re.compile(r"^\d{4} \d{2}:\d{2}:\d{2}\.\d{1,9}$")


def inspect_dataset(request: DatasetInspectionRequest) -> DatasetObservation:
    path = _selected_path(request)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if not columns:
            raise DatasetInspectionError(f"empty selected table: {path}")
        labels = _labels_for(request.dataset)
        timestamp_field = (
            active_config().scientific.datasets.clients[request.dataset].expected_timestamp_field
        )
        class_counts: Counter[str] = Counter()
        binary_counts: Counter[str] = Counter()
        event_time_tally = EventTimeTally()
        rows = 0
        inconsistent = 0
        for row in reader:
            rows += 1
            multiclass = row.get(labels.multiclass_field)
            binary = row.get(labels.binary_field)
            if multiclass is None or binary is None:
                raise DatasetInspectionError("selected table row is missing a required label field")
            class_counts[multiclass] += 1
            binary_counts[binary] += 1
            if _binary_label_disagrees(multiclass, binary):
                inconsistent += 1
            if timestamp_field in columns:
                timestamp = row.get(timestamp_field)
                if timestamp is None:
                    raise DatasetInspectionError(
                        "selected table row is missing its event-time field"
                    )
                event_time_tally.observe(timestamp.strip())
    return DatasetObservation(
        dataset=request.dataset,
        row_count=rows,
        observed_columns=columns,
        local_class_counts=_sorted_counts(class_counts),
        binary_label_counts=_sorted_counts(binary_counts),
        inconsistent_binary_label_rows=inconsistent,
        event_time=_inspect_event_time(timestamp_field, columns, event_time_tally, inconsistent),
    )


def persist_dataset_observation(request: DatasetObservationPersistenceRequest) -> Path:
    fingerprint = request.observation.fingerprint()
    destination = (
        request.preprocessing_root
        / "validation"
        / f"{request.observation.dataset.value}.{fingerprint[:16]}.json"
    )
    atomic_write_json(destination, cast(StableJsonPayload, request.observation))
    return destination


@dataclass(frozen=True, slots=True)
class LabelFields:
    multiclass_field: str
    binary_field: str


@dataclass(slots=True)
class EventTimeTally:
    observed_row_count: int = 0
    timestamp_pattern_row_count: int = 0

    def observe(self, value: str) -> None:
        self.observed_row_count += 1
        self.timestamp_pattern_row_count += int(_AMBIGUOUS_EDGE_TIME.fullmatch(value) is not None)


def _labels_for(dataset: DatasetId) -> LabelFields:
    if dataset == DatasetId.EDGE_IIOTSET_NETWORK:
        return LabelFields("Attack_type", "Attack_label")
    return LabelFields("type", "label")


def _selected_path(request: DatasetInspectionRequest) -> Path:
    if request.dataset == DatasetId.EDGE_IIOTSET_NETWORK:
        return discover_edge_tabular_files(request.raw_root / RawDatasetDirectory.EDGE_IIOTSET)[0]
    return discover_ton_iot_component_files(
        request.raw_root / RawDatasetDirectory.TON_IOT,
        component_for(request.dataset),
    )[0]


def _binary_label_disagrees(multiclass: str, binary: str) -> bool:
    normalized = normalize_label(multiclass)
    if binary not in ("0", "1"):
        return True
    return (normalized == "normal") != (binary == "0")


def _sorted_counts(counts: Counter[str]) -> tuple[LabelCount, ...]:
    return tuple(LabelCount(label, count) for label, count in sorted(counts.items()))


def _inspect_event_time(
    field: str,
    columns: tuple[str, ...],
    tally: EventTimeTally,
    inconsistent_label_rows: int,
) -> EventTimeInspection:
    rows = tally.observed_row_count
    if inconsistent_label_rows:
        return EventTimeInspection(
            field,
            rows,
            0,
            rows,
            ChronologyValidationState.LABEL_INCONSISTENCY,
            "binary and multiclass label partitions disagree",
        )
    if field not in columns:
        return EventTimeInspection(
            field,
            0,
            0,
            0,
            ChronologyValidationState.MISSING_FIELD,
            "the configured event-time field is absent from the selected table",
        )
    unusable_rows = rows - tally.timestamp_pattern_row_count
    if unusable_rows:
        return EventTimeInspection(
            field,
            rows,
            tally.timestamp_pattern_row_count,
            unusable_rows,
            ChronologyValidationState.UNPARSEABLE_EVENT_TIME,
            "some event-time cells are not timestamp-shaped after CSV parsing",
        )
    if tally.timestamp_pattern_row_count:
        return EventTimeInspection(
            field,
            rows,
            tally.timestamp_pattern_row_count,
            0,
            ChronologyValidationState.AMBIGUOUS_EVENT_TIME,
            "event-time values omit calendar month and day and cannot establish chronology",
        )
    return EventTimeInspection(
        field,
        rows,
        0,
        rows,
        ChronologyValidationState.UNPARSEABLE_EVENT_TIME,
        "event-time parsing has not accepted this selected-table representation",
    )
