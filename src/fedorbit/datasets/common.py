from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter, OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

from fedorbit.config.loading import active_config
from fedorbit.datasets.ontology import normalize_label
from fedorbit.types import DatasetId, RawDatasetDirectory, StableJsonPayload, stable_json


class FieldRole(StrEnum):
    TIMESTAMP = "timestamp"
    MULTICLASS_LABEL = "multiclass_label"
    BINARY_LABEL = "binary_label"
    BEHAVIORAL_NUMERIC = "behavioral_numeric"
    BEHAVIORAL_CATEGORICAL = "behavioral_categorical"
    FORBIDDEN_IDENTITY = "forbidden_identity"
    FORBIDDEN_PAYLOAD = "forbidden_payload"
    FORBIDDEN_PROVENANCE = "forbidden_provenance"


IDENTITY_MARKERS = (
    "ip.",
    "mac",
    "host",
    "device",
    "process",
    "thread",
    "flow",
    "session",
    "row",
    "index",
    "filename",
    "src_ip",
    "dst_ip",
    "pid",
    "uid",
    "gid",
)
PAYLOAD_MARKERS = (
    "payload",
    "file_data",
    "full_uri",
    "uri.query",
    "msg",
    "options",
    "referer",
)
PROVENANCE_MARKERS = ("source_file", "capture", "acquisition", "provenance", "file_name")


class DatasetSchemaError(ValueError):
    pass


def _empty_roles() -> Mapping[str, FieldRole]:
    return OrderedDict()


RawCellValue = str | int | float | None


@dataclass(frozen=True, slots=True)
class ObservedColumnSamples:
    samples_by_column: Mapping[str, tuple[RawCellValue, ...]]

    def samples_of(self, column_name: str) -> tuple[RawCellValue, ...]:
        return self.samples_by_column.get(column_name, ())


@dataclass(frozen=True, slots=True)
class AdapterSchema:
    dataset_id: DatasetId
    feature_order: tuple[str, ...]
    roles: Mapping[str, FieldRole] = field(default_factory=_empty_roles)
    timestamp_column: str | None = None
    multiclass_label_column: str | None = None
    binary_label_column: str | None = None
    observed_columns: tuple[str, ...] = field(default_factory=tuple)
    excluded_columns: tuple[str, ...] = field(default_factory=tuple)

    def role_of(self, column: str) -> FieldRole:
        return self.roles.get(column, FieldRole.BEHAVIORAL_CATEGORICAL)

    def behavioral_features(self) -> tuple[str, ...]:
        return tuple(
            column
            for column in self.feature_order
            if self.role_of(column)
            in (FieldRole.BEHAVIORAL_NUMERIC, FieldRole.BEHAVIORAL_CATEGORICAL)
        )


@dataclass(frozen=True, slots=True)
class ResolvedLabelColumns:
    multiclass_label_field: str
    binary_label_field: str


@dataclass(frozen=True, slots=True)
class AdapterContract:
    dataset_id: DatasetId
    timestamp_candidates: tuple[str, ...]
    multiclass_label_candidates: tuple[str, ...]
    binary_label_candidates: tuple[str, ...]
    additional_exclusions: frozenset[str] = frozenset()
    official_feature_order: tuple[str, ...] = ()


def exactly_one_candidate(
    columns: tuple[str, ...],
    candidates: tuple[str, ...],
    semantic_role: str,
) -> str:
    observed = tuple(column for column in columns if column in candidates)
    if len(observed) != 1:
        message = (
            f"{semantic_role}: expected exactly one observed column among {candidates}, "
            f"found {observed}"
        )
        raise DatasetSchemaError(message)
    return observed[0]


def resolve_timestamp_column(
    columns: tuple[str, ...],
    candidates: tuple[str, ...],
    parse_success_fraction: float,
    minimum_fraction: float,
) -> str:
    column = exactly_one_candidate(columns, candidates, "timestamp")
    if parse_success_fraction < minimum_fraction:
        message = (
            f"timestamp alias {column!r} parse success {parse_success_fraction} "
            f"below minimum {minimum_fraction}"
        )
        raise DatasetSchemaError(message)
    return column


def resolve_label_columns(
    columns: tuple[str, ...],
    expected_multiclass: tuple[str, ...],
    expected_binary: tuple[str, ...],
) -> ResolvedLabelColumns:
    return ResolvedLabelColumns(
        exactly_one_candidate(columns, expected_multiclass, "multiclass label"),
        exactly_one_candidate(columns, expected_binary, "binary label"),
    )


def is_missing_sample(value: RawCellValue) -> bool:
    return str(value).strip().casefold() in ("", "0", "0.0", "nan", "none", "null")


def _lossless_float64(value: str | int | float) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    text = value.strip().casefold()
    if text in ("nan", "inf", "-inf", "infinity", "-infinity"):
        return False
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def infer_feature_type(samples: tuple[RawCellValue, ...]) -> FieldRole:
    non_missing = tuple(
        sample for sample in samples if sample is not None and not is_missing_sample(sample)
    )
    if not non_missing:
        return FieldRole.BEHAVIORAL_CATEGORICAL
    if all(_lossless_float64(sample) for sample in non_missing):
        return FieldRole.BEHAVIORAL_NUMERIC
    return FieldRole.BEHAVIORAL_CATEGORICAL


def role_for_field(field: str) -> FieldRole:
    lowered = field.casefold()
    if any(marker in lowered for marker in PROVENANCE_MARKERS):
        return FieldRole.FORBIDDEN_PROVENANCE
    if any(marker in lowered for marker in PAYLOAD_MARKERS):
        return FieldRole.FORBIDDEN_PAYLOAD
    if any(marker in lowered for marker in IDENTITY_MARKERS):
        return FieldRole.FORBIDDEN_IDENTITY
    return FieldRole.BEHAVIORAL_CATEGORICAL


class DatasetAdapter:
    def __init__(self, contract: AdapterContract) -> None:
        self._contract = contract

    @property
    def dataset_id(self) -> DatasetId:
        return self._contract.dataset_id

    def resolve_schema(
        self,
        observed_columns: tuple[str, ...],
        timestamp_parse_success_fraction: float,
        timestamp_alias_minimum: float,
        observed_value_samples: ObservedColumnSamples | None = None,
    ) -> AdapterSchema:
        if len(set(observed_columns)) != len(observed_columns):
            raise DatasetSchemaError("duplicate observed column names are invalid")
        timestamp = resolve_timestamp_column(
            observed_columns,
            self._contract.timestamp_candidates,
            timestamp_parse_success_fraction,
            timestamp_alias_minimum,
        )
        labels = resolve_label_columns(
            observed_columns,
            self._contract.multiclass_label_candidates,
            self._contract.binary_label_candidates,
        )
        roles: OrderedDict[str, FieldRole] = OrderedDict()
        excluded = self._contract.additional_exclusions
        for column in observed_columns:
            if column == timestamp:
                roles[column] = FieldRole.TIMESTAMP
            elif column == labels.multiclass_label_field:
                roles[column] = FieldRole.MULTICLASS_LABEL
            elif column == labels.binary_label_field:
                roles[column] = FieldRole.BINARY_LABEL
            elif column in excluded:
                inferred = role_for_field(column)
                roles[column] = (
                    inferred
                    if inferred != FieldRole.BEHAVIORAL_CATEGORICAL
                    else FieldRole.FORBIDDEN_PROVENANCE
                )
            else:
                inferred = role_for_field(column)
                if (
                    inferred == FieldRole.BEHAVIORAL_CATEGORICAL
                    and observed_value_samples is not None
                ):
                    inferred = infer_feature_type(observed_value_samples.samples_of(column))
                roles[column] = inferred
        feature_order = (
            self._contract.official_feature_order
            if self._contract.official_feature_order
            else tuple(
                column
                for column in observed_columns
                if column
                not in (timestamp, labels.multiclass_label_field, labels.binary_label_field)
            )
        )
        return AdapterSchema(
            dataset_id=self._contract.dataset_id,
            feature_order=feature_order,
            roles=roles,
            timestamp_column=timestamp,
            multiclass_label_column=labels.multiclass_label_field,
            binary_label_column=labels.binary_label_field,
            observed_columns=observed_columns,
            excluded_columns=tuple(column for column in observed_columns if column in excluded),
        )


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
        event_time=inspect_event_time(timestamp_field, columns, event_time_tally, inconsistent),
    )


def persist_dataset_observation(request: DatasetObservationPersistenceRequest) -> Path:
    from fedorbit.infrastructure.execution import atomic_write_json

    destination = request.preprocessing_root / "validation" / request.observation.dataset.value
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "validation.json", cast(StableJsonPayload, request.observation))
    atomic_write_json(
        destination / "leakage.json",
        cast(
            StableJsonPayload,
            OrderedDict(
                inconsistent_binary_label_rows=request.observation.inconsistent_binary_label_rows,
                chronology_state=request.observation.event_time.state.value,
                chronology_reason=request.observation.event_time.reason,
            ),
        ),
    )
    atomic_write_json(
        destination / "timestamp_aliases.json",
        cast(
            StableJsonPayload,
            OrderedDict(
                timestamp_field=request.observation.event_time.field,
                observed_pattern_rows=request.observation.event_time.timestamp_pattern_row_count,
                unusable_rows=request.observation.event_time.unusable_row_count,
            ),
        ),
    )
    return destination / "validation.json"


@dataclass(frozen=True, slots=True)
class LabelFields:
    multiclass_field: str
    binary_field: str


@dataclass(slots=True)
class EventTimeTally:
    observed_row_count: int = 0
    timestamp_pattern_row_count: int = 0
    resolvable_row_count: int = 0

    def observe(self, value: str) -> None:
        self.observed_row_count += 1
        ambiguous = _AMBIGUOUS_EDGE_TIME.fullmatch(value) is not None
        resolvable = _is_resolvable_event_time(value)
        self.timestamp_pattern_row_count += int(ambiguous or resolvable)
        self.resolvable_row_count += int(resolvable)


def _is_resolvable_event_time(value: str) -> bool:
    if _AMBIGUOUS_EDGE_TIME.fullmatch(value) is not None:
        return False
    try:
        epoch_seconds = float(value)
    except ValueError:
        epoch_seconds = None
    if epoch_seconds is not None:
        try:
            timestamp = datetime.fromtimestamp(epoch_seconds, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return False
        return 2000 <= timestamp.year <= 2100
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return False
    return True


def _labels_for(dataset: DatasetId) -> LabelFields:
    if dataset == DatasetId.EDGE_IIOTSET_NETWORK:
        return LabelFields("Attack_type", "Attack_label")
    return LabelFields("type", "label")


def _selected_path(request: DatasetInspectionRequest) -> Path:
    from fedorbit.datasets.edge_iiotset.loader import discover_edge_tabular_files
    from fedorbit.datasets.ton_iot.components import component_for
    from fedorbit.datasets.ton_iot.loader import discover_ton_iot_component_files

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


def inspect_event_time(
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
    if tally.resolvable_row_count == rows:
        return EventTimeInspection(
            field,
            rows,
            tally.timestamp_pattern_row_count,
            0,
            ChronologyValidationState.VALID,
            "all event-time cells are uniquely resolvable",
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
