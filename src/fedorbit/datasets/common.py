from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from fedorbit.domain.enums import DatasetId


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
