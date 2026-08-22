from __future__ import annotations

import math
from dataclasses import dataclass, field

from fedorbit.domain.enums import DatasetId

TIMESTAMP_ROLE = "timestamp"
MULTICLASS_LABEL_ROLE = "multiclass_label"
BINARY_LABEL_ROLE = "binary_label"
BEHAVIORAL_NUMERIC_ROLE = "behavioral_numeric"
BEHAVIORAL_CATEGORICAL_ROLE = "behavioral_categorical"
FORBIDDEN_IDENTITY_ROLE = "forbidden_identity"
FORBIDDEN_PAYLOAD_ROLE = "forbidden_payload"
FORBIDDEN_PROVENANCE_ROLE = "forbidden_provenance"

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

PAYLOAD_MARKERS = ("payload", "file_data", "full_uri", "uri.query", "msg", "options", "referer")

PROVENANCE_MARKERS = ("source_file", "capture", "acquisition", "provenance", "file_name")


class SchemaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AdapterSchema:
    dataset_id: DatasetId
    canonical_feature_order: tuple[str, ...]
    roles: dict[str, str] = field(default_factory=lambda: {})
    timestamp_column: str | None = None
    multiclass_label_column: str | None = None
    binary_label_column: str | None = None
    observed_columns: tuple[str, ...] = field(default_factory=tuple)
    excluded_columns: tuple[str, ...] = field(default_factory=tuple)

    def role_of(self, column: str) -> str:
        return self.roles.get(column, BEHAVIORAL_CATEGORICAL_ROLE)

    def behavioral_features(self) -> tuple[str, ...]:
        return tuple(
            column
            for column in self.canonical_feature_order
            if self.role_of(column) in (BEHAVIORAL_NUMERIC_ROLE, BEHAVIORAL_CATEGORICAL_ROLE)
        )


def exactly_one_candidate(
    columns: tuple[str, ...], candidates: tuple[str, ...], semantic_role: str
) -> str:
    observed = [column for column in columns if column in candidates]
    if len(observed) != 1:
        raise SchemaError(
            f"{semantic_role}: expected exactly one observed column "
            f"among {candidates}, found {observed}"
        )
    return observed[0]


def resolve_timestamp_column(
    columns: tuple[str, ...],
    candidates: tuple[str, ...],
    parse_success_fraction: float,
    minimum_fraction: float,
) -> str:
    column = exactly_one_candidate(columns, candidates, "timestamp")
    if parse_success_fraction < minimum_fraction:
        raise SchemaError(
            f"timestamp alias {column!r} parse success {parse_success_fraction} "
            f"below minimum {minimum_fraction}"
        )
    return column


def resolve_label_columns(
    columns: tuple[str, ...],
    expected_multiclass: tuple[str, ...],
    expected_binary: tuple[str, ...],
) -> tuple[str, str]:
    multiclass = exactly_one_candidate(columns, expected_multiclass, "multiclass label")
    binary = exactly_one_candidate(columns, expected_binary, "binary label")
    return multiclass, binary


def _lossless_float64(value: str | int | float) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    text = str(value).strip()
    if text in ("nan", "inf", "-inf", "infinity", "-infinity"):
        return False
    try:
        parsed = float(text)
    except ValueError:
        return False
    return math.isfinite(parsed)


def infer_feature_type(field: str, samples: tuple[str | int | float | None, ...]) -> str:
    non_missing: list[str | int | float] = [
        sample for sample in samples if sample is not None and not is_missing_sample(sample)
    ]
    if not non_missing:
        return BEHAVIORAL_CATEGORICAL_ROLE
    if all(_lossless_float64(sample) for sample in non_missing):
        return BEHAVIORAL_NUMERIC_ROLE
    return BEHAVIORAL_CATEGORICAL_ROLE


def is_missing_sample(value: str | int | float | None) -> bool:
    lowered = str(value).strip().lower()
    return lowered in ("", "0", "0.0", "nan", "none", "null")


def role_for_field(field: str) -> str:
    lowered = field.lower()
    if any(marker in lowered for marker in PROVENANCE_MARKERS):
        return FORBIDDEN_PROVENANCE_ROLE
    if any(marker in lowered for marker in PAYLOAD_MARKERS):
        return FORBIDDEN_PAYLOAD_ROLE
    if any(marker in lowered for marker in IDENTITY_MARKERS):
        return FORBIDDEN_IDENTITY_ROLE
    return BEHAVIORAL_CATEGORICAL_ROLE
