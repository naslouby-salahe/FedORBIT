from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fedorbit.config.loading import active_config
from fedorbit.datasets.common import (
    AdapterSchema,
    DatasetAdapter,
    MissingSampleToken,
    NonFiniteFloatToken,
    ObservedColumnSamples,
    is_lossless_float64,
    reconcile_component_columns,
)
from fedorbit.datasets.materialization import (
    MaterializationError,
    build_normalized_rows,
    iter_component_frame_chunks,
    iter_component_raw_row_chunks,
    read_component_header_columns,
)
from fedorbit.datasets.preprocessing import (
    RowNormalizationError,
    conflicting_duplicate_labels_error,
    exact_duplicate_hash,
)
from fedorbit.datasets.ton_iot.components import ton_iot_adapter
from fedorbit.infrastructure.runtime import estimate_memory_budget, measure_efficiency
from fedorbit.types import (
    ByteCount,
    DatasetId,
    FineLabel,
    Index,
    MemoryMib,
    RawCellSamples,
    RawCellText,
    Sha256Digest,
    TabularColumnName,
    TabularColumns,
    ValidationReason,
)

DEFAULT_VALIDATION_CHUNK_ROWS = 8192
ESTIMATED_BYTES_PER_RAW_CELL = 128
HASH_BYTES = 32
DIGEST_HEX_LENGTH = 64


class ComponentContentState(StrEnum):
    VALID_FOR_CHRONOLOGICAL_PREPROCESSING = "valid_for_chronological_preprocessing"
    INVALID_DATA = "invalid_data"
    RESOURCE_BLOCKED = "resource_blocked"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ComponentContentVerdict:
    state: ComponentContentState
    reason: ValidationReason | None
    observed_row_count: Index
    distinct_duplicate_group_count: Index
    resident_row_count: Index
    accumulator_bytes: ByteCount
    peak_host_rss_mib: MemoryMib

    def __post_init__(self) -> None:
        if self.state is ComponentContentState.INVALID_DATA and not self.reason:
            raise MaterializationError("an Invalid Data verdict requires a registered reason")
        if self.state is ComponentContentState.RESOURCE_BLOCKED and not self.reason:
            raise MaterializationError("a resource-blocked verdict requires a measured reason")


@dataclass(slots=True)
class _ColumnSemantics:
    has_non_missing_observed: bool = False
    all_non_missing_lossless_float64: bool = True


class _AccumulatedColumnSamples(Mapping[TabularColumnName, RawCellSamples]):
    def __init__(
        self,
        columns: TabularColumns,
        semantics: Mapping[TabularColumnName, _ColumnSemantics],
    ) -> None:
        self._columns = columns
        self._semantics = semantics

    def _samples_of(self, column: TabularColumnName) -> RawCellSamples:
        column_semantics = self._semantics[column]
        if not column_semantics.has_non_missing_observed:
            return ()
        if column_semantics.all_non_missing_lossless_float64:
            return (RawCellText("1"),)
        return (RawCellText("observed"),)

    def __getitem__(self, key: TabularColumnName) -> RawCellSamples:
        return self._samples_of(key)

    def __iter__(self) -> Iterator[TabularColumnName]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)


def _missing_series(text: pd.Series) -> pd.Series:
    stripped = cast(pd.Series, text.str.strip().str.casefold())
    missing_tokens = tuple(token.value for token in MissingSampleToken)
    return stripped.isin(missing_tokens)


def _vectorized_lossless_float64(text: pd.Series, non_missing: pd.Series) -> pd.Series:
    stripped = cast(pd.Series, text.str.strip().str.casefold())
    non_finite_tokens = tuple(token.value for token in NonFiniteFloatToken)
    parsed = cast(pd.Series, pd.to_numeric(stripped.where(non_missing), errors="coerce"))
    parsed_values = cast(NDArray[np.float64], parsed.to_numpy(dtype=np.float64, na_value=np.nan))
    finite = np.isfinite(parsed_values)
    token_free = cast(NDArray[np.bool_], stripped.isin(non_finite_tokens).to_numpy(dtype=bool))
    return pd.Series(finite & token_free, index=text.index)


def _accumulate_column_semantics(
    paths: tuple[Path, ...],
    columns: TabularColumns,
    chunk_rows: int,
) -> tuple[dict[TabularColumnName, _ColumnSemantics], int]:
    semantics: dict[TabularColumnName, _ColumnSemantics] = OrderedDict(
        (column, _ColumnSemantics()) for column in columns
    )
    observed_row_count = 0
    for chunk in iter_component_frame_chunks(paths, chunk_rows):
        observed_row_count += len(chunk)
        chunk = chunk.reindex(columns=list(columns))
        for column in columns:
            text = cast(pd.Series, chunk[column])
            non_missing = ~_missing_series(text)
            if not bool(non_missing.any()):
                continue
            column_semantics = semantics[column]
            column_semantics.has_non_missing_observed = True
            if not column_semantics.all_non_missing_lossless_float64:
                continue
            vectorized = _vectorized_lossless_float64(text, non_missing)
            if bool(vectorized.where(non_missing, True).all()):
                continue
            candidates = cast(pd.Series, text[non_missing & ~vectorized])
            candidate_values = cast(list[object], candidates.tolist())
            exact = all(is_lossless_float64(RawCellText(str(value))) for value in candidate_values)
            if not exact:
                column_semantics.all_non_missing_lossless_float64 = False
    return semantics, observed_row_count


def _resolve_streaming_schema(
    dataset: DatasetId,
    columns: TabularColumns,
    semantics: Mapping[TabularColumnName, _ColumnSemantics],
) -> AdapterSchema:
    adapter: DatasetAdapter = ton_iot_adapter(dataset)
    samples = ObservedColumnSamples(_AccumulatedColumnSamples(columns, semantics))
    return adapter.resolve_schema(
        columns,
        timestamp_parse_success_fraction=1.0,
        timestamp_alias_minimum=(
            active_config().scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum
        ),
        observed_value_samples=samples,
    )


class _HashAccumulator:
    def __init__(self) -> None:
        self.hashes: NDArray[np.uint8] = np.zeros((0, HASH_BYTES), dtype=np.uint8)
        self.label_codes = np.zeros(0, dtype=np.int32)
        self.label_vocabulary: list[FineLabel] = []
        self.label_codes_by_label: dict[FineLabel, int] = {}
        self.row_count = 0

    def _ensure_capacity(self, additional: int) -> None:
        required = self.row_count + additional
        if required <= self.hashes.shape[0]:
            return
        capacity = max(required, max(1, self.hashes.shape[0]) * 2)
        grown_hashes = np.zeros((capacity, HASH_BYTES), dtype=np.uint8)
        grown_hashes[: self.row_count] = self.hashes[: self.row_count]
        self.hashes = grown_hashes
        grown_codes = np.zeros(capacity, dtype=np.int32)
        grown_codes[: self.row_count] = self.label_codes[: self.row_count]
        self.label_codes = grown_codes

    def observe(self, row_hash: Sha256Digest, label: FineLabel) -> None:
        self._ensure_capacity(1)
        position = self.row_count
        self.hashes[position] = np.frombuffer(bytes.fromhex(row_hash), dtype=np.uint8)
        code = self.label_codes_by_label.get(label)
        if code is None:
            code = len(self.label_vocabulary)
            self.label_codes_by_label[label] = code
            self.label_vocabulary.append(label)
        self.label_codes[position] = code
        self.row_count = position + 1

    def bytes_resident(self) -> ByteCount:
        return int(self.hashes.nbytes + self.label_codes.nbytes)

    def conflicting_group(self) -> tuple[Sha256Digest, tuple[FineLabel, ...]] | None:
        if self.row_count == 0:
            return None
        hashes = self.hashes[: self.row_count]
        labels = self.label_codes[: self.row_count]
        order = np.lexsort(tuple(hashes[:, index] for index in range(HASH_BYTES - 1, -1, -1)))
        sorted_hashes = hashes[order]
        sorted_labels = labels[order]
        boundaries = np.flatnonzero(np.any(sorted_hashes[1:] != sorted_hashes[:-1], axis=1)) + 1
        starts = np.concatenate((np.zeros(1, dtype=np.int64), boundaries))
        stops = np.concatenate((boundaries, np.zeros(1, dtype=np.int64) + self.row_count))
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
            group_labels = {int(code) for code in sorted_labels[start:stop]}
            if len(group_labels) > 1:
                digest = Sha256Digest(bytes(sorted_hashes[start]).hex())
                labels_in_group = tuple(
                    sorted(self.label_vocabulary[code] for code in group_labels)
                )
                return digest, labels_in_group
        return None

    def distinct_group_count(self) -> Index:
        if self.row_count == 0:
            return 0
        hashes = self.hashes[: self.row_count]
        order = np.lexsort(tuple(hashes[:, index] for index in range(HASH_BYTES - 1, -1, -1)))
        sorted_hashes = hashes[order]
        boundaries = np.flatnonzero(np.any(sorted_hashes[1:] != sorted_hashes[:-1], axis=1))
        return int(boundaries.size + 1)


def _chunk_buffer_bytes(columns: TabularColumns, chunk_rows: int) -> ByteCount:
    return int(chunk_rows * max(1, len(columns)) * ESTIMATED_BYTES_PER_RAW_CELL)


def _accumulator_budget_bytes(raw_bytes: ByteCount) -> ByteCount:
    return estimate_memory_budget(raw_bytes).budget_bytes


def validate_component_content(
    dataset: DatasetId,
    paths: tuple[Path, ...],
    chunk_rows: int = DEFAULT_VALIDATION_CHUNK_ROWS,
) -> ComponentContentVerdict:
    if chunk_rows < 1:
        raise MaterializationError("component validation requires at least one row per chunk")
    raw_bytes: ByteCount = sum(path.stat().st_size for path in paths)
    budget = _accumulator_budget_bytes(raw_bytes)
    header_columns = read_component_header_columns(paths)
    columns = reconcile_component_columns(header_columns)
    required_chunk_bytes = _chunk_buffer_bytes(columns, chunk_rows)
    if required_chunk_bytes > budget:
        return ComponentContentVerdict(
            ComponentContentState.RESOURCE_BLOCKED,
            ValidationReason(
                f"bounded component validation needs at least {required_chunk_bytes} bytes "
                f"of chunk buffers, exceeding the safe budget of {budget} bytes"
            ),
            0,
            0,
            0,
            0,
            0.0,
        )
    with measure_efficiency() as efficiency:
        semantics, observed_row_count = _accumulate_column_semantics(paths, columns, chunk_rows)
        schema = _resolve_streaming_schema(dataset, columns, semantics)
        accumulator = _HashAccumulator()
        rows_seen = 0
        for chunk in iter_component_raw_row_chunks(paths, columns, chunk_rows):
            try:
                normalized = build_normalized_rows(schema, chunk)
            except (RowNormalizationError, MaterializationError) as error:
                return ComponentContentVerdict(
                    ComponentContentState.INVALID_DATA,
                    ValidationReason(str(error)),
                    observed_row_count,
                    0,
                    0,
                    accumulator.bytes_resident(),
                    efficiency.result.peak_host_rss_mib,
                )
            for row in normalized:
                accumulator.observe(exact_duplicate_hash(row.features, schema), row.label)
            rows_seen += len(normalized)
            if accumulator.bytes_resident() > budget:
                return ComponentContentVerdict(
                    ComponentContentState.RESOURCE_BLOCKED,
                    ValidationReason(
                        f"bounded component validation accumulated {accumulator.bytes_resident()} "
                        f"bytes after {rows_seen} rows, exceeding the safe budget of {budget} bytes"
                    ),
                    rows_seen,
                    accumulator.distinct_group_count(),
                    0,
                    accumulator.bytes_resident(),
                    efficiency.result.peak_host_rss_mib,
                )
        conflict = accumulator.conflicting_group()
        peak = efficiency.result.peak_host_rss_mib
    valid_reason: ValidationReason | None = None
    if conflict is not None:
        group_sha256, labels = conflict
        return ComponentContentVerdict(
            ComponentContentState.INVALID_DATA,
            ValidationReason(
                "invalid duplicate groups: "
                f"{conflicting_duplicate_labels_error(group_sha256, labels)}"
            ),
            rows_seen,
            accumulator.distinct_group_count(),
            0,
            accumulator.bytes_resident(),
            peak,
        )
    return ComponentContentVerdict(
        ComponentContentState.VALID_FOR_CHRONOLOGICAL_PREPROCESSING,
        valid_reason,
        rows_seen,
        accumulator.distinct_group_count(),
        0,
        accumulator.bytes_resident(),
        peak,
    )
