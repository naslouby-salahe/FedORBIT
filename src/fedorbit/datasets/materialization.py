from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import psutil
import torch

from fedorbit.config.loading import active_config
from fedorbit.datasets.common import (
    AdapterSchema,
    FieldRole,
    ObservedColumnSamples,
    reconcile_component_columns,
)
from fedorbit.datasets.ontology import normalize_label, transfer_concept_for, transfer_eligibility
from fedorbit.datasets.preprocessing import (
    CategoricalPreprocessor,
    FeatureQualityReport,
    NormalizedFeatureVector,
    NormalizedRow,
    NumericPreprocessor,
    RawFeatureValue,
    TrainingFeatureValues,
    evaluate_feature_quality,
    fit_categorical_preprocessor,
    fit_numeric_preprocessor,
    normalize_and_split_training_rows,
    normalize_value,
    numeric_zero_is_not_missing,
    one_hot,
    transform_numeric,
)
from fedorbit.datasets.splitting import DuplicateGroupId
from fedorbit.datasets.ton_iot.components import component_for, ton_iot_adapter
from fedorbit.datasets.ton_iot.loader import discover_ton_iot_component_files
from fedorbit.types import DatasetId, OracleTransferConcept, RawDatasetDirectory, Split


class MaterializationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LocalClassManifest:
    class_names: tuple[str, ...]
    excluded_classes: tuple[tuple[str, int], ...]

    def index_of(self, normalized_label: str) -> int:
        return self.class_names.index(normalized_label)

    @property
    def class_count(self) -> int:
        return len(self.class_names)


@dataclass(frozen=True, slots=True)
class SplitTensors:
    features: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True, slots=True)
class MaterializedClient:
    dataset: DatasetId
    schema: AdapterSchema
    class_manifest: LocalClassManifest
    feature_names: tuple[str, ...]
    splits: Mapping[Split, SplitTensors]
    feature_quality: FeatureQualityReport
    class_row_counts: Mapping[str, Mapping[Split, int]]


def _read_component_rows(paths: tuple[Path, ...]) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    per_file_columns: list[tuple[str, ...]] = []
    for path in paths:
        frame = pd.read_csv(
            path,
            dtype=object,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
            dtype_backend="numpy_nullable",
        )
        observed: tuple[str, ...] = tuple(cast(str, column) for column in frame.columns)
        if not observed:
            raise MaterializationError(f"empty selected table: {path}")
        per_file_columns.append(observed)
        frames.append(frame)
    columns = reconcile_component_columns(tuple(per_file_columns))
    combined = pd.concat(
        [frame.reindex(columns=list(columns)) for frame in frames],
        ignore_index=True,
    )
    column_arrays: list[list[str]] = [
        [cast(str, value) for value in combined[column].to_numpy(dtype=object)]
        for column in columns
    ]
    rows = [dict(zip(columns, values, strict=True)) for values in zip(*column_arrays, strict=True)]
    return columns, rows


class _LazyColumnSamples(Mapping[str, tuple[str, ...]]):
    def __init__(self, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
        self._rows = rows
        self._columns = columns
        self._cache: dict[str, tuple[str, ...]] = OrderedDict()

    def __getitem__(self, key: str) -> tuple[str, ...]:
        if key not in self._cache:
            self._cache[key] = tuple(row.get(key, "") for row in self._rows)
        return self._cache[key]

    def __iter__(self):
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)


def _resolve_schema(
    dataset: DatasetId, columns: tuple[str, ...], rows: list[dict[str, str]]
) -> AdapterSchema:
    adapter = ton_iot_adapter(dataset)
    samples = ObservedColumnSamples(_LazyColumnSamples(rows, columns))
    return adapter.resolve_schema(
        columns,
        timestamp_parse_success_fraction=1.0,
        timestamp_alias_minimum=active_config().scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum,
        observed_value_samples=samples,
    )


def _parse_epoch_seconds(value: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise MaterializationError(f"unparseable event-time cell: {value!r}") from error


def _build_normalized_rows(
    schema: AdapterSchema,
    rows: list[dict[str, str]],
) -> tuple[NormalizedRow, ...]:
    behavioral = schema.behavioral_features()
    categorical_columns = frozenset(
        column
        for column in behavioral
        if schema.role_of(column) == FieldRole.BEHAVIORAL_CATEGORICAL
    )
    assert schema.timestamp_column is not None
    assert schema.multiclass_label_column is not None
    timestamp_column = schema.timestamp_column
    label_column = schema.multiclass_label_column
    normalized_rows: list[NormalizedRow] = []
    for row in rows:
        normalized_label = normalize_label(row[label_column])
        timestamp = _parse_epoch_seconds(row[timestamp_column])
        values: OrderedDict[str, RawFeatureValue] = OrderedDict()
        for column in behavioral:
            is_categorical = column in categorical_columns
            values[column] = normalize_value(row.get(column, ""), is_categorical)
        normalized_rows.append(
            NormalizedRow(
                features=NormalizedFeatureVector(values),
                label=normalized_label,
                timestamp_fraction=timestamp,
                group_id="",
            )
        )
    return tuple(normalized_rows)


def _retained_local_classes(rows: tuple[NormalizedRow, ...]) -> LocalClassManifest:
    minimum = (
        active_config().scientific.transfer_support.local_prediction_attack_class_total_rows_minimum
    )
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.label] += 1
    retained = sorted(
        label for label, count in counts.items() if label == "normal" or count >= minimum
    )
    excluded = tuple(
        sorted((label, count) for label, count in counts.items() if label not in retained)
    )
    if "normal" not in retained:
        raise MaterializationError("retained local class set does not contain Normal")
    return LocalClassManifest(tuple(retained), excluded)


def _assign_splits(
    schema: AdapterSchema,
    rows: tuple[NormalizedRow, ...],
    manifest: LocalClassManifest,
) -> dict[Split, list[NormalizedRow]]:
    buckets: dict[Split, list[NormalizedRow]] = OrderedDict((split, []) for split in Split)
    by_class: defaultdict[str, list[NormalizedRow]] = defaultdict(list)
    for row in rows:
        if row.label in manifest.class_names:
            by_class[row.label].append(row)
    for class_rows in by_class.values():
        normalized = normalize_and_split_training_rows(schema, tuple(class_rows))
        for group_sha256, members in normalized.duplicate_groups.groups:
            split = normalized.split_assignment.split_of(DuplicateGroupId(group_sha256))
            if split is None:
                raise MaterializationError("duplicate group received no split assignment")
            buckets[split].extend(members)
    return buckets


def _numeric_array(rows: list[NormalizedRow], column: str) -> np.ndarray:
    return np.array([row.features.value_of(column) for row in rows], dtype=object)


def _categorical_array(rows: list[NormalizedRow], column: str) -> tuple[str, ...]:
    return tuple(str(row.features.value_of(column)) for row in rows)


def _missing_indicator(rows: list[NormalizedRow], column: str) -> np.ndarray:
    indicator = np.zeros(len(rows), dtype=np.float32)
    for index, row in enumerate(rows):
        value = row.features.value_of(column)
        numeric = float("nan") if value is None or isinstance(value, str) else float(value)
        is_missing = not (np.isfinite(numeric) or numeric_zero_is_not_missing(numeric))
        indicator[index] = 1.0 if is_missing else 0.0
    return indicator


_OBSERVED_PEAK_MEMORY_TO_RAW_BYTES_RATIO = 15.0
_MAXIMUM_MEMORY_BUDGET_FRACTION = 0.65


def require_safe_memory_budget(dataset: DatasetId, paths: tuple[Path, ...]) -> None:
    raw_bytes = sum(path.stat().st_size for path in paths)
    estimated_peak_bytes = raw_bytes * _OBSERVED_PEAK_MEMORY_TO_RAW_BYTES_RATIO
    available_bytes = psutil.virtual_memory().available
    budget_bytes = available_bytes * _MAXIMUM_MEMORY_BUDGET_FRACTION
    if estimated_peak_bytes > budget_bytes:
        raise MaterializationError(
            f"{dataset.value} materialization is estimated to need "
            f"{estimated_peak_bytes / 1e9:.1f} GB, exceeding the safe budget of "
            f"{budget_bytes / 1e9:.1f} GB ({available_bytes / 1e9:.1f} GB currently available); "
            "refusing to proceed to avoid an out-of-memory crash"
        )


def materialize_client(dataset: DatasetId, raw_root: Path) -> MaterializedClient:
    if dataset == DatasetId.EDGE_IIOTSET_NETWORK:
        raise MaterializationError(
            "Edge-IIoTset network is Invalid Data for chronological materialization"
        )
    component = component_for(dataset)
    paths = discover_ton_iot_component_files(raw_root / RawDatasetDirectory.TON_IOT, component)
    require_safe_memory_budget(dataset, paths)
    columns, raw_rows = _read_component_rows(paths)
    schema = _resolve_schema(dataset, columns, raw_rows)
    rows = _build_normalized_rows(schema, raw_rows)
    del raw_rows
    manifest = _retained_local_classes(rows)
    buckets = _assign_splits(schema, rows, manifest)
    del rows
    class_row_counts = OrderedDict(
        (
            label,
            OrderedDict(
                (split, sum(1 for row in buckets[split] if row.label == label)) for split in Split
            ),
        )
        for label in manifest.class_names
    )
    behavioral = schema.behavioral_features()
    categorical_columns = frozenset(
        column
        for column in behavioral
        if schema.role_of(column) == FieldRole.BEHAVIORAL_CATEGORICAL
    )
    train_rows = buckets[Split.TRAIN]
    if not train_rows:
        raise MaterializationError(f"{dataset.value} has no TRAIN rows after chronological split")
    train_values = TrainingFeatureValues(
        OrderedDict(
            (
                column,
                _numeric_array(train_rows, column)
                if column not in categorical_columns
                else np.array(_categorical_array(train_rows, column), dtype=object),
            )
            for column in behavioral
        )
    )
    quality = evaluate_feature_quality(behavioral, categorical_columns, train_values)
    if quality.client_invalid:
        raise MaterializationError(
            f"{dataset.value} is Invalid Data: {quality.client_invalid_reason}"
        )
    numeric_preprocessors: OrderedDict[str, NumericPreprocessor] = OrderedDict()
    categorical_preprocessors: OrderedDict[str, CategoricalPreprocessor] = OrderedDict()
    missing_indicator_columns: list[str] = []
    for candidate in quality.candidate_features:
        if candidate.dropped:
            continue
        if candidate.is_categorical:
            categorical_preprocessors[candidate.name] = fit_categorical_preprocessor(
                _categorical_array(train_rows, candidate.name)
            )
        else:
            fitted = fit_numeric_preprocessor(_numeric_array(train_rows, candidate.name))
            if fitted.constant_after_imputation:
                continue
            numeric_preprocessors[candidate.name] = fitted
            if candidate.missing_indicator:
                missing_indicator_columns.append(candidate.name)
    feature_names = (
        *numeric_preprocessors.keys(),
        *(f"{name}__missing" for name in missing_indicator_columns),
        *(
            f"{name}={category}"
            for name in categorical_preprocessors
            for category in categorical_preprocessors[name].vocabulary
        ),
    )
    splits: dict[Split, SplitTensors] = OrderedDict()
    for split in Split:
        split_rows = buckets[split]
        if not split_rows:
            splits[split] = SplitTensors(
                torch.empty((0, len(feature_names)), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )
            continue
        blocks: list[np.ndarray] = []
        for name, fitted in numeric_preprocessors.items():
            blocks.append(
                transform_numeric(_numeric_array(split_rows, name), fitted).astype(np.float32)
            )
        for name in missing_indicator_columns:
            blocks.append(_missing_indicator(split_rows, name))
        for name, fitted in categorical_preprocessors.items():
            one_hot_columns = np.array(
                [one_hot(value, fitted) for value in _categorical_array(split_rows, name)],
                dtype=np.float32,
            )
            for column_index in range(len(fitted.vocabulary)):
                blocks.append(one_hot_columns[:, column_index])
        matrix = (
            np.stack(blocks, axis=1) if blocks else np.empty((len(split_rows), 0), dtype=np.float32)
        )
        targets = np.array([manifest.index_of(row.label) for row in split_rows], dtype=np.int64)
        splits[split] = SplitTensors(
            torch.from_numpy(matrix.astype(np.float32, copy=False)),
            torch.from_numpy(targets),
        )
    return MaterializedClient(
        dataset=dataset,
        schema=schema,
        class_manifest=manifest,
        feature_names=feature_names,
        splits=splits,
        feature_quality=quality,
        class_row_counts=class_row_counts,
    )


@dataclass(frozen=True, slots=True)
class TransferConceptGroup:
    concept: OracleTransferConcept
    native_class_indices: tuple[int, ...]
    train_support: int
    meta_support: int
    source_eligible: bool


def transfer_concept_groups(
    dataset: DatasetId,
    materialized: MaterializedClient,
) -> tuple[TransferConceptGroup, ...]:
    grouped: OrderedDict[OracleTransferConcept, list[int]] = OrderedDict()
    for index, label in enumerate(materialized.class_manifest.class_names):
        concept = transfer_concept_for(dataset, label)
        if concept is not None:
            grouped.setdefault(concept, []).append(index)
    groups: list[TransferConceptGroup] = []
    for concept in OracleTransferConcept:
        indices = grouped.get(concept)
        if indices is None:
            continue
        train_support = sum(
            materialized.class_row_counts[materialized.class_manifest.class_names[index]][
                Split.TRAIN
            ]
            for index in indices
        )
        meta_support = sum(
            materialized.class_row_counts[materialized.class_manifest.class_names[index]][
                Split.META
            ]
            for index in indices
        )
        eligibility = transfer_eligibility(train_support, meta_support, 0, 0, 0)
        groups.append(
            TransferConceptGroup(
                concept=concept,
                native_class_indices=tuple(indices),
                train_support=train_support,
                meta_support=meta_support,
                source_eligible=eligibility.source_eligible,
            )
        )
    return tuple(groups)


def eligible_source_transfer_node_classes(
    groups: tuple[TransferConceptGroup, ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(group.native_class_indices for group in groups if group.source_eligible)
