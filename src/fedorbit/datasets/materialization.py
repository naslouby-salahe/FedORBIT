from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from fedorbit.config.loading import active_config
from fedorbit.datasets.common import (
    AdapterSchema,
    FieldRole,
    ObservedColumnSamples,
    file_sha256,
    reconcile_component_columns,
)
from fedorbit.datasets.ontology import (
    NORMAL_LABEL,
    normalize_label,
    transfer_concept_for,
    transfer_eligibility,
)
from fedorbit.datasets.preprocessing import (
    CategoricalPreprocessor,
    FeatureQualityReport,
    FeatureValue,
    NormalizedFeatureVector,
    NormalizedRow,
    NumericPreprocessor,
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
from fedorbit.infrastructure.runtime import estimate_memory_budget
from fedorbit.types import (
    ByteCount,
    ClassCount,
    ClassIndex,
    ClientComponentName,
    DatasetId,
    DatasetLabel,
    DuplicateGroupIdentifier,
    ExcludedLocalClasses,
    FeatureName,
    FeatureNames,
    FineLabel,
    Fraction,
    Index,
    LocalClassNames,
    NormalizedGroupIdentifier,
    NumericFeatureValue,
    OracleTransferConcept,
    RandomSeed,
    RawCellText,
    RawDatasetDirectory,
    RawDatasetPath,
    RawTabularColumns,
    RawTabularRows,
    Sha256Digest,
    Split,
    TabularColumnName,
    TabularColumns,
    TimestampRange,
    TimestampSeconds,
)


class MaterializationError(ValueError):
    pass


class MaterializationResourceLimitError(MaterializationError):
    pass


@dataclass(frozen=True, slots=True)
class LocalClassManifest:
    class_names: LocalClassNames
    excluded_classes: ExcludedLocalClasses

    def index_of(self, normalized_label: FineLabel) -> ClassIndex:
        return ClassIndex(self.class_names.index(normalized_label))

    @property
    def class_count(self) -> ClassCount:
        count: ClassCount = len(self.class_names)
        return count


@dataclass(frozen=True, slots=True)
class SplitTensors:
    features: torch.Tensor
    targets: torch.Tensor


def deterministic_smallest_hash_subsample_indices(
    class_row_indices: tuple[Index, ...],
    fraction: Fraction,
    seed: RandomSeed,
    coordinates_text: str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> tuple[Index, ...]:
    keep = max(1, round(fraction * len(class_row_indices)))
    ranked: list[tuple[int, Index]] = []
    for row_index in class_row_indices:
        payload = f"FedORBIT|weak-signal-support-subsample|{seed}|{coordinates_text}|{row_index}" #TODO: should be enums not hardcoded strings
        digest_int = int.from_bytes(
            hashlib.sha256(payload.encode("utf-8")).digest(), byteorder="big"
        )
        ranked.append((digest_int, row_index))
    ranked.sort()
    return tuple(sorted(row_index for _, row_index in ranked[:keep]))


def subsample_split_tensors(
    split: SplitTensors,
    fraction: Fraction,
    seed: RandomSeed,
    coordinates_text: str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> SplitTensors:
    target_values: list[int] = split.targets.tolist()
    keep_indices: list[int] = []
    for class_value in sorted({int(value) for value in target_values}):
        class_row_indices: tuple[Index, ...] = tuple(
            row_index for row_index, value in enumerate(target_values) if int(value) == class_value
        )
        kept = deterministic_smallest_hash_subsample_indices(
            class_row_indices, fraction, seed, f"{coordinates_text}|class-{class_value}"
        )
        keep_indices.extend(int(row_index) for row_index in kept)
    keep_indices.sort()
    index_tensor = torch.tensor(keep_indices, dtype=torch.long)
    return SplitTensors(features=split.features[index_tensor], targets=split.targets[index_tensor])


_SUBSAMPLED_SPLITS = (Split.TRAIN, Split.META, Split.CONFIRM)


def _recompute_class_row_counts(
    original: Mapping[FineLabel, Mapping[Split, Index]],
    class_names: LocalClassNames,
    subsampled_splits: Mapping[Split, SplitTensors],
) -> Mapping[FineLabel, Mapping[Split, Index]]:
    updated: OrderedDict[FineLabel, OrderedDict[Split, Index]] = OrderedDict(
        (fine_label, OrderedDict(per_split)) for fine_label, per_split in original.items()
    )
    for split_name in _SUBSAMPLED_SPLITS:
        counts: dict[int, int] = OrderedDict()
        split_target_values: list[int] = subsampled_splits[split_name].targets.tolist()
        for value in split_target_values:
            counts[value] = counts.get(value, 0) + 1
        for class_index, fine_label in enumerate(class_names):
            count: Index = counts.get(class_index, 0)
            updated[fine_label][split_name] = count
    return updated


def subsampled_materialized_client(
    materialized: MaterializedClient,
    fraction: Fraction,
    seed: RandomSeed,
) -> MaterializedClient:
    coordinates_text = f"{materialized.dataset.value}|support-fraction-{fraction}"
    subsampled_splits: OrderedDict[Split, SplitTensors] = OrderedDict(materialized.splits)
    for split_name in _SUBSAMPLED_SPLITS:
        subsampled_splits[split_name] = subsample_split_tensors(
            materialized.splits[split_name],
            fraction,
            seed,
            f"{coordinates_text}|{split_name.value}",
        )
    class_row_counts = _recompute_class_row_counts(
        materialized.class_row_counts, materialized.class_manifest.class_names, subsampled_splits
    )
    return MaterializedClient(
        dataset=materialized.dataset,
        schema=materialized.schema,
        class_manifest=materialized.class_manifest,
        feature_names=materialized.feature_names,
        splits=subsampled_splits,
        feature_quality=materialized.feature_quality,
        class_row_counts=class_row_counts,
        provenance=materialized.provenance,
    )


@dataclass(frozen=True, slots=True)
class RawFileProvenance:
    path: RawDatasetPath
    sha256: Sha256Digest
    row_count: Index


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    component: ClientComponentName
    raw_files: tuple[RawFileProvenance, ...]
    accepted_timestamp_column: TabularColumnName
    timestamp_range: TimestampRange
    duplicate_group_count: Index
    conflicting_duplicate_group_count: Index


@dataclass(frozen=True, slots=True)
class MaterializedClient:
    dataset: DatasetId
    schema: AdapterSchema
    class_manifest: LocalClassManifest
    feature_names: FeatureNames
    splits: Mapping[Split, SplitTensors]
    feature_quality: FeatureQualityReport
    class_row_counts: Mapping[FineLabel, Mapping[Split, Index]]
    provenance: DatasetProvenance


def _read_component_rows(
    paths: tuple[Path, ...],
) -> tuple[TabularColumns, RawTabularRows, tuple[RawFileProvenance, ...]]:
    frames: list[pd.DataFrame] = []
    per_file_columns: list[TabularColumns] = []
    raw_files: list[RawFileProvenance] = []
    for path in paths:
        frame = pd.read_csv(
            path,
            dtype=object,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
            dtype_backend="numpy_nullable",
        )
        observed = tuple(TabularColumnName(column) for column in frame.columns)
        if not observed:
            raise MaterializationError(f"empty selected table: {path}")
        per_file_columns.append(observed)
        frames.append(frame)
        raw_files.append(
            RawFileProvenance(RawDatasetPath(str(path)), file_sha256(path), len(frame))
        )
    columns = reconcile_component_columns(tuple(per_file_columns))
    combined = pd.concat(
        [frame.reindex(columns=list(columns)) for frame in frames],
        ignore_index=True,
    )
    column_arrays: list[RawTabularColumns] = [
        [RawCellText(cast(str, value)) for value in combined[column].to_numpy(dtype=object)]
        for column in columns
    ]
    rows: RawTabularRows = [
        dict(zip(columns, values, strict=True)) for values in zip(*column_arrays, strict=True)
    ]
    return columns, rows, tuple(raw_files)


class _LazyColumnSamples(Mapping[TabularColumnName, tuple[RawCellText, ...]]):
    def __init__(self, rows: RawTabularRows, columns: TabularColumns) -> None:
        self._rows = rows
        self._columns = columns
        self._cache: dict[TabularColumnName, tuple[RawCellText, ...]] = OrderedDict()

    def __getitem__(self, key: TabularColumnName) -> tuple[RawCellText, ...]:
        if key not in self._cache:
            self._cache[key] = tuple(row.get(key, RawCellText("")) for row in self._rows)
        return self._cache[key]

    def __iter__(self):
        return iter(self._columns)

    def __len__(self) -> Index:
        return len(self._columns)


def _resolve_schema(
    dataset: DatasetId,
    columns: TabularColumns,
    rows: RawTabularRows,
) -> AdapterSchema:
    adapter = ton_iot_adapter(dataset)
    samples = ObservedColumnSamples(_LazyColumnSamples(rows, columns))
    return adapter.resolve_schema(
        columns,
        timestamp_parse_success_fraction=1.0,
        timestamp_alias_minimum=active_config().scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum,
        observed_value_samples=samples,
    )


def _parse_epoch_seconds(value: RawCellText) -> TimestampSeconds:
    try:
        seconds: TimestampSeconds = float(value)
        return seconds
    except ValueError as error:
        raise MaterializationError(f"unparseable event-time cell: {value!r}") from error


def _build_normalized_rows(
    schema: AdapterSchema,
    rows: RawTabularRows,
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
        normalized_label = normalize_label(DatasetLabel(row[label_column]))
        timestamp = _parse_epoch_seconds(row[timestamp_column])
        values: OrderedDict[TabularColumnName, FeatureValue] = OrderedDict()
        for column in behavioral:
            is_categorical = column in categorical_columns
            values[column] = normalize_value(row.get(column, RawCellText("")), is_categorical)
        normalized_rows.append(
            NormalizedRow(
                features=NormalizedFeatureVector(values),
                label=normalized_label,
                timestamp_fraction=timestamp,
                group_id=NormalizedGroupIdentifier(""),
            )
        )
    return tuple(normalized_rows)


def _retained_local_classes(rows: tuple[NormalizedRow, ...]) -> LocalClassManifest:
    minimum = (
        active_config().scientific.transfer_support.local_prediction_attack_class_total_rows_minimum
    )
    counts: Counter[FineLabel] = Counter(row.label for row in rows)
    retained = sorted(
        label
        for label, count in counts.items()
        if label == FineLabel(NORMAL_LABEL) or count >= minimum
    )
    excluded = tuple(
        sorted((label, count) for label, count in counts.items() if label not in retained)
    )
    if FineLabel(NORMAL_LABEL) not in retained:
        raise MaterializationError("retained local class set does not contain Normal")
    return LocalClassManifest(tuple(retained), excluded)


@dataclass(frozen=True, slots=True)
class SplitAssignmentResult:
    buckets: Mapping[Split, tuple[NormalizedRow, ...]]
    duplicate_group_count: Index
    conflicting_duplicate_group_count: Index


def _assign_splits(
    schema: AdapterSchema,
    rows: tuple[NormalizedRow, ...],
    manifest: LocalClassManifest,
) -> SplitAssignmentResult:
    buckets: dict[Split, list[NormalizedRow]] = OrderedDict((split, []) for split in Split)
    by_class: defaultdict[FineLabel, list[NormalizedRow]] = defaultdict(list)
    for row in rows:
        if row.label in manifest.class_names:
            by_class[row.label].append(row)
    duplicate_group_count = 0
    conflicting_duplicate_group_count = 0
    for class_rows in by_class.values():
        normalized = normalize_and_split_training_rows(schema, tuple(class_rows))
        for group_sha256, members in normalized.duplicate_groups.groups:
            duplicate_group_count += 1
            if len({member.label for member in members}) > 1:
                conflicting_duplicate_group_count += 1
            split = normalized.split_assignment.split_of(
                DuplicateGroupId(DuplicateGroupIdentifier(group_sha256))
            )
            if split is None:
                raise MaterializationError("duplicate group received no split assignment")
            buckets[split].extend(members)
    frozen_buckets = OrderedDict((split, tuple(rows)) for split, rows in buckets.items())
    return SplitAssignmentResult(
        frozen_buckets, duplicate_group_count, conflicting_duplicate_group_count
    )


def _numeric_array(rows: Sequence[NormalizedRow], column: TabularColumnName) -> np.ndarray:
    return np.array([row.features.value_of(column) for row in rows], dtype=object)


def _categorical_array(
    rows: Sequence[NormalizedRow], column: TabularColumnName
) -> tuple[RawCellText, ...]:
    return tuple(RawCellText(str(row.features.value_of(column))) for row in rows)


def _missing_indicator(rows: Sequence[NormalizedRow], column: TabularColumnName) -> np.ndarray:
    indicator = np.zeros(len(rows), dtype=np.float32)
    for index, row in enumerate(rows):
        value = row.features.value_of(column)
        numeric = float("nan") if value is None or isinstance(value, str) else float(value)
        is_missing = not (
            np.isfinite(numeric) or numeric_zero_is_not_missing(NumericFeatureValue(numeric))
        )
        indicator[index] = 1.0 if is_missing else 0.0
    return indicator


def require_safe_memory_budget(dataset: DatasetId, paths: tuple[Path, ...]) -> None:
    raw_bytes: ByteCount = sum(path.stat().st_size for path in paths)
    estimate = estimate_memory_budget(raw_bytes)
    if not estimate.within_budget:
        raise MaterializationResourceLimitError(
            f"{dataset.value} materialization is estimated to need "
            f"{estimate.estimated_peak_bytes / 1e9:.1f} GB, exceeding the safe budget of "
            f"{estimate.budget_bytes / 1e9:.1f} GB "
            f"({estimate.available_bytes / 1e9:.1f} GB currently available); "
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
    columns, raw_rows, raw_files = _read_component_rows(paths)
    schema = _resolve_schema(dataset, columns, raw_rows)
    rows = _build_normalized_rows(schema, raw_rows)
    del raw_rows
    assert schema.timestamp_column is not None
    timestamp_column_array = np.fromiter(
        (row.timestamp_fraction for row in rows), dtype=np.float64, count=len(rows)
    )
    timestamp_lower: TimestampSeconds = float(timestamp_column_array.min())
    timestamp_upper: TimestampSeconds = float(timestamp_column_array.max())
    timestamp_range: TimestampRange = (timestamp_lower, timestamp_upper)
    manifest = _retained_local_classes(rows)
    split_result = _assign_splits(schema, rows, manifest)
    buckets = split_result.buckets
    del rows
    provenance = DatasetProvenance(
        component=component.component_name,
        raw_files=raw_files,
        accepted_timestamp_column=schema.timestamp_column,
        timestamp_range=timestamp_range,
        duplicate_group_count=split_result.duplicate_group_count,
        conflicting_duplicate_group_count=split_result.conflicting_duplicate_group_count,
    )
    class_counts_by_split = OrderedDict(
        (split, Counter(row.label for row in buckets[split])) for split in Split
    )
    class_row_counts = OrderedDict(
        (
            label,
            OrderedDict((split, class_counts_by_split[split][label]) for split in Split),
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
    numeric_preprocessors: OrderedDict[TabularColumnName, NumericPreprocessor] = OrderedDict()
    categorical_preprocessors: OrderedDict[TabularColumnName, CategoricalPreprocessor] = (
        OrderedDict()
    )
    missing_indicator_columns: list[TabularColumnName] = []
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
    feature_names: FeatureNames = (
        *(FeatureName(name) for name in numeric_preprocessors),
        *(FeatureName(f"{name}__missing") for name in missing_indicator_columns),
        *(
            FeatureName(f"{name}={category}")
            for name in categorical_preprocessors
            for category in categorical_preprocessors[name].vocabulary
        ),
    )
    splits: dict[Split, SplitTensors] = OrderedDict()
    class_indices = OrderedDict((label, manifest.index_of(label)) for label in manifest.class_names)
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
        targets = np.fromiter(
            (class_indices[row.label] for row in split_rows),
            dtype=np.int64,
            count=len(split_rows),
        )
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
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class TransferConceptGroup:
    concept: OracleTransferConcept
    native_class_indices: tuple[ClassIndex, ...]
    train_support: Index
    meta_support: Index
    source_eligible: bool


def transfer_concept_groups(
    dataset: DatasetId,
    materialized: MaterializedClient,
) -> tuple[TransferConceptGroup, ...]:
    grouped: OrderedDict[OracleTransferConcept, list[ClassIndex]] = OrderedDict()
    for index, label in enumerate(materialized.class_manifest.class_names):
        concept = transfer_concept_for(dataset, FineLabel(label))
        if concept is not None:
            grouped.setdefault(concept, []).append(ClassIndex(index))
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
