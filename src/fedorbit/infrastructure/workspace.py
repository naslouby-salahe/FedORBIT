from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import cast

import pandas as pd

from fedorbit.config.loading import active_config, repository_root
from fedorbit.datasets.common import PreprocessingObservationArtifact
from fedorbit.datasets.edge_iiotset.loader import inspect_edge_tabular_files
from fedorbit.datasets.ton_iot.components import component_for
from fedorbit.datasets.ton_iot.loader import inspect_ton_iot_component_files
from fedorbit.infrastructure.storage import (
    promote_json,
    promote_table_payload,
)
from fedorbit.types import (
    ByteCount,
    ClientComponentName,
    DatasetId,
    DatasetRelativePath,
    DatasetReleaseIdentity,
    ExperimentName,
    FilesystemSlug,
    RawDatasetDirectory,
    RawInventoryArtifact,
    ReportColumnName,
    Sha256Digest,
    Split,
    StableJsonPayload,
    StorageLayoutSegment,
    TabularColumnName,
    stable_json,
)


class WorkspaceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    execution_root: Path
    preprocessing: Path
    artifacts: Path
    experiments: Path
    cache: Path
    staging: Path
    results_experiments: Path
    project_summary: Path


def safe_slug(
    value: str,
) -> FilesystemSlug:
    normalized = unicodedata.normalize("NFC", value).casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    if not slug:
        raise WorkspaceError("descriptive name does not produce a filesystem slug")
    return FilesystemSlug(slug)


def build_layout(root: Path | None = None) -> WorkspaceLayout:
    layout = active_config().runtime.artifact_layout
    base = root if root is not None else repository_root()
    execution_root = base / layout.execution_root
    manuscript_root = base / layout.manuscript_root
    return WorkspaceLayout(
        execution_root=execution_root,
        preprocessing=execution_root / layout.preprocessing_directory,
        artifacts=execution_root / layout.artifacts_directory,
        experiments=execution_root / layout.experiments_directory,
        cache=execution_root / layout.cache_directory,
        staging=execution_root / layout.cache_directory / layout.staging_directory,
        results_experiments=manuscript_root / layout.results_experiments_directory,
        project_summary=manuscript_root / layout.project_summary_directory,
    )


def experiment_workspace(layout: WorkspaceLayout, experiment: ExperimentName) -> Path:
    return layout.experiments / safe_slug(experiment.value)


def results_workspace(layout: WorkspaceLayout, experiment: ExperimentName) -> Path:
    return layout.results_experiments / safe_slug(experiment.value)


class PreprocessingDirectorySegment(StrEnum):
    SPLITS = auto()


class PreprocessingPayloadFileName(StrEnum):
    DATASET_MANIFEST = "data.json"
    MATERIALIZED_CLIENT = "client.pt"
    SPLIT_PAYLOAD = "data.parquet"


class DatasetManifestFileName(StrEnum):
    PREFIX = "dataset-manifest"
    SUFFIX = "json"


class ExperimentLogDirectory(StrEnum):
    LOGS = auto()
    EXECUTION = auto()
    FAILURES = auto()


class ExperimentLogFileName(StrEnum):
    EVENTS = "events.jsonl"


def prepared_dataset_directory(layout: WorkspaceLayout, dataset: DatasetId) -> Path:
    return layout.preprocessing / StorageLayoutSegment.PREPARED / dataset.value


def feature_dataset_directory(layout: WorkspaceLayout, dataset: DatasetId) -> Path:
    return layout.preprocessing / StorageLayoutSegment.FEATURES / dataset.value


def split_dataset_directory(
    layout: WorkspaceLayout,
    dataset: DatasetId,
    split: Split,
) -> Path:
    return layout.preprocessing / PreprocessingDirectorySegment.SPLITS / dataset.value / split.value


def dataset_manifest_path(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    dataset: DatasetId,
) -> Path:
    return (
        experiment_workspace(layout, experiment)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
        / f"{DatasetManifestFileName.PREFIX.value}.{dataset.value}."
        f"{DatasetManifestFileName.SUFFIX.value}"
    )


def experiment_event_log_path(
    execution_root: Path,
    experiment: ExperimentName,
    log_directory: ExperimentLogDirectory,
) -> Path:
    return (
        execution_root
        / StorageLayoutSegment.EXPERIMENTS
        / safe_slug(experiment.value)
        / ExperimentLogDirectory.LOGS
        / log_directory
        / ExperimentLogFileName.EVENTS
    )


class RawInventoryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RawInventoryRequest:
    dataset: DatasetId
    raw_root: Path
    release_identity: DatasetReleaseIdentity | None = None


@dataclass(frozen=True, slots=True)
class RawFileInventory:
    relative_path: DatasetRelativePath
    byte_size: ByteCount
    sha256: Sha256Digest
    columns: tuple[TabularColumnName, ...]


@dataclass(frozen=True, slots=True)
class RawDatasetInventory:
    dataset: DatasetId
    files: tuple[RawFileInventory, ...]
    component: ClientComponentName | None = None
    release_identity: DatasetReleaseIdentity | None = None

    def __post_init__(self) -> None:
        if not self.files:
            raise RawInventoryError("raw dataset inventory requires at least one file")

    def release_payload(self) -> StableJsonPayload:
        identity = self.release_identity
        if identity is None:
            return cast(StableJsonPayload, None)
        return cast(
            StableJsonPayload,
            OrderedDict[str, StableJsonPayload](
                release=identity.release,
                acquisition_source=identity.acquisition_source,
                acquisition_timestamp=identity.acquisition_timestamp,
                license_note=identity.license_note,
            ),
        )

    def fingerprint(self) -> Sha256Digest:
        return Sha256Digest(
            hashlib.sha256(stable_json(self.serialization_payload()).encode("utf-8")).hexdigest()
        )

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
                component=self.component,
                release_identity=self.release_payload(),
                files=file_entries,
            ),
        )


@dataclass(frozen=True, slots=True)
class RawInventoryPersistenceRequest:
    inventory: RawDatasetInventory
    preprocessing_root: Path
    staging_root: Path | None = None


@dataclass(frozen=True, slots=True)
class RawDuplicateReportRequest:
    dataset: DatasetId
    raw_root: Path
    preprocessing_root: Path
    staging_root: Path | None = None


def execution_staging_root(execution_root: Path) -> Path:
    layout = active_config().runtime.artifact_layout
    return execution_root / layout.cache_directory / layout.staging_directory


def request_staging_root(preprocessing_root: Path, staging_root: Path | None) -> Path:
    if staging_root is not None:
        return staging_root
    return execution_staging_root(preprocessing_root.parent)


def promote_parquet(
    frame: pd.DataFrame,
    destination: Path,
    filename: str,
    staging_root: Path,
    sort_columns: Sequence[TabularColumnName] | None = None,
) -> Path:
    return promote_table_payload(destination / filename, frame, sort_columns, staging_root)


def persist_raw_inventory(request: RawInventoryPersistenceRequest) -> Path:
    destination = (
        request.preprocessing_root
        / RawInventoryArtifact.INVENTORIES
        / request.inventory.dataset.value
    )
    staging_root = request_staging_root(request.preprocessing_root, request.staging_root)
    promote_json(
        destination / RawInventoryArtifact.MANIFEST_JSON,
        request.inventory.serialization_payload(),
        staging_root,
    )
    promote_json(
        destination / RawInventoryArtifact.CHECKSUMS_JSON,
        cast(
            StableJsonPayload,
            OrderedDict((entry.relative_path, entry.sha256) for entry in request.inventory.files),
        ),
        staging_root,
    )
    promote_json(
        destination / RawInventoryArtifact.SCHEMA_JSON,
        cast(
            StableJsonPayload,
            OrderedDict(
                (entry.relative_path, list(entry.columns)) for entry in request.inventory.files
            ),
        ),
        staging_root,
    )
    frame = pd.DataFrame(
        OrderedDict(
            relative_path=[entry.relative_path for entry in request.inventory.files],
            byte_size=[entry.byte_size for entry in request.inventory.files],
            sha256=[entry.sha256 for entry in request.inventory.files],
            columns=[list(entry.columns) for entry in request.inventory.files],
        )
    )
    promote_parquet(
        frame,
        destination,
        "files.parquet",
        staging_root,
        (TabularColumnName("relative_path"),),
    )
    return destination / RawInventoryArtifact.MANIFEST_JSON


def persist_raw_duplicate_report(request: RawDuplicateReportRequest) -> Path:
    destination = (
        request.preprocessing_root / StorageLayoutSegment.VALIDATION / request.dataset.value
    )
    occurrence_counts: Counter[Sha256Digest] = Counter()
    for source in _selected_raw_paths(request.dataset, request.raw_root):
        with source.open("rb") as handle:
            if not handle.readline().strip():
                raise RawInventoryError(f"empty selected table: {source}")
            for row in handle:
                if row.strip():
                    occurrence_counts[Sha256Digest(hashlib.sha256(row).hexdigest())] += 1
    duplicate_rows = tuple(
        (row_sha256, count, count - 1)
        for row_sha256, count in sorted(occurrence_counts.items())
        if count > 1
    )
    frame = pd.DataFrame(
        duplicate_rows,
        columns=(
            ReportColumnName.RAW_ROW_SHA256,
            ReportColumnName.OCCURRENCE_COUNT,
            ReportColumnName.DUPLICATE_ROW_COUNT,
        ),
    )
    return promote_parquet(
        frame,
        destination,
        PreprocessingObservationArtifact.DUPLICATES,
        request_staging_root(request.preprocessing_root, request.staging_root),
        (TabularColumnName(ReportColumnName.RAW_ROW_SHA256.value),),
    )


def _selected_raw_paths(dataset: DatasetId, raw_root: Path) -> tuple[Path, ...]:
    if dataset == DatasetId.EDGE_IIOTSET_NETWORK:
        from fedorbit.datasets.edge_iiotset.loader import discover_edge_tabular_files

        return discover_edge_tabular_files(raw_root / RawDatasetDirectory.EDGE_IIOTSET)
    from fedorbit.datasets.ton_iot.loader import discover_ton_iot_component_files

    return discover_ton_iot_component_files(
        raw_root / RawDatasetDirectory.TON_IOT,
        component_for(dataset),
    )


def inspect_raw_inventory(request: RawInventoryRequest) -> RawDatasetInventory:
    if request.dataset == DatasetId.EDGE_IIOTSET_NETWORK:
        inspected = inspect_edge_tabular_files(request.raw_root / RawDatasetDirectory.EDGE_IIOTSET)
    else:
        inspected = inspect_ton_iot_component_files(
            request.raw_root / RawDatasetDirectory.TON_IOT, component_for(request.dataset)
        )
    return RawDatasetInventory(
        dataset=request.dataset,
        component=(
            ClientComponentName(request.dataset.value)
            if request.dataset == DatasetId.EDGE_IIOTSET_NETWORK
            else ClientComponentName(component_for(request.dataset).component_name)
        ),
        release_identity=request.release_identity,
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
