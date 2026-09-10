from __future__ import annotations

import hashlib
import os
import re
import tempfile
import unicodedata
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from filelock import FileLock

from fedorbit.config.loading import active_config, repository_root
from fedorbit.datasets.edge_iiotset.loader import inspect_edge_tabular_files
from fedorbit.datasets.ton_iot.components import component_for
from fedorbit.datasets.ton_iot.loader import inspect_ton_iot_component_files
from fedorbit.infrastructure.storage import atomic_write_json
from fedorbit.types import (
    ArtifactFileSuffix,
    ByteCount,
    DatasetId,
    DatasetRelativePath,
    DuplicateReportColumn,
    ExperimentName,
    FilesystemSlug,
    RawDatasetDirectory,
    RawInventoryArtifact,
    SemanticCoordinates,
    Sha256Digest,
    StableJsonPayload,
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


def safe_slug(value: str) -> FilesystemSlug:
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


def leaf_path(
    layout: WorkspaceLayout,
    workspace: Path,
    semantic_coordinates: SemanticCoordinates,
    fingerprint_sha256: Sha256Digest,
    suffix: ArtifactFileSuffix,
) -> Path:
    if not workspace.is_absolute():
        workspace = layout.execution_root / workspace
    semantic_slug = safe_slug(semantic_coordinates.value)
    return workspace / f"{semantic_slug}.{fingerprint_sha256[:16]}{suffix}"


class RawInventoryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RawInventoryRequest:
    dataset: DatasetId
    raw_root: Path


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

    def __post_init__(self) -> None:
        if not self.files:
            raise RawInventoryError("raw dataset inventory requires at least one file")

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
                files=file_entries,
            ),
        )


@dataclass(frozen=True, slots=True)
class RawInventoryPersistenceRequest:
    inventory: RawDatasetInventory
    preprocessing_root: Path


@dataclass(frozen=True, slots=True)
class RawDuplicateReportRequest:
    dataset: DatasetId
    raw_root: Path
    preprocessing_root: Path


def promote_parquet(frame: pd.DataFrame, destination: Path, filename: str) -> Path:
    target = destination / filename
    with FileLock(str(destination / f"{filename}.lock")):
        descriptor, temporary_name = tempfile.mkstemp(dir=destination, suffix=".parquet")
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def persist_raw_inventory(request: RawInventoryPersistenceRequest) -> Path:
    destination = (
        request.preprocessing_root
        / RawInventoryArtifact.INVENTORIES
        / request.inventory.dataset.value
    )
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        destination / RawInventoryArtifact.MANIFEST_JSON,
        request.inventory.serialization_payload(),
    )
    atomic_write_json(
        destination / RawInventoryArtifact.CHECKSUMS_JSON,
        cast(
            StableJsonPayload,
            OrderedDict((entry.relative_path, entry.sha256) for entry in request.inventory.files),
        ),
    )
    atomic_write_json(
        destination / RawInventoryArtifact.SCHEMA_JSON,
        cast(
            StableJsonPayload,
            OrderedDict(
                (entry.relative_path, list(entry.columns)) for entry in request.inventory.files
            ),
        ),
    )
    frame = pd.DataFrame(
        OrderedDict(
            relative_path=[entry.relative_path for entry in request.inventory.files],
            byte_size=[entry.byte_size for entry in request.inventory.files],
            sha256=[entry.sha256 for entry in request.inventory.files],
            columns=[list(entry.columns) for entry in request.inventory.files],
        )
    )
    promote_parquet(frame, destination, "files.parquet")
    return destination / RawInventoryArtifact.MANIFEST_JSON


def persist_raw_duplicate_report(request: RawDuplicateReportRequest) -> Path:
    destination = request.preprocessing_root / "validation" / request.dataset.value
    destination.mkdir(parents=True, exist_ok=True)
    occurrence_counts: Counter[str] = Counter()
    for source in _selected_raw_paths(request.dataset, request.raw_root):
        with source.open("rb") as handle:
            if not handle.readline().strip():
                raise RawInventoryError(f"empty selected table: {source}")
            for row in handle:
                if row.strip():
                    occurrence_counts[hashlib.sha256(row).hexdigest()] += 1
    duplicate_rows = tuple(
        (row_sha256, count, count - 1)
        for row_sha256, count in sorted(occurrence_counts.items())
        if count > 1
    )
    frame = pd.DataFrame(
        duplicate_rows,
        columns=(
            DuplicateReportColumn.RAW_ROW_SHA256,
            DuplicateReportColumn.OCCURRENCE_COUNT,
            DuplicateReportColumn.DUPLICATE_ROW_COUNT,
        ),
    )
    return promote_parquet(frame, destination, "duplicates.parquet")


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
