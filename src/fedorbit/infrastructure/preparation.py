from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import torch

from fedorbit.config.loading import raw_dataset_root
from fedorbit.datasets.common import (
    DatasetInspectionRequest,
    DatasetObservation,
    DatasetObservationPersistenceRequest,
    inspect_dataset,
    persist_dataset_observation,
)
from fedorbit.datasets.materialization import (
    MaterializationError,
    MaterializationResourceLimitError,
    MaterializedClient,
    materialize_client,
    transfer_concept_groups,
)
from fedorbit.infrastructure.artifacts import (
    ExecutionError,
    execution_store,
)
from fedorbit.infrastructure.manifests import (
    DatasetManifest,
    FeatureQualityManifest,
)
from fedorbit.infrastructure.provenance import (
    implementation_fingerprint,
)
from fedorbit.infrastructure.runtime import (
    execution_logger,
)
from fedorbit.infrastructure.storage import atomic_write_json
from fedorbit.infrastructure.workspace import (
    RawDuplicateReportRequest,
    RawInventoryPersistenceRequest,
    RawInventoryRequest,
    WorkspaceLayout,
    build_layout,
    experiment_workspace,
    inspect_raw_inventory,
    persist_raw_duplicate_report,
    persist_raw_inventory,
    promote_parquet,
)
from fedorbit.types import (
    ArtifactPath,
    ArtifactSchemaVersion,
    ArtifactState,
    ConfigurationSection,
    DatasetId,
    DatasetPreprocessingState,
    ExperimentName,
    InvalidReason,
    OverwritePolicy,
    ProducerModuleName,
    ResourceLimitReason,
    SampleCount,
    Sha256Digest,
    StableJsonPayload,
    StorageLayoutSegment,
)

_MODULE_NAME = ProducerModuleName("fedorbit.infrastructure.preparation")


@dataclass(frozen=True, slots=True)
class DatasetPreparationRequest:
    datasets: tuple[DatasetId, ...]
    overwrite_policy: OverwritePolicy


@dataclass(frozen=True, slots=True)
class DatasetPreparationResult:
    observations: tuple[DatasetObservation, ...]
    validation_artifact_paths: tuple[ArtifactPath, ...]
    duplicate_artifact_paths: tuple[ArtifactPath, ...]
    resource_blocked_datasets: tuple[tuple[DatasetId, ResourceLimitReason], ...] = ()

    @property
    def blocked_datasets(self) -> tuple[DatasetId, ...]:
        return tuple(
            observation.dataset
            for observation in self.observations
            if not observation.valid_for_chronological_preprocessing
        )


def preprocess_datasets(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    logger = execution_logger()
    logger.event(
        "preprocess_start", #TODO: should be enum not hardcoded string
        datasets=[dataset.value for dataset in request.datasets],
        overwrite_policy=request.overwrite_policy.value,
    )
    raw_root = raw_dataset_root()
    inventories = tuple(
        inspect_raw_inventory(RawInventoryRequest(dataset, raw_root))
        for dataset in request.datasets
    )
    if len(inventories) != len(request.datasets):
        raise ExecutionError("raw inventory collection did not cover every requested dataset")
    store = execution_store()
    persisted_inventory_paths = tuple(
        persist_raw_inventory(
            RawInventoryPersistenceRequest(
                inventory, store.root / StorageLayoutSegment.PREPROCESSING
            )
        )
        for inventory in inventories
    )
    if len(persisted_inventory_paths) != len(inventories):
        raise ExecutionError("raw inventory persistence did not cover every requested dataset")
    observations = tuple(
        inspect_dataset(DatasetInspectionRequest(dataset, raw_root)) for dataset in request.datasets
    )
    if len(observations) != len(request.datasets):
        raise ExecutionError("dataset observation collection did not cover every requested dataset")
    validation_paths = tuple(
        persist_dataset_observation(
            DatasetObservationPersistenceRequest(
                observation, store.root / StorageLayoutSegment.PREPROCESSING
            )
        )
        for observation in observations
    )
    duplicate_paths = tuple(
        persist_raw_duplicate_report(
            RawDuplicateReportRequest(
                dataset, raw_root, store.root / StorageLayoutSegment.PREPROCESSING
            )
        )
        for dataset in request.datasets
    )
    if len(duplicate_paths) != len(request.datasets):
        raise ExecutionError("duplicate diagnostics did not cover every requested dataset")
    layout = build_layout()
    resource_blocked: list[tuple[DatasetId, ResourceLimitReason]] = []
    for observation in observations:
        if not observation.valid_for_chronological_preprocessing:
            continue
        try:
            materialized = materialize_client(observation.dataset, raw_root)
        except MaterializationResourceLimitError as error:
            resource_blocked.append((observation.dataset, ResourceLimitReason(str(error))))
            continue
        except MaterializationError as error:
            raise ExecutionError(
                f"could not materialize {observation.dataset.value}: {error}"
            ) from error
        persist_materialized_client(layout, materialized, request.overwrite_policy)
    logger.event(
        "preprocess_end", #TODO: should be enum not hardcoded string
        datasets=[dataset.value for dataset in request.datasets],
        blocked=len(resource_blocked),
    )
    return DatasetPreparationResult(
        observations=observations,
        validation_artifact_paths=tuple(ArtifactPath(path) for path in validation_paths),
        duplicate_artifact_paths=tuple(ArtifactPath(path) for path in duplicate_paths),
        resource_blocked_datasets=tuple(resource_blocked),
    )


BASE_MODEL_PILOT_CONFIGURATION_SECTIONS = frozenset({ConfigurationSection.MODELS})


def build_dataset_manifest(materialized: MaterializedClient) -> DatasetManifest:
    provenance = materialized.provenance
    raw_files = tuple(entry.path for entry in provenance.raw_files)
    raw_sha256 = Sha256Digest(
        hashlib.sha256(
            ",".join(f"{entry.path}:{entry.sha256}" for entry in provenance.raw_files).encode(
                "utf-8"
            )
        ).hexdigest()
    )
    raw_counts = OrderedDict((entry.path, entry.row_count) for entry in provenance.raw_files)
    adapter_feature_roles = OrderedDict(
        (column, materialized.schema.role_of(column).value)
        for column in materialized.schema.feature_order
    )
    local_class_counts = OrderedDict(
        (label, sum(counts.values())) for label, counts in materialized.class_row_counts.items()
    )
    transfer_candidate_counts: Mapping[str, SampleCount] = OrderedDict( #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        (str(group.concept.value), group.train_support)
        for group in transfer_concept_groups(materialized.dataset, materialized)
    )
    feature_quality = FeatureQualityManifest(
        dropped_feature_count=materialized.feature_quality.dropped_feature_count,
        candidate_count_before_filtering=(
            materialized.feature_quality.candidate_count_before_filtering
        ),
        client_invalid=materialized.feature_quality.client_invalid,
        client_invalid_reason=materialized.feature_quality.client_invalid_reason,
    )
    dependency_fingerprint_sha256 = Sha256Digest(
        hashlib.sha256(
            f"{raw_sha256}|{','.join(materialized.schema.feature_order)}".encode()
        ).hexdigest()
    )
    return DatasetManifest.model_validate(
        OrderedDict(
            dataset=materialized.dataset,
            component=provenance.component,
            raw_files=raw_files,
            raw_sha256=raw_sha256,
            raw_counts=raw_counts,
            schema=ArtifactSchemaVersion("1.0"), #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
            adapter_feature_order=materialized.schema.feature_order,
            adapter_feature_roles=adapter_feature_roles,
            accepted_schema_aliases=(provenance.accepted_timestamp_column,),
            adapter_adaptations=(),
            timestamp_field=provenance.accepted_timestamp_column,
            timestamp_range=provenance.timestamp_range,
            duplicate_counts=OrderedDict(total=provenance.duplicate_group_count),
            conflicting_duplicate_counts=OrderedDict(
                total=provenance.conflicting_duplicate_group_count
            ),
            local_class_counts=local_class_counts,
            transfer_candidate_counts=transfer_candidate_counts,
            feature_quality=feature_quality,
            preprocessing_state=DatasetPreprocessingState.MATERIALIZED,
            dependency_fingerprint_sha256=dependency_fingerprint_sha256,
            producer_code_sha256=Sha256Digest(
                implementation_fingerprint(ProducerModuleName("fedorbit.datasets.materialization"))
            ),
        )
    )


def persist_dataset_manifest(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    dataset: DatasetId,
    manifest: DatasetManifest,
) -> Path:
    destination = (
        experiment_workspace(layout, experiment)
        / "artifacts" #TODO: should be enums not hardcoded strings
        / "derived" #TODO: should be enums not hardcoded strings
        / f"dataset-manifest.{dataset.value}.json" #TODO: should be enums not hardcoded strings
    )
    atomic_write_json(destination, manifest.model_dump(mode="json"))
    return destination


def persist_materialized_client(
    layout: WorkspaceLayout,
    materialized: MaterializedClient,
    overwrite_policy: OverwritePolicy,
) -> tuple[Path, ...]:
    dataset = materialized.dataset
    split_paths: list[Path] = []
    for split, tensors in materialized.splits.items():
        destination = layout.preprocessing / "splits" / dataset.value / split.value / "data.parquet" #TODO: should be enums not hardcoded strings
        if overwrite_policy == OverwritePolicy.REUSE and destination.is_file():
            split_paths.append(destination)
            continue
        frame = pd.DataFrame(
            tensors.features.detach().cpu().numpy(),
            columns=materialized.feature_names,
        )
        frame.insert(len(frame.columns), "target", tensors.targets.detach().cpu().numpy()) #TODO: should be enums not hardcoded strings
        destination.parent.mkdir(parents=True, exist_ok=True)
        promote_parquet(frame, destination.parent, destination.name)
        split_paths.append(destination)
    manifest = build_dataset_manifest(materialized)
    atomic_write_json(
        layout.preprocessing / "prepared" / dataset.value / "data.json", #TODO: should be enums not hardcoded strings
        manifest.model_dump(mode="json"),
    )
    eligibility = tuple(
        cast(
            StableJsonPayload,
            OrderedDict(
                concept=group.concept.value,
                native_class_indices=group.native_class_indices,
                train_support=group.train_support,
                meta_support=group.meta_support,
                source_eligible=group.source_eligible,
            ),
        )
        for group in transfer_concept_groups(dataset, materialized)
    )
    atomic_write_json(
        layout.preprocessing / "features" / dataset.value / "data.json", #TODO: should be enums not hardcoded strings
        cast(
            StableJsonPayload,
            OrderedDict(
                feature_names=materialized.feature_names,
                local_class_names=materialized.class_manifest.class_names,
                excluded_classes=materialized.class_manifest.excluded_classes,
                transfer_eligibility=eligibility,
            ),
        ),
    )
    torch.save(
        materialized,
        layout.preprocessing / "prepared" / dataset.value / "client.pt", #TODO: should be enums not hardcoded strings
    )
    return tuple(split_paths)


def persist_client_invalid(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    dataset: DatasetId,
    reason: InvalidReason,
) -> None:
    destination = experiment_workspace(layout, experiment) / "artifacts" / "derived" #TODO: should be enums not hardcoded strings
    payload = cast(
        StableJsonPayload,
        OrderedDict(
            experiment=experiment.value,
            dataset=dataset.value,
            state=ArtifactState.INVALID.value,
            reason=reason,
        ),
    )
    atomic_write_json(destination / f"{dataset.value}-invalid.json", payload) #TODO: should be enums not hardcoded strings


def load_or_materialize_client(
    dataset: DatasetId, raw_root: Path, layout: WorkspaceLayout
) -> MaterializedClient:
    prepared = layout.preprocessing / "prepared" / dataset.value / "client.pt" #TODO: should be enums not hardcoded strings
    if prepared.is_file():
        loaded = torch.load(prepared, map_location="cpu", weights_only=False)
        if isinstance(loaded, MaterializedClient):
            execution_logger().event(
                "prepared_client_load", #TODO: should be enum not hardcoded string
                dataset=dataset.value,
                artifact_path=str(prepared),
            )
            return loaded
    execution_logger().event("prepared_client_miss", dataset=dataset.value) #TODO: should be enum not hardcoded string
    materialized = materialize_client(dataset, raw_root)
    if materialized.feature_quality.candidate_count_before_filtering > 0:
        persist_materialized_client(layout, materialized, OverwritePolicy.REUSE)
    return materialized
