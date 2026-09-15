from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from pydantic import JsonValue, TypeAdapter

from fedorbit.config.loading import active_config, raw_dataset_root
from fedorbit.datasets.common import (
    DatasetInspectionRequest,
    DatasetObservation,
    DatasetObservationPersistenceRequest,
    PreprocessingObservationArtifact,
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
from fedorbit.infrastructure.storage import (
    atomic_write_json,
    promote_json,
    promote_staged,
    staged_payload_path,
)
from fedorbit.infrastructure.workspace import (
    PreprocessingPayloadFileName,
    RawDuplicateReportRequest,
    RawInventoryPersistenceRequest,
    RawInventoryRequest,
    WorkspaceLayout,
    build_layout,
    dataset_manifest_path,
    experiment_workspace,
    feature_dataset_directory,
    inspect_raw_inventory,
    persist_raw_duplicate_report,
    persist_raw_inventory,
    prepared_dataset_directory,
    promote_parquet,
    split_dataset_directory,
)
from fedorbit.types import (
    ArtifactPath,
    ArtifactSchemaVersion,
    ArtifactState,
    ConfigurationSection,
    DatasetId,
    DatasetPreprocessingState,
    ExecutionEventName,
    ExperimentName,
    ImplementationIdentity,
    InvalidReason,
    OracleTransferConcept,
    OverwritePolicy,
    ResourceLimitReason,
    SampleCount,
    Sha256Digest,
    StableJsonPayload,
    StorageLayoutSegment,
    ValidationReason,
    stable_json,
)


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
    invalid_datasets: tuple[tuple[DatasetId, ValidationReason], ...] = ()

    @property
    def blocked_datasets(self) -> tuple[DatasetId, ...]:
        return tuple(
            observation.dataset
            for observation in self.observations
            if not observation.valid_for_chronological_preprocessing
        )


_PREPARATION_RECORD_FILENAME = "preparation.json"
_OBSERVATION_ADAPTER: TypeAdapter[DatasetObservation] = TypeAdapter(DatasetObservation)


def _preparation_record_path(layout: WorkspaceLayout, dataset: DatasetId) -> Path:
    return (
        layout.preprocessing
        / StorageLayoutSegment.METADATA
        / dataset.value
        / _PREPARATION_RECORD_FILENAME
    )


def _preparation_contract_sha256() -> Sha256Digest:
    configuration = active_config().model_dump(mode="json")
    payload = cast(
        StableJsonPayload,
        OrderedDict(
            implementation_identity=ImplementationIdentity.DATASET_MATERIALIZATION_V3,
            configuration=configuration,
        ),
    )
    return Sha256Digest(hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest())


def _prepared_payloads_exist(layout: WorkspaceLayout, dataset: DatasetId) -> bool:
    return all(
        path.is_file()
        for path in (
            prepared_dataset_directory(layout, dataset)
            / PreprocessingPayloadFileName.DATASET_MANIFEST,
            prepared_dataset_directory(layout, dataset)
            / PreprocessingPayloadFileName.MATERIALIZED_CLIENT,
            feature_dataset_directory(layout, dataset)
            / PreprocessingPayloadFileName.DATASET_MANIFEST,
        )
    )


def load_cached_preparation(
    layout: WorkspaceLayout,
    inventory_fingerprint: Sha256Digest,
    contract_sha256: Sha256Digest,
    dataset: DatasetId,
) -> (
    tuple[
        DatasetObservation,
        ArtifactPath,
        ArtifactPath,
        ResourceLimitReason | None,
        ValidationReason | None,
    ]
    | None
):
    record_path = _preparation_record_path(layout, dataset)
    validation_path = (
        layout.preprocessing
        / StorageLayoutSegment.VALIDATION
        / dataset.value
        / PreprocessingObservationArtifact.VALIDATION
    )
    duplicate_path = (
        layout.preprocessing
        / StorageLayoutSegment.VALIDATION
        / dataset.value
        / PreprocessingObservationArtifact.DUPLICATES
    )
    if not record_path.is_file() or not validation_path.is_file() or not duplicate_path.is_file():
        return None
    try:
        loaded = cast(Mapping[str, JsonValue], json.loads(record_path.read_text(encoding="utf-8")))
        if (
            loaded.get("raw_inventory_fingerprint") != inventory_fingerprint
            or loaded.get("preparation_contract_sha256") != contract_sha256
        ):
            return None
        observation_text = validation_path.read_text(encoding="utf-8")
        observation = _OBSERVATION_ADAPTER.validate_json(observation_text)
        state = loaded.get("state")
        if state == DatasetPreprocessingState.MATERIALIZED and not _prepared_payloads_exist(
            layout, dataset
        ):
            return None
        if state == DatasetPreprocessingState.MATERIALIZED:
            reason: ResourceLimitReason | None = None
            invalid_reason: ValidationReason | None = None
        elif state == DatasetPreprocessingState.RESOURCE_BLOCKED:
            reason_value = loaded.get("resource_block_reason")
            if not isinstance(reason_value, str):
                return None
            reason = ResourceLimitReason(reason_value)
            invalid_reason = None
        elif state == DatasetPreprocessingState.INVALID:
            reason = None
            invalid_value = loaded.get("materialization_invalid_reason")
            invalid_reason = (
                ValidationReason(invalid_value) if isinstance(invalid_value, str) else None
            )
        else:
            return None
    except (OSError, ValueError, TypeError):
        return None
    return (
        observation,
        ArtifactPath(validation_path),
        ArtifactPath(duplicate_path),
        reason,
        invalid_reason,
    )


def persist_preparation_record(
    layout: WorkspaceLayout,
    inventory_fingerprint: Sha256Digest,
    contract_sha256: Sha256Digest,
    observation: DatasetObservation,
    resource_block_reason: ResourceLimitReason | None,
    materialization_invalid_reason: ValidationReason | None = None,
) -> None:
    if resource_block_reason is not None:
        state = DatasetPreprocessingState.RESOURCE_BLOCKED
    elif (
        materialization_invalid_reason is None and observation.valid_for_chronological_preprocessing
    ):
        state = DatasetPreprocessingState.MATERIALIZED
    else:
        state = DatasetPreprocessingState.INVALID
    payload = cast(
        StableJsonPayload,
        OrderedDict(
            dataset=observation.dataset.value,
            raw_inventory_fingerprint=inventory_fingerprint,
            preparation_contract_sha256=contract_sha256,
            state=state.value,
            resource_block_reason=resource_block_reason,
            materialization_invalid_reason=materialization_invalid_reason,
        ),
    )
    atomic_write_json(_preparation_record_path(layout, observation.dataset), payload)


def preprocess_datasets(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    logger = execution_logger()
    logger.event(
        ExecutionEventName.PREPROCESS_START,
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
    layout = build_layout()
    store = execution_store()
    contract_sha256 = _preparation_contract_sha256()
    observations: list[DatasetObservation] = []
    validation_paths: list[ArtifactPath] = []
    duplicate_paths: list[ArtifactPath] = []
    resource_blocked: list[tuple[DatasetId, ResourceLimitReason]] = []
    invalid_datasets: list[tuple[DatasetId, ValidationReason]] = []
    for inventory in inventories:
        cached = (
            load_cached_preparation(
                layout, inventory.fingerprint(), contract_sha256, inventory.dataset
            )
            if request.overwrite_policy == OverwritePolicy.REUSE
            else None
        )
        if cached is not None:
            observation, validation_path, duplicate_path, resource_reason, invalid_reason = cached
            observations.append(observation)
            validation_paths.append(validation_path)
            duplicate_paths.append(duplicate_path)
            if resource_reason is not None:
                resource_blocked.append((observation.dataset, resource_reason))
            if invalid_reason is not None:
                invalid_datasets.append((observation.dataset, invalid_reason))
            continue
        persist_raw_inventory(
            RawInventoryPersistenceRequest(
                inventory, store.root / StorageLayoutSegment.PREPROCESSING
            )
        )
        observation = inspect_dataset(DatasetInspectionRequest(inventory.dataset, raw_root))
        validation_path = persist_dataset_observation(
            DatasetObservationPersistenceRequest(
                observation, store.root / StorageLayoutSegment.PREPROCESSING
            )
        )
        duplicate_path = persist_raw_duplicate_report(
            RawDuplicateReportRequest(
                inventory.dataset, raw_root, store.root / StorageLayoutSegment.PREPROCESSING
            )
        )
        observations.append(observation)
        validation_paths.append(ArtifactPath(validation_path))
        duplicate_paths.append(ArtifactPath(duplicate_path))
        resource_reason: ResourceLimitReason | None = None
        invalid_reason: ValidationReason | None = None
        if not observation.valid_for_chronological_preprocessing:
            persist_preparation_record(
                layout, inventory.fingerprint(), contract_sha256, observation, resource_reason
            )
            continue
        try:
            materialized = materialize_client(observation.dataset, raw_root)
        except MaterializationResourceLimitError as error:
            resource_reason = ResourceLimitReason(str(error))
            resource_blocked.append((observation.dataset, resource_reason))
            persist_preparation_record(
                layout, inventory.fingerprint(), contract_sha256, observation, resource_reason
            )
            continue
        except MaterializationError as error:
            invalid_reason = ValidationReason(str(error))
            invalid_datasets.append((observation.dataset, invalid_reason))
            persist_preparation_record(
                layout,
                inventory.fingerprint(),
                contract_sha256,
                observation,
                resource_reason,
                invalid_reason,
            )
            continue
        persist_materialized_client(layout, materialized, request.overwrite_policy)
        persist_preparation_record(
            layout, inventory.fingerprint(), contract_sha256, observation, resource_reason
        )
    logger.event(
        ExecutionEventName.PREPROCESS_END,
        datasets=[dataset.value for dataset in request.datasets],
        blocked=len(resource_blocked),
    )
    return DatasetPreparationResult(
        observations=tuple(observations),
        validation_artifact_paths=tuple(validation_paths),
        duplicate_artifact_paths=tuple(duplicate_paths),
        resource_blocked_datasets=tuple(resource_blocked),
        invalid_datasets=tuple(invalid_datasets),
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
        (column, materialized.schema.role_of(column))
        for column in materialized.schema.feature_order
    )
    local_class_counts = OrderedDict(
        (label, sum(counts.values())) for label, counts in materialized.class_row_counts.items()
    )
    transfer_candidate_counts: Mapping[OracleTransferConcept, SampleCount] = OrderedDict(
        (group.concept, group.train_support)
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
            schema=ArtifactSchemaVersion.V1,
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
                implementation_fingerprint(ImplementationIdentity.DATASET_MATERIALIZATION_V3)
            ),
        )
    )


def persist_dataset_manifest(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    dataset: DatasetId,
    manifest: DatasetManifest,
) -> Path:
    destination = dataset_manifest_path(layout, experiment, dataset)
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
        destination = (
            split_dataset_directory(layout, dataset, split)
            / PreprocessingPayloadFileName.SPLIT_PAYLOAD
        )
        if overwrite_policy == OverwritePolicy.REUSE and destination.is_file():
            split_paths.append(destination)
            continue
        frame = pd.DataFrame(
            tensors.features.detach().cpu().numpy(),
            columns=materialized.feature_names,
        )
        frame.insert(len(frame.columns), "target", tensors.targets.detach().cpu().numpy())
        promote_parquet(frame, destination.parent, destination.name, layout.staging)
        split_paths.append(destination)
    manifest = build_dataset_manifest(materialized)
    promote_json(
        prepared_dataset_directory(layout, dataset) / PreprocessingPayloadFileName.DATASET_MANIFEST,
        cast(StableJsonPayload, manifest.model_dump(mode="json")),
        layout.staging,
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
                confirm_support=group.confirm_support,
                test_support=group.test_support,
                target_eligible=group.target_eligible,
            ),
        )
        for group in transfer_concept_groups(dataset, materialized)
    )
    promote_json(
        feature_dataset_directory(layout, dataset) / PreprocessingPayloadFileName.DATASET_MANIFEST,
        cast(
            StableJsonPayload,
            OrderedDict(
                feature_names=materialized.feature_names,
                local_class_names=materialized.class_manifest.class_names,
                excluded_classes=materialized.class_manifest.excluded_classes,
                preprocessing_fit_accesses=tuple(
                    cast(
                        StableJsonPayload,
                        OrderedDict(
                            stage=access.stage.value,
                            accessed_splits=tuple(split.value for split in access.accessed_splits),
                        ),
                    )
                    for access in materialized.preprocessing_fit_accesses
                ),
                transfer_eligibility=eligibility,
            ),
        ),
        layout.staging,
    )
    promote_materialized_client(layout, dataset, materialized)
    return tuple(split_paths)


def promote_materialized_client(
    layout: WorkspaceLayout,
    dataset: DatasetId,
    materialized: MaterializedClient,
) -> Path:
    destination = (
        prepared_dataset_directory(layout, dataset)
        / PreprocessingPayloadFileName.MATERIALIZED_CLIENT
    )
    staged = staged_client_path(layout, destination)
    staged.parent.mkdir(parents=True, exist_ok=True)
    torch.save(materialized, staged)
    return promote_staged(staged, destination)


def staged_client_path(layout: WorkspaceLayout, destination: Path) -> Path:
    return staged_payload_path(destination, layout.staging)


def persist_client_invalid(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    dataset: DatasetId,
    reason: InvalidReason,
) -> None:
    destination = (
        experiment_workspace(layout, experiment)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
    )
    payload = cast(
        StableJsonPayload,
        OrderedDict(
            experiment=experiment.value,
            dataset=dataset.value,
            state=ArtifactState.INVALID.value,
            reason=reason,
        ),
    )
    atomic_write_json(destination / f"{dataset.value}-invalid.json", payload)


def load_or_materialize_client(
    dataset: DatasetId, raw_root: Path, layout: WorkspaceLayout
) -> MaterializedClient:
    prepared = (
        prepared_dataset_directory(layout, dataset)
        / PreprocessingPayloadFileName.MATERIALIZED_CLIENT
    )
    if prepared.is_file():
        loaded = torch.load(prepared, map_location="cpu", weights_only=False)
        if isinstance(loaded, MaterializedClient) and hasattr(loaded, "preprocessing_fit_accesses"):
            execution_logger().event(
                ExecutionEventName.PREPARED_CLIENT_LOAD,
                dataset=dataset.value,
                artifact_path=str(prepared),
            )
            return loaded
    execution_logger().event(ExecutionEventName.PREPARED_CLIENT_MISS, dataset=dataset.value)
    materialized = materialize_client(dataset, raw_root)
    if materialized.feature_quality.candidate_count_before_filtering > 0:
        persist_materialized_client(layout, materialized, OverwritePolicy.REUSE)
    return materialized
