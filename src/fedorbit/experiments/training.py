from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch

from fedorbit.analysis.metrics import (
    EfficiencyRecord,
)
from fedorbit.config.loading import active_config, raw_dataset_root
from fedorbit.datasets.common import (
    file_sha256,
)
from fedorbit.datasets.materialization import (
    MaterializationError,
    MaterializedClient,
    TransferConceptGroup,
    transfer_concept_groups,
)
from fedorbit.datasets.ontology import TRANSFER_ONTOLOGY
from fedorbit.experiments.cells import experiment_relevance
from fedorbit.experiments.protocol import ExperimentExecutionRequest
from fedorbit.experiments.scoring import _completion
from fedorbit.experiments.validation import _persist_synthetic_experiment_payload
from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
    ExecutionError,
)
from fedorbit.infrastructure.environment import environment_snapshot
from fedorbit.infrastructure.evidence import TableScalar
from fedorbit.infrastructure.manifests import (
    ReusableArtifactManifest,
    artifact_id,
)
from fedorbit.infrastructure.preparation import (
    _BASE_MODEL_PILOT_CONFIGURATION_SECTIONS,
    _load_or_materialize_client,
    _persist_client_invalid,
    build_dataset_manifest,
    persist_dataset_manifest,
)
from fedorbit.infrastructure.provenance import (
    configuration_subset_digest,
    implementation_fingerprint,
    runtime_fingerprint,
    stage_dependency_fingerprint,
)
from fedorbit.infrastructure.runtime import (
    EfficiencyMeasurement,
    ExecutionLogEvent,
    ExecutionLogger,
    RandomSeed,
    current_code_revision,
    execution_logger,
    measure_efficiency,
    principal_determinism,
)
from fedorbit.infrastructure.storage import atomic_write_bytes, atomic_write_json
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
    experiment_workspace,
)
from fedorbit.learning.checkpoints import load_base_checkpoint, save_base_checkpoint
from fedorbit.learning.pilot import (
    HOST_DATASETS,
    NETWORK_DATASETS,
    PilotData,
    create_classifier,
    run_base_model_pilot,
    select_pilot_configuration,
)
from fedorbit.learning.training import (
    BaseCheckpoint,
    ClassWeights,
    SelectedHyperparameters,
    train_base_model,
)
from fedorbit.response.packet import (
    PacketConstructionContext,
    construct_source_packet,
)
from fedorbit.response.pilot import (
    PilotCheckpoint,
    ResponseCandidate,
    run_pooled_source_response_pilot,
    select_response_configuration,
)
from fedorbit.response.pilot import PilotData as ResponsePilotData
from fedorbit.types import (
    AnonymousNodeDisplayId,
    ArtifactFingerprint,
    ArtifactPath,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    ArtifactTypeName,
    CheckpointDirectorySegment,
    CoarseGroup,
    ConfigurationSection,
    DatasetId,
    ExecutionStageName,
    ExperimentName,
    ExperimentSeed,
    InvalidReason,
    OverwritePolicy,
    ProducerModuleName,
    ReuseDecision,
    Rfc3339UtcTimestamp,
    SemanticCell,
    SemanticCoordinate,
    SemanticCoordinates,
    SemanticCoordinateText,
    Sha256Digest,
    Split,
    StableJsonPayload,
)

_MODULE_NAME = ProducerModuleName("fedorbit.experiments.training")


def execute_base_model_pilot(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    experiment = request.experiment
    relevance = experiment_relevance(experiment)
    raw_root = raw_dataset_root()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    device = torch.device("cuda")
    logger = execution_logger()
    for dataset in active_config().scientific.datasets.clients:
        logger.record(
            ExecutionLogEvent(
                occurred_at=datetime.now(UTC),
                cell_coordinates=SemanticCoordinates(f"{experiment.value}:{dataset.value}"),
                artifact_id=None,
                state=ArtifactState.RUNNING,
                stage=ExecutionStageName(ArtifactStage.PREPROCESSING.value),
                experiment=experiment,
                dataset=dataset,
            )
        )

        materialize_started_at = time.monotonic()
        try:
            materialized = _load_or_materialize_client(dataset, raw_root, layout)
        except MaterializationError as error:
            _persist_client_invalid(layout, experiment, dataset, InvalidReason(str(error)))
            logger.record(
                ExecutionLogEvent(
                    occurred_at=datetime.now(UTC),
                    cell_coordinates=SemanticCoordinates(f"{experiment.value}:{dataset.value}"),
                    artifact_id=None,
                    state=ArtifactState.INVALID,
                    stage=ExecutionStageName(ArtifactStage.PREPROCESSING.value),
                    experiment=experiment,
                    dataset=dataset,
                    elapsed_seconds=time.monotonic() - materialize_started_at,
                )
            )
            continue
        logger.record(
            ExecutionLogEvent(
                occurred_at=datetime.now(UTC),
                cell_coordinates=SemanticCoordinates(f"{experiment.value}:{dataset.value}"),
                artifact_id=None,
                state=ArtifactState.COMPLETED,
                stage=ExecutionStageName(ArtifactStage.PREPROCESSING.value),
                experiment=experiment,
                dataset=dataset,
                elapsed_seconds=time.monotonic() - materialize_started_at,
            )
        )
        persist_dataset_manifest(layout, experiment, dataset, build_dataset_manifest(materialized))
        _execute_client_base_model_pilot(
            store,
            layout,
            experiment,
            relevance,
            dataset,
            materialized,
            confirmatory_seeds,
            request.overwrite_policy,
            device,
            logger,
        )


_SOURCE_RESPONSE_PILOT_CONFIGURATION_SECTIONS = frozenset(
    {ConfigurationSection.MODELS, ConfigurationSection.RESPONSE}
)


def execute_source_response_estimator_pilot(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> ReusableArtifactManifest:
    seed = ExperimentSeed(active_config().scientific.randomness.pilot_seeds[0])
    return _persist_synthetic_experiment_payload(
        store,
        layout,
        request,
        seed,
        lambda fingerprint: _source_response_estimator_payload(layout, request, fingerprint),
        _SOURCE_RESPONSE_PILOT_CONFIGURATION_SECTIONS,
        _MODULE_NAME,
        "source-response-pilot",
    )


def _source_response_estimator_payload(
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    fingerprint: Sha256Digest,
) -> StableJsonPayload:
    return _source_response_estimator_client_results(layout, request, fingerprint)


def _execute_final_source_response_band_validation(
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    selected_root = (
        experiment_workspace(layout, ExperimentName.SOURCE_RESPONSE_ESTIMATOR_PILOT)
        / "artifacts"
        / "fitted"
    )
    checkpoint_root = (
        experiment_workspace(layout, ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT)
        / "checkpoints"
        / "training"
    )
    for dataset in active_config().scientific.datasets.clients:
        try:
            materialized = _load_or_materialize_client(dataset, raw_root, layout)
        except MaterializationError as error:
            _persist_client_invalid(layout, request.experiment, dataset, InvalidReason(str(error)))
            continue
        selected_path = selected_root / f"{dataset.value}.json"
        if not selected_path.is_file():
            raise ExecutionError(f"missing selected response configuration: {selected_path}")
        selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
        candidate = ResponseCandidate(
            selected_payload["intervention_magnitude"],
            selected_payload["optimizer_step_horizon"],
        )
        eligible_by_group: OrderedDict[CoarseGroup, list[TransferConceptGroup]] = OrderedDict()
        for group in transfer_concept_groups(dataset, materialized):
            if group.source_eligible:
                eligible_by_group.setdefault(TRANSFER_ONTOLOGY[group.concept][0], []).append(group)
        for seed in active_config().scientific.randomness.confirmatory_seeds:
            checkpoint_path = checkpoint_root / dataset.value / f"seed-{seed}" / "checkpoint.pt"
            if not checkpoint_path.is_file():
                raise ExecutionError(f"missing confirmatory checkpoint: {checkpoint_path}")
            checkpoint = load_base_checkpoint(checkpoint_path)
            for coarse_group, groups in eligible_by_group.items():
                node_classes = tuple(group.native_class_indices for group in groups)
                model = create_classifier(
                    dataset,
                    materialized.splits[Split.TRAIN].features.shape[1],
                    materialized.class_manifest.class_count,
                    checkpoint.selected_hyperparameters.dropout_probability,
                    seed,
                )
                checkpoint.state_dict.load_into(model)
                packet = construct_source_packet(
                    PacketConstructionContext(
                        dataset,
                        materialized.splits[Split.TRAIN].features.shape[1],
                        materialized.class_manifest.class_count,
                        coarse_group,
                        tuple(AnonymousNodeDisplayId(group.concept.value) for group in groups),
                        tuple(group.train_support for group in groups),
                        tuple(group.meta_support for group in groups),
                        file_sha256(checkpoint_path),
                        Sha256Digest(hashlib.sha256(selected_path.read_bytes()).hexdigest()),
                        seed,
                    ),
                    checkpoint,
                    model,
                    materialized.splits[Split.TRAIN].features,
                    materialized.splits[Split.TRAIN].targets,
                    materialized.splits[Split.META].features,
                    materialized.splits[Split.META].targets,
                    node_classes,
                    node_classes,
                    checkpoint.train_class_weights,
                    candidate,
                    Rfc3339UtcTimestamp(datetime.now(UTC).isoformat().replace("+00:00", "Z")),
                ).packet
                destination = (
                    experiment_workspace(layout, request.experiment)
                    / "artifacts"
                    / "packets"
                    / dataset.value
                    / f"seed-{seed}"
                    / f"{coarse_group.value.casefold().replace(' ', '-')}.json"
                )
                atomic_write_bytes(destination, (packet.serialized() + "\n").encode("utf-8"))


def _source_response_estimator_client_results(
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    fingerprint: Sha256Digest,
) -> StableJsonPayload:
    raw_root = raw_dataset_root()
    pilot_seeds = active_config().scientific.randomness.pilot_seeds
    base_workspace = experiment_workspace(layout, ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT)
    destination = experiment_workspace(layout, request.experiment) / "artifacts" / "fitted"
    diagnostics_destination = (
        experiment_workspace(layout, request.experiment) / "artifacts" / "derived"
    )
    client_results: list[StableJsonPayload] = []
    for dataset in active_config().scientific.datasets.clients:
        try:
            materialized = _load_or_materialize_client(dataset, raw_root, layout)
        except MaterializationError as error:
            _persist_client_invalid(layout, request.experiment, dataset, InvalidReason(str(error)))
            client_results.append(
                cast(
                    StableJsonPayload,
                    OrderedDict(dataset=dataset.value, state=ArtifactState.INVALID.value),
                )
            )
            continue
        groups = transfer_concept_groups(dataset, materialized)
        eligible = tuple(group for group in groups if group.source_eligible)
        if len(eligible) < 2:
            _persist_client_invalid(
                layout,
                request.experiment,
                dataset,
                InvalidReason("fewer than two eligible source transfer concepts"),
            )
            client_results.append(
                cast(
                    StableJsonPayload,
                    OrderedDict(dataset=dataset.value, state=ArtifactState.INVALID.value),
                )
            )
            continue
        checkpoints: list[PilotCheckpoint] = []
        for seed in pilot_seeds:
            path = (
                base_workspace
                / "checkpoints"
                / "pilot"
                / dataset.value
                / f"seed-{seed}"
                / "checkpoint.pt"
            )
            if not path.is_file():
                raise ExecutionError(f"missing selected pilot checkpoint: {path}")
            checkpoint = load_base_checkpoint(path)
            model = create_classifier(
                dataset,
                materialized.splits[Split.TRAIN].features.shape[1],
                materialized.class_manifest.class_count,
                checkpoint.selected_hyperparameters.dropout_probability,
                seed,
            )
            checkpoint.state_dict.load_into(model)
            checkpoints.append(PilotCheckpoint(model, checkpoint, seed))
        checkpoint = checkpoints[0].checkpoint
        intervention_classes = tuple(group.native_class_indices for group in eligible)
        data = ResponsePilotData(
            materialized.splits[Split.TRAIN].features,
            materialized.splits[Split.TRAIN].targets,
            materialized.splits[Split.META].features,
            materialized.splits[Split.META].targets,
            intervention_classes,
            checkpoint.train_class_weights,
            checkpoint.selected_hyperparameters.learning_rate,
            checkpoint.selected_hyperparameters.weight_decay,
        )
        results = run_pooled_source_response_pilot(tuple(checkpoints), data, intervention_classes)
        selected = select_response_configuration(results)
        atomic_write_json(
            diagnostics_destination / f"{dataset.value}-candidates.json",
            cast(
                StableJsonPayload,
                OrderedDict(
                    dataset=dataset.value,
                    pilot_checkpoint_seeds=pilot_seeds,
                    candidates=tuple(asdict(result) for result in results),
                ),
            ),
        )
        atomic_write_json(
            destination / f"{dataset.value}.json",
            cast(
                StableJsonPayload,
                OrderedDict(
                    dataset=dataset.value,
                    intervention_magnitude=selected.intervention_magnitude,
                    optimizer_step_horizon=selected.optimizer_step_horizon,
                    pilot_checkpoint_seeds=pilot_seeds,
                ),
            ),
        )
        client_results.append(
            cast(
                StableJsonPayload,
                OrderedDict(
                    dataset=dataset.value,
                    state=ArtifactState.COMPLETED.value,
                    selected_configuration=cast(
                        StableJsonPayload,
                        OrderedDict(
                            intervention_magnitude=selected.intervention_magnitude,
                            optimizer_step_horizon=selected.optimizer_step_horizon,
                        ),
                    ),
                ),
            )
        )
    return cast(
        StableJsonPayload,
        OrderedDict(
            experiment=request.experiment.value,
            dependency_fingerprint_sha256=fingerprint,
            clients=tuple(client_results),
        ),
    )


def execute_final_source_response_band_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> ReusableArtifactManifest:
    _execute_final_source_response_band_validation(layout, request)
    return _persist_synthetic_experiment_payload(
        store,
        layout,
        request,
        ExperimentSeed(request.definition.seeds[0]),
        lambda fingerprint: cast(
            StableJsonPayload,
            OrderedDict(
                experiment=request.experiment.value,
                dependency_fingerprint_sha256=fingerprint,
                packets_persisted=True,
            ),
        ),
        frozenset({ConfigurationSection.RESPONSE, ConfigurationSection.MODELS}),
        _MODULE_NAME,
        "source-response-band-validation",
    )


@dataclass(frozen=True, slots=True)
class _ModelArchitectureFacts:
    architecture: str
    normalization: str
    activation: str
    initialization: str


_NETWORK_MODEL_ARCHITECTURE_FACTS = _ModelArchitectureFacts(
    architecture="256-128-64 MLP (NetworkFlowClassifier)",
    normalization="LayerNorm",
    activation="GELU",
    initialization="Xavier uniform",
)
_HOST_MODEL_ARCHITECTURE_FACTS = _ModelArchitectureFacts(
    architecture="192-96-48 MLP (HostClassifier)",
    normalization="BatchNorm1d",
    activation="ReLU",
    initialization="Kaiming uniform",
)


def _pilot_selected_hyperparameters(
    store: ArtifactStore, dataset: DatasetId
) -> SelectedHyperparameters | None:
    experiment_value = ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT.value
    for manifest in store.all_manifests():
        if (
            experiment_value not in manifest.semantic_producer_coordinates
            or dataset.value not in manifest.semantic_producer_coordinates
            or len(manifest.payload_paths) != 1
        ):
            continue
        payload_path = Path(manifest.payload_paths[0])
        if CheckpointDirectorySegment.PILOT.value not in payload_path.parts:
            continue
        try:
            resolved = store.resolve(manifest.artifact_id)
        except ValueError:
            continue
        if resolved.state != ArtifactState.COMPLETED:
            continue
        return load_base_checkpoint(payload_path).selected_hyperparameters
    return None


def training_protocol_rows(
    store: ArtifactStore,
) -> tuple[Mapping[str, TableScalar], ...]:
    training = active_config().scientific.training
    stopping_rule = (
        f"early stop after {training.early_stopping.patience_completed_epochs} epochs without "
        f">= {training.early_stopping.minimum_improvement} macro-CE improvement "
        f"(maximum {training.maximum_epochs} epochs)"
    )
    rows: list[Mapping[str, TableScalar]] = []
    for dataset in active_config().scientific.datasets.clients:
        if dataset in NETWORK_DATASETS:
            facts = _NETWORK_MODEL_ARCHITECTURE_FACTS
        elif dataset in HOST_DATASETS:
            facts = _HOST_MODEL_ARCHITECTURE_FACTS
        else:
            continue
        selected = _pilot_selected_hyperparameters(store, dataset)
        if selected is None:
            continue
        rows.append(
            OrderedDict(
                model=dataset.value,
                architecture=facts.architecture,
                normalization=facts.normalization,
                activation=facts.activation,
                initialization=facts.initialization,
                optimizer="AdamW",
                batch=training.batch_size,
                selected_learning_rate=selected.learning_rate,
                selected_weight_decay=selected.weight_decay,
                selected_dropout=selected.dropout_probability,
                stopping_rule=stopping_rule,
            )
        )
    return tuple(rows)


def _execute_client_base_model_pilot(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    relevance: frozenset[SemanticCoordinate],
    dataset: DatasetId,
    materialized: MaterializedClient,
    confirmatory_seeds: tuple[RandomSeed, ...],
    overwrite_policy: OverwritePolicy,
    device: torch.device,
    logger: ExecutionLogger,
) -> None:
    train = materialized.splits[Split.TRAIN]
    valid = materialized.splits[Split.VALID]
    n_classes = materialized.class_manifest.class_count
    pilot_data = PilotData(train.features, train.targets, valid.features, valid.targets, n_classes)
    class_weights = ClassWeights.from_targets(train.targets, n_classes)
    with principal_determinism():
        pilot_started_at = time.monotonic()
        logger.record(
            ExecutionLogEvent(
                occurred_at=datetime.now(UTC),
                cell_coordinates=SemanticCoordinates(f"{experiment.value}:{dataset.value}:pilot"),
                artifact_id=None,
                state=ArtifactState.RUNNING,
                stage=ExecutionStageName(ArtifactStage.PILOT_SELECTION.value),
                experiment=experiment,
                dataset=dataset,
            )
        )
        pilot_results = run_base_model_pilot(pilot_data, dataset, device)
        selection = select_pilot_configuration(pilot_results)
        logger.record(
            ExecutionLogEvent(
                occurred_at=datetime.now(UTC),
                cell_coordinates=SemanticCoordinates(f"{experiment.value}:{dataset.value}:pilot"),
                artifact_id=None,
                state=ArtifactState.COMPLETED,
                stage=ExecutionStageName(ArtifactStage.PILOT_SELECTION.value),
                experiment=experiment,
                dataset=dataset,
                elapsed_seconds=time.monotonic() - pilot_started_at,
            )
        )
        for pilot_result in pilot_results:
            if pilot_result.configuration != selection.configuration:
                continue
            _persist_base_checkpoint(
                store,
                layout,
                experiment,
                relevance,
                dataset,
                pilot_result.seed,
                pilot_result.outcome.checkpoint,
                ArtifactStage.PILOT_SELECTION,
                CheckpointDirectorySegment.PILOT,
                overwrite_policy,
            )
        for seed_index, seed in enumerate(confirmatory_seeds):
            checkpoint_coordinates = SemanticCoordinates(
                f"{experiment.value}:{dataset.value}:confirmatory:{seed}"
            )
            checkpoint_cell = SemanticCell(
                experiment=experiment,
                dataset=dataset,
                source_client=dataset,
                seed=ExperimentSeed(seed),
            )
            fingerprint = stage_dependency_fingerprint(
                ArtifactStage.TRAINING,
                checkpoint_cell,
                relevance,
                (),
                _BASE_MODEL_PILOT_CONFIGURATION_SECTIONS,
                _MODULE_NAME,
            )
            if overwrite_policy == OverwritePolicy.REUSE:
                existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
                if existing is not None:
                    logger.record(
                        ExecutionLogEvent(
                            occurred_at=datetime.now(UTC),
                            cell_coordinates=checkpoint_coordinates,
                            artifact_id=existing.artifact_id,
                            state=ArtifactState.COMPLETED,
                            stage=ExecutionStageName(ArtifactStage.TRAINING.value),
                            experiment=experiment,
                            dataset=dataset,
                            seed=seed,
                            reuse_decision=ReuseDecision("reused"),
                        )
                    )
                    continue
            logger.record(
                ExecutionLogEvent(
                    occurred_at=datetime.now(UTC),
                    cell_coordinates=checkpoint_coordinates,
                    artifact_id=None,
                    state=ArtifactState.RUNNING,
                    stage=ExecutionStageName(ArtifactStage.TRAINING.value),
                    experiment=experiment,
                    dataset=dataset,
                    seed=seed,
                    reuse_decision=ReuseDecision(
                        f"{seed_index + 1}/{len(confirmatory_seeds)} confirmatory checkpoints"
                    ),
                )
            )
            checkpoint_started_at = time.monotonic()
            with principal_determinism(), measure_efficiency() as efficiency:
                model = create_classifier(
                    dataset,
                    train.features.shape[1],
                    n_classes,
                    selection.configuration.dropout,
                    seed,
                    device,
                )
                outcome = train_base_model(
                    model,
                    train.features,
                    train.targets,
                    valid.features,
                    valid.targets,
                    class_weights,
                    seed,
                    selection.configuration.hyperparameters(),
                )
            _persist_training_efficiency(layout, experiment, dataset, seed, efficiency.result)
            _persist_base_checkpoint(
                store,
                layout,
                experiment,
                relevance,
                dataset,
                seed,
                outcome.checkpoint,
                ArtifactStage.TRAINING,
                CheckpointDirectorySegment.TRAINING,
                overwrite_policy,
            )
            logger.record(
                ExecutionLogEvent(
                    occurred_at=datetime.now(UTC),
                    cell_coordinates=checkpoint_coordinates,
                    artifact_id=None,
                    state=ArtifactState.COMPLETED,
                    stage=ExecutionStageName(ArtifactStage.TRAINING.value),
                    experiment=experiment,
                    dataset=dataset,
                    seed=seed,
                    elapsed_seconds=time.monotonic() - checkpoint_started_at,
                )
            )


def _persist_training_efficiency(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    dataset: DatasetId,
    seed: RandomSeed,
    measurement: EfficiencyMeasurement,
) -> Path:
    record = EfficiencyRecord(
        wall_time_seconds=measurement.wall_time_seconds,
        peak_host_rss_mib=measurement.peak_host_rss_mib,
        peak_cuda_allocated_bytes=measurement.peak_cuda_allocated_bytes,
        packet_serialized_byte_count=0,
        source_response_optimizer_steps=0,
        target_confirmation_optimizer_steps=0,
        live_assimilation_optimizer_steps=0,
        timeout_indicator=False,
        resource_limit_indicator=False,
    )
    destination = (
        experiment_workspace(layout, experiment)
        / "artifacts"
        / "derived"
        / f"training-efficiency.{dataset.value}.{seed}.json"
    )
    atomic_write_json(destination, OrderedDict[str, StableJsonPayload](asdict(record)))
    return destination


def _persist_base_checkpoint(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    relevance: frozenset[SemanticCoordinate],
    dataset: DatasetId,
    seed: RandomSeed,
    checkpoint: BaseCheckpoint,
    stage: ArtifactStage,
    directory_segment: CheckpointDirectorySegment,
    overwrite_policy: OverwritePolicy,
) -> None:
    checkpoint_cell = SemanticCell(
        experiment=experiment,
        dataset=dataset,
        source_client=dataset,
        seed=ExperimentSeed(seed),
    )
    coordinates = SemanticCoordinateText(checkpoint_cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            stage,
            checkpoint_cell,
            relevance,
            (),
            _BASE_MODEL_PILOT_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return
    payload_path = (
        experiment_workspace(layout, experiment)
        / "checkpoints"
        / directory_segment
        / dataset.value
        / f"seed-{seed}"
        / "checkpoint.pt"
    )
    save_base_checkpoint(checkpoint, payload_path)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_BASE_MODEL_PILOT_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(stage).sha256)
    completion = _completion(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=stage,
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.CHECKPOINT.value),
                cast(
                    StableJsonPayload,
                    OrderedDict(coordinates=coordinates, payload_sha256=payload_sha256),
                ),
                Sha256Digest(fingerprint),
            ),
            artifact_type=ArtifactType.CHECKPOINT,
            semantic_producer_coordinates=coordinates,
            producer_stage=stage,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=(),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            material_runtime_sha256=runtime_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_sha256,
            schema_version="1.0",
            created_git_commit=current_code_revision().commit,
            created_environment_sha256=environment_snapshot().fingerprint_sha256,
            state=ArtifactState.COMPLETED,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
    )
    store.write_completed(manifest, completion)
