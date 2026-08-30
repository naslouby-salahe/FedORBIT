from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.artifacts.paths import build_layout
from fedorbit.artifacts.storage import ArtifactStore
from fedorbit.config.context import active_config
from fedorbit.config.loading import repository_root
from fedorbit.datasets.inspection import (
    DatasetInspectionRequest,
    DatasetObservation,
    DatasetObservationPersistenceRequest,
    inspect_dataset,
    persist_dataset_observation,
)
from fedorbit.domain.enums import ArtifactState, DatasetId, ExperimentName, ScalabilityBlockPattern
from fedorbit.domain.records import ArtifactPath
from fedorbit.execution.inventory import (
    RawInventoryPersistenceRequest,
    RawInventoryRequest,
    inspect_raw_inventory,
    persist_raw_inventory,
)
from fedorbit.execution.recovery import RecoveryBoundary
from fedorbit.execution.reuse import CellDecision, ExecutionAction, ExecutionReuse
from fedorbit.experiments.catalogue import ExperimentDefinition
from fedorbit.response.packet import build_source_packet
from fedorbit.response.uncertainty import FinalResponseEntry, FinalResponseEstimate
from fedorbit.runtime.logging import ExecutionLogEvent, ExecutionLogger, execution_logger
from fedorbit.runtime.seeds import RandomSeed
from fedorbit.synthetic.exactness import (
    ExactSeparatorInstanceRequest,
    generate_exact_separator_instance,
)
from fedorbit.synthetic.mechanisms import (
    UnresolvedMapWorldKind,
    UnresolvedMapWorldRequest,
    generate_unresolved_map_world,
)
from fedorbit.synthetic.scalability import ScalabilityInstanceRequest, generate_scalability_instance


class ExecutionError(ValueError):
    pass


class OverwritePolicy(StrEnum):
    REUSE = "reuse"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class DatasetPreparationRequest:
    datasets: tuple[DatasetId, ...]
    overwrite_policy: OverwritePolicy

    @property
    def overwrite_requested(self) -> bool:
        return self.overwrite_policy == OverwritePolicy.REPLACE


@dataclass(frozen=True, slots=True)
class ExperimentExecutionRequest:
    experiment: ExperimentName
    definition: ExperimentDefinition
    overwrite_policy: OverwritePolicy

    @property
    def overwrite_requested(self) -> bool:
        return self.overwrite_policy == OverwritePolicy.REPLACE


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    decision: CellDecision
    manifest: ReusableArtifactManifest | None


@dataclass(frozen=True, slots=True)
class DatasetPreparationResult:
    observations: tuple[DatasetObservation, ...]
    validation_artifact_paths: tuple[ArtifactPath, ...]

    @property
    def blocked_datasets(self) -> tuple[DatasetId, ...]:
        return tuple(
            observation.dataset
            for observation in self.observations
            if not observation.valid_for_chronological_preprocessing
        )


class ExecutionExecutor:
    def __init__(self, store: ArtifactStore, logger: ExecutionLogger | None = None) -> None:
        self._store = store
        self._logger = logger if logger is not None else execution_logger()

    def execute(
        self,
        decisions: tuple[CellDecision, ...],
        producer: Callable[[CellDecision], ReusableArtifactManifest],
    ) -> tuple[ExecutionResult, ...]:
        results: list[ExecutionResult] = []
        for decision in decisions:
            if decision.action == ExecutionAction.REUSE:
                if decision.manifest is None:
                    raise ExecutionError("reuse decision has no manifest")
                manifest = self._store.resolve(decision.manifest.artifact_id)
                results.append(ExecutionResult(decision, manifest))
                self._logger.record(
                    ExecutionLogEvent(
                        occurred_at=datetime.now(UTC),
                        cell_coordinates=decision.cell_coordinates,
                        artifact_id=manifest.artifact_id,
                        state=ArtifactState.COMPLETED,
                    )
                )
                continue
            manifest = producer(decision)
            self._store.write_reusable(manifest)
            validated = self._store.resolve(manifest.artifact_id)
            results.append(ExecutionResult(decision, validated))
            self._logger.record(
                ExecutionLogEvent(
                    occurred_at=datetime.now(UTC),
                    cell_coordinates=decision.cell_coordinates,
                    artifact_id=validated.artifact_id,
                    state=ArtifactState.COMPLETED,
                )
            )
        return tuple(results)


def execution_store() -> ArtifactStore:
    return ArtifactStore(build_layout().execution_root)


def _recover(store: ArtifactStore, cells: tuple[tuple[str, str], ...]) -> None:
    recovery = RecoveryBoundary(store)
    recovery.discard_interrupted_staging()
    recovery.next_resume(cells)


def preprocess_datasets(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    raw_root = repository_root() / "data" / "raw"
    inventories = tuple(
        inspect_raw_inventory(RawInventoryRequest(dataset, raw_root))
        for dataset in request.datasets
    )
    if len(inventories) != len(request.datasets):
        raise ExecutionError("raw inventory collection did not cover every requested dataset")
    store = execution_store()
    persisted_inventory_paths = tuple(
        persist_raw_inventory(
            RawInventoryPersistenceRequest(inventory, store.root / "preprocessing")
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
            DatasetObservationPersistenceRequest(observation, store.root / "preprocessing")
        )
        for observation in observations
    )
    return DatasetPreparationResult(
        observations=observations,
        validation_artifact_paths=tuple(ArtifactPath(path) for path in validation_paths),
    )


def run_smoke_validation(overwrite_policy: OverwritePolicy) -> None:
    del overwrite_policy
    seed = RandomSeed(active_config().scientific.randomness.pilot_seeds[0])
    exact = generate_exact_separator_instance(ExactSeparatorInstanceRequest((2, 2), seed))
    if exact.lower_response_matrix.shape != (4, 4):
        raise ExecutionError("synthetic exactness smoke instance has an invalid matrix shape")
    mechanism = generate_unresolved_map_world(
        UnresolvedMapWorldRequest(UnresolvedMapWorldKind.COMMON_ACTION, seed)
    )
    if mechanism.lower_response_matrix.shape != (4, 4):
        raise ExecutionError("synthetic mechanism smoke instance has an invalid matrix shape")
    scalability = generate_scalability_instance(
        ScalabilityInstanceRequest(4, ScalabilityBlockPattern.BALANCED, 1, seed)
    )
    if scalability.fixed_action.shape != (4,):
        raise ExecutionError("synthetic scalability smoke instance has an invalid action shape")
    estimate = FinalResponseEstimate(
        entries=(FinalResponseEntry(0, 0, 1.0, 0.0, 1.0, 1.0, True),),
        critical_value=1.0,
        useful_intervention_columns=1,
        median_band_width_ratio=0.0,
        stability_rule_passed=True,
    )
    packet = build_source_packet(
        estimate,
        anonymous_fine_node_ids=("node-0001",),
        exposed_coarse_group_id="smoke",
        per_node_train_support=(1,),
        per_node_meta_support=(1,),
        per_node_effective_replicate_count=(1,),
        source_checkpoint_sha256="0" * 64,
        response_configuration_sha256="1" * 64,
        creation_timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    packet.validate()


def run_experiment(request: ExperimentExecutionRequest) -> None:
    store = execution_store()
    reuse = ExecutionReuse(store)
    cells = tuple(
        (f"{request.experiment.value}:{seed}", f"cell-{request.experiment.value}-{seed}")
        for seed in request.definition.seeds
    )
    _recover(store, cells)
    decisions = reuse.decide(cells, request.overwrite_requested)
    reuse.validate_existing(decisions)
    if any(decision.execute or decision.overwrite for decision in decisions):
        raise ExecutionError("registered experiment execution has no completed producer")
