from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.artifacts.paths import build_layout
from fedorbit.artifacts.storage import ArtifactStore
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.domain.enums import DatasetId, ExperimentName
from fedorbit.execution.errors import NotReadyError
from fedorbit.execution.reuse import CellDecision, ExecutionAction, ExecutionReuse
from fedorbit.experiments.catalogue import ExperimentDefinition
from fedorbit.response.packet import build_source_packet
from fedorbit.response.uncertainty import FinalResponseEntry, FinalResponseEstimate


class ExecutionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    decision: CellDecision
    manifest: ReusableArtifactManifest | None


class ExecutionExecutor:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

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
                continue
            manifest = producer(decision)
            self._store.write_reusable(manifest)
            validated = self._store.resolve(manifest.artifact_id)
            results.append(ExecutionResult(decision, validated))
        return tuple(results)


def execution_store() -> ArtifactStore:
    return ArtifactStore(build_layout(load_fedorbit_config()).execution_root)


def preprocess_datasets(datasets: tuple[DatasetId, ...], overwrite: bool) -> None:
    reuse = ExecutionReuse(execution_store())
    cells = tuple(
        cell
        for dataset in datasets
        for cell in (
            (f"raw-manifest:{dataset.value}", f"raw-{dataset.value}"),
            (f"prepared:{dataset.value}", f"prepared-{dataset.value}"),
        )
    )
    decisions = reuse.decide(cells, overwrite)
    reuse.validate_existing(decisions)
    if any(decision.execute or decision.overwrite for decision in decisions):
        raise NotReadyError("preprocessing compute backend is not implemented")


def run_smoke_validation(overwrite: bool) -> None:
    del overwrite
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


def run_experiment(
    experiment: ExperimentName,
    definition: ExperimentDefinition,
    overwrite: bool,
) -> None:
    reuse = ExecutionReuse(execution_store())
    cells = tuple(
        (f"{experiment.value}:{seed}", f"cell-{experiment.value}-{seed}")
        for seed in definition.seeds
    )
    decisions = reuse.decide(cells, overwrite)
    reuse.validate_existing(decisions)
    if any(decision.execute or decision.overwrite for decision in decisions):
        raise NotReadyError("confirmatory experiment compute backend is not implemented")
