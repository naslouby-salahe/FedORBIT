from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fedorbit.artifacts.paths import build_layout
from fedorbit.artifacts.reuse import ArtifactStore
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.domain.enums import DatasetId, ExperimentName
from fedorbit.execution.errors import NotReadyError
from fedorbit.execution.semantics import ExecutionSemantics
from fedorbit.experiments.catalogue import ExperimentDefinition
from fedorbit.response.packet import build_source_packet
from fedorbit.response.uncertainty import FinalResponseEntry, FinalResponseEstimate


def _execution_root() -> Path:
    return build_layout(load_fedorbit_config()).execution_root


def _cells_for_datasets(datasets: tuple[DatasetId, ...]) -> tuple[tuple[str, str], ...]:
    cells: list[tuple[str, str]] = []
    for dataset in datasets:
        cells.append((f"raw-manifest:{dataset.value}", f"raw-{dataset.value}"))
        cells.append((f"prepared:{dataset.value}", f"prepared-{dataset.value}"))
    return tuple(cells)


def preprocess_pipeline(datasets: tuple[DatasetId, ...], overwrite: bool) -> None:
    store = ArtifactStore(_execution_root())
    semantics = ExecutionSemantics(store)
    cells = _cells_for_datasets(datasets)
    decisions = semantics.decide(cells, overwrite)
    semantics.validate_existing(decisions)
    for decision in decisions:
        if decision.execute or decision.overwrite:
            raise NotReadyError("preprocessing compute backend is not implemented")


def smoke_pipeline(overwrite: bool) -> None:
    del overwrite
    estimate = FinalResponseEstimate(
        entries=(FinalResponseEntry(0, 0, 1.0, 0.0, 1.0, 1.0, True),),
        critical_value=1.0,
        useful_intervention_columns=1,
        median_band_width_ratio=0.0,
        stability_rule_passed=True,
    )
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    packet = build_source_packet(
        estimate,
        anonymous_fine_node_ids=("node-0001",),
        exposed_coarse_group_id="smoke",
        per_node_train_support=(1,),
        per_node_meta_support=(1,),
        per_node_effective_replicate_count=(1,),
        source_checkpoint_sha256="0" * 64,
        response_configuration_sha256="1" * 64,
        creation_timestamp=timestamp,
    )
    packet.validate()


def run_pipeline(
    experiment: ExperimentName,
    definition: ExperimentDefinition,
    overwrite: bool,
) -> None:
    store = ArtifactStore(_execution_root())
    semantics = ExecutionSemantics(store)
    cells = tuple(
        (f"{experiment.value}:{seed}", f"cell-{experiment.value}-{seed}")
        for seed in definition.seeds
    )
    decisions = semantics.decide(cells, overwrite)
    semantics.validate_existing(decisions)
    for decision in decisions:
        if decision.execute or decision.overwrite:
            raise NotReadyError("confirmatory experiment compute backend is not implemented")
