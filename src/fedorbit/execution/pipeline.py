from __future__ import annotations

from pathlib import Path

from fedorbit.artifacts.paths import build_layout
from fedorbit.artifacts.reuse import ArtifactStore
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.domain.enums import DatasetId, ExperimentName
from fedorbit.execution.errors import NotReadyError
from fedorbit.execution.semantics import ExecutionSemantics
from fedorbit.experiments.catalogue import ExperimentDefinition


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
            raise NotReadyError("preprocessing compute backend is delivered by the M03 milestone")


def smoke_pipeline(overwrite: bool) -> None:
    store = ArtifactStore(_execution_root())
    semantics = ExecutionSemantics(store)
    decisions = semantics.decide((("smoke-nonclaim", "smoke-fixture"),), overwrite)
    semantics.validate_existing(decisions)
    for decision in decisions:
        if decision.execute or decision.overwrite:
            raise NotReadyError("extended smoke check registry is delivered by the M07 milestone")


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
            raise NotReadyError("confirmatory experiment compute backends are delivered by M07-M08")
