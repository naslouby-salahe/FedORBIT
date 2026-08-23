from __future__ import annotations

from dataclasses import dataclass

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import ExperimentClassification, ExperimentName
from fedorbit.experiments.catalogue import build_catalogue


@dataclass(frozen=True, slots=True)
class PlanRow:
    experiment: ExperimentName
    classification: ExperimentClassification
    planned_cells: int
    prerequisites: tuple[str, ...]


def build_plan(config: FedorbitConfig) -> tuple[PlanRow, ...]:
    catalogue = build_catalogue(config)
    return tuple(
        PlanRow(
            experiment=name,
            classification=catalogue.definition(name).classification,
            planned_cells=catalogue.definition(name).derived_planned_cells,
            prerequisites=catalogue.definition(name).prerequisites,
        )
        for name in catalogue.registered_names()
    )
