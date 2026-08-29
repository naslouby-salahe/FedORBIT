from __future__ import annotations

from fedorbit.domain.enums import ExperimentName
from fedorbit.experiments.catalogue import ExperimentCatalogue


class ExperimentValidationError(ValueError):
    pass


def validate_catalogue(catalogue: ExperimentCatalogue) -> None:
    registered = catalogue.registered_names()
    if set(registered) != set(ExperimentName):
        raise ExperimentValidationError("catalogue must define every registered experiment")
    if len(registered) != len(set(registered)):
        raise ExperimentValidationError("catalogue registers an experiment more than once")
    for definition in (catalogue.definition(name) for name in registered):
        if definition.derived_planned_cells < 0:
            raise ExperimentValidationError(
                f"experiment has negative planned cell count: {definition.name.value}"
            )
        if definition.derived_planned_cells > 0 and not definition.seeds:
            raise ExperimentValidationError(
                f"executable experiment has no registered seeds: {definition.name.value}"
            )
        if any(not prerequisite for prerequisite in definition.prerequisites):
            raise ExperimentValidationError(
                f"experiment has an empty prerequisite: {definition.name.value}"
            )
