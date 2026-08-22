from __future__ import annotations

from fedorbit.cli.errors import CliUsageError
from fedorbit.domain.enums import DatasetId, ExperimentName


def dataset_identifier(name: str) -> DatasetId:
    for candidate in DatasetId:
        if candidate.value == name:
            return candidate
    raise CliUsageError(
        f"unknown dataset identifier {name!r}: use the exact registered "
        "identifier (display names, filesystem names, aliases, and source-dataset "
        "names such as Edge-IIoTset or ToN-IoT are not accepted)"
    )


def experiment_identifier(name: str) -> ExperimentName:
    for candidate in ExperimentName:
        if candidate.value == name:
            return candidate
    raise CliUsageError(
        f"unknown experiment name {name!r}: use the exact registered experiment name"
    )
