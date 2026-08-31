from __future__ import annotations

import typer

from fedorbit.cli.errors import CliUsageError, exit_from_error
from fedorbit.cli.parsing import dataset_identifier
from fedorbit.config.context import active_config
from fedorbit.domain.enums import DatasetId, OverwritePolicy
from fedorbit.execution.executor import (
    DatasetPreparationRequest,
    ExecutionError,
    preprocess_datasets,
)


def preprocess(
    dataset_name: str | None = typer.Argument(None),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    try:
        selected = (
            (dataset_identifier(dataset_name),)
            if dataset_name is not None
            else _registered_datasets()
        )
        preprocess_datasets(
            DatasetPreparationRequest(
                datasets=selected,
                overwrite_policy=OverwritePolicy.REPLACE if overwrite else OverwritePolicy.REUSE,
            )
        )
    except (CliUsageError, ExecutionError) as error:
        exit_from_error(error)


def _registered_datasets() -> tuple[DatasetId, ...]:
    return tuple(active_config().scientific.datasets.clients.keys())
