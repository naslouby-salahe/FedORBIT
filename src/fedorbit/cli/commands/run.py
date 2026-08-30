from __future__ import annotations

import typer

from fedorbit.cli.errors import CliUsageError, exit_from_error
from fedorbit.cli.parsing import experiment_identifier
from fedorbit.execution.errors import NotReadyError
from fedorbit.execution.executor import ExperimentExecutionRequest, OverwritePolicy, run_experiment
from fedorbit.experiments.catalogue import build_catalogue


def run(
    experiment_name: str,
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    try:
        experiment = experiment_identifier(experiment_name)
        definition = build_catalogue().definition(experiment)
        run_experiment(
            ExperimentExecutionRequest(
                experiment=experiment,
                definition=definition,
                overwrite_policy=OverwritePolicy.REPLACE if overwrite else OverwritePolicy.REUSE,
            )
        )
    except (CliUsageError, NotReadyError) as error:
        exit_from_error(error)
