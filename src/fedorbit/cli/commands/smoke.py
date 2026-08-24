from __future__ import annotations

import typer

from fedorbit.cli.errors import CliUsageError, exit_from_error
from fedorbit.config.testing import load_smoke_config, load_tests_config
from fedorbit.execution.errors import NotReadyError
from fedorbit.execution.executor import run_smoke_validation


def smoke(overwrite: bool = typer.Option(False, "--overwrite")) -> None:
    try:
        load_tests_config()
        load_smoke_config()
        run_smoke_validation(overwrite)
    except (CliUsageError, NotReadyError) as error:
        exit_from_error(error)
