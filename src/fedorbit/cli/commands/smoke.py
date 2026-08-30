from __future__ import annotations

import typer

from fedorbit.cli.errors import CliUsageError, exit_from_error
from fedorbit.config.testing import load_smoke_config, load_tests_config
from fedorbit.execution.executor import ExecutionError, OverwritePolicy, run_smoke_validation


def smoke(overwrite: bool = typer.Option(False, "--overwrite")) -> None:
    try:
        load_tests_config()
        load_smoke_config()
        run_smoke_validation(OverwritePolicy.REPLACE if overwrite else OverwritePolicy.REUSE)
    except (CliUsageError, ExecutionError) as error:
        exit_from_error(error)
