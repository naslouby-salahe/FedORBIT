from __future__ import annotations

from typing import NoReturn

import typer
from typer import Exit

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2


class CliUsageError(ValueError):
    pass


def exit_from_error(error: BaseException) -> NoReturn:
    if isinstance(error, CliUsageError):
        raise Exit(EXIT_USAGE) from error
    typer.echo(f"error: {error}", err=True)
    raise Exit(EXIT_RUNTIME) from error
