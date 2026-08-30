from __future__ import annotations

from pathlib import Path

import typer
from typer import Exit

from fedorbit.cli.errors import EXIT_RUNTIME, CliUsageError
from fedorbit.runtime.environment import (
    EnvironmentMismatchError,
    environment_snapshot,
    reference_gpu_matches,
    validate_lockfile,
)


def doctor() -> None:
    try:
        snapshot = environment_snapshot()
        lockfile = validate_lockfile()
        gpu_ok = reference_gpu_matches()
        raw_root = Path("data/raw")
        typer.echo(f"python: {snapshot.python_version}")
        typer.echo(f"dependencies: {len(snapshot.dependencies)} registered")
        typer.echo(f"lockfile packages: {lockfile.hashed_package_count} hashed")
        typer.echo(f"reference gpu matches: {gpu_ok}")
        typer.echo(f"raw data root present: {raw_root.is_dir()}")
    except (EnvironmentMismatchError, CliUsageError) as error:
        typer.echo(f"environment mismatch: {error}")
        raise Exit(EXIT_RUNTIME) from error
    if not gpu_ok or not raw_root.is_dir():
        raise Exit(EXIT_RUNTIME)
