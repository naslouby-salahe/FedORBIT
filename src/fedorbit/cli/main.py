from __future__ import annotations

from pathlib import Path

import typer
from typer import Exit

from fedorbit.cli.errors import CliUsageError, NotReadyError
from fedorbit.cli.parsing import dataset_identifier, experiment_identifier
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.testing import load_smoke_config, load_tests_config
from fedorbit.domain.enums import DatasetId
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.runtime.environment import (
    EnvironmentMismatchError,
    environment_snapshot,
    reference_gpu_matches,
    validate_lockfile,
)

app = typer.Typer(name="fedorbit", no_args_is_help=True)

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2
EXIT_NOT_READY = 3


def _translate(error: BaseException) -> None:
    if isinstance(error, CliUsageError):
        raise Exit(EXIT_USAGE) from error
    if isinstance(error, NotReadyError):
        typer.echo(f"not ready: {error}", err=True)
        raise Exit(EXIT_NOT_READY) from error
    typer.echo(f"error: {error}", err=True)
    raise Exit(EXIT_RUNTIME) from error


@app.command()
def doctor() -> None:
    config = load_fedorbit_config()
    try:
        snapshot = environment_snapshot(config)
        lockfile = validate_lockfile(config)
        gpu_ok = reference_gpu_matches(config)
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


@app.command()
def preprocess(
    dataset_name: str | None = typer.Argument(None),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    try:
        selected = (
            (dataset_identifier(dataset_name),)
            if dataset_name is not None
            else tuple(_registered_datasets())
        )
        from fedorbit.execution.pipeline import preprocess_pipeline

        preprocess_pipeline(selected, overwrite=overwrite)
    except (CliUsageError, NotReadyError) as error:
        _translate(error)


@app.command()
def plan() -> None:
    config = load_fedorbit_config()
    catalogue = build_catalogue(config)
    typer.echo(f"registered experiments: {len(catalogue)}")
    for name, definition in catalogue.items():
        typer.echo(
            f"{name.value} | {definition.classification.value} | "
            f"planned cells: {definition.derived_planned_cells}"
        )


@app.command()
def smoke(overwrite: bool = typer.Option(False, "--overwrite")) -> None:
    try:
        load_tests_config()
        load_smoke_config()
        from fedorbit.execution.pipeline import smoke_pipeline

        smoke_pipeline(overwrite=overwrite)
    except (CliUsageError, NotReadyError) as error:
        _translate(error)


@app.command()
def run(
    experiment_name: str,
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    try:
        experiment = experiment_identifier(experiment_name)
        config = load_fedorbit_config()
        catalogue = build_catalogue(config)
        definition = catalogue[experiment]
        from fedorbit.execution.pipeline import run_pipeline

        run_pipeline(experiment, definition, overwrite=overwrite)
    except (CliUsageError, NotReadyError) as error:
        _translate(error)


@app.command()
def status(experiment_name: str | None = typer.Argument(None)) -> None:
    try:
        config = load_fedorbit_config()
        catalogue = build_catalogue(config)
        selected = (
            {experiment_identifier(experiment_name)}
            if experiment_name is not None
            else set(catalogue)
        )
        typer.echo(
            f"{'#':>2} {'experiment':<50} {'role':<22} {'status':<10} {'est-run':<8} {'est-end':<8}"
        )
        for index, (name, definition) in enumerate(
            (name, definition) for name, definition in catalogue.items() if name in selected
        ):
            typer.echo(
                f"{index:>2} {name.value:<50} {definition.classification.value:<22} "
                f"{'pending':<10} {'-':<8} {'-':<8}"
            )
    except CliUsageError as error:
        _translate(error)


@app.command()
def report(
    experiment_name: str | None = typer.Argument(None),
    _overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    try:
        if experiment_name is not None:
            experiment_identifier(experiment_name)
        typer.echo("no verified persisted evidence available for report generation")
    except CliUsageError as error:
        _translate(error)


def _registered_datasets() -> tuple[DatasetId, ...]:
    config = load_fedorbit_config()
    return tuple(config.scientific.datasets.clients.keys())


def main() -> None:
    raise SystemExit(app())
