from __future__ import annotations

from pathlib import Path
from typing import Annotated

import torch
import typer

from fedorbit.analysis.synthesis import StudyAnalysis
from fedorbit.config.loading import REPOSITORY_ROOT, config_digest, load_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.preparation import device_sources
from fedorbit.infrastructure.runtime import code_revision, configure_logging
from fedorbit.pipeline.execution import (
    ExecutionContext,
    analysis_provenance,
    analysis_state,
    build_context,
    execute_analysis,
    execute_experiment,
    experiment_state,
    prepare_dataset,
)
from fedorbit.pipeline.smoke import smoke_config
from fedorbit.reporting import write_report
from fedorbit.types import (
    CliCommand,
    DatasetId,
    EvidenceState,
    ExperimentId,
    OverwritePolicy,
)

PACKAGE_DIRECTORY = Path(__file__).resolve().parent
SMOKE_DIRECTORY = "smoke"
SMOKE_RESULTS = "results"
RESULTS_DIRECTORY = "results"

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _context(config: FedorbitConfig, smoke_run: bool = False) -> ExecutionContext:
    workspace = REPOSITORY_ROOT / config.workspace_directory
    if smoke_run:
        workspace = workspace / SMOKE_DIRECTORY
    results = workspace / SMOKE_RESULTS if smoke_run else REPOSITORY_ROOT / RESULTS_DIRECTORY
    return build_context(config, REPOSITORY_ROOT, PACKAGE_DIRECTORY, workspace, results)


def _overwrite(flag: bool) -> OverwritePolicy:
    return OverwritePolicy.OVERWRITE if flag else OverwritePolicy.REUSE


@app.command(CliCommand.DOCTOR.value)
def doctor() -> None:
    config = load_config()
    context = _context(config)
    typer.echo(f"configuration digest {config_digest(config)}")
    typer.echo(f"code revision {code_revision(REPOSITORY_ROOT)}")
    typer.echo(f"cuda available {torch.cuda.is_available()}")
    failures = 0
    for dataset in DatasetId:
        try:
            sources = device_sources(config, context.raw_directory, dataset)
        except (FileNotFoundError, ValueError) as error:
            typer.echo(f"{dataset.value}: {error}")
            failures += 1
            continue
        typer.echo(f"{dataset.value}: {len(sources)} devices with source files present")
    raise typer.Exit(code=1 if failures else 0)


@app.command(CliCommand.PREPROCESS.value)
def preprocess(
    dataset: Annotated[DatasetId | None, typer.Argument()] = None, overwrite: bool = False
) -> None:
    configure_logging()
    config = load_config()
    context = _context(config)
    for selected in [dataset] if dataset is not None else list(DatasetId):
        prepare_dataset(config, context, selected, _overwrite(overwrite))


@app.command(CliCommand.PLAN.value)
def plan() -> None:
    config = load_config()
    for experiment in config.experiments:
        typer.echo(
            f"{experiment.id.value}: dataset={experiment.dataset.value} "
            f"detector={experiment.detector.value} support={list(experiment.support_sizes)} "
            f"replicates={experiment.replicates} conditions={len(experiment.conditions)} "
            f"contrasts={len(experiment.contrasts)}"
        )


@app.command(CliCommand.STATUS.value)
def status(experiment: Annotated[ExperimentId | None, typer.Argument()] = None) -> None:
    config = load_config()
    context = _context(config)
    for declared in config.experiments:
        if experiment is None or declared.id is experiment:
            typer.echo(f"{declared.id.value}: {experiment_state(config, context, declared).value}")
    typer.echo(f"analysis: {analysis_state(config, context).value}")


@app.command(CliCommand.RUN.value)
def run(experiment: ExperimentId, overwrite: bool = False) -> None:
    configure_logging()
    config = load_config()
    state = execute_experiment(config, _context(config), experiment, _overwrite(overwrite))
    typer.echo(f"{experiment.value}: {state.value}")


def _report(config: FedorbitConfig, context: ExecutionContext, overwrite: bool) -> StudyAnalysis:
    analysis = execute_analysis(config, context, _overwrite(overwrite))
    write_report(
        config,
        analysis,
        context.workspace.results_directory,
        analysis_provenance(config, context),
        code_revision(REPOSITORY_ROOT),
    )
    return analysis


@app.command(CliCommand.REPORT.value)
def report(overwrite: bool = False) -> None:
    configure_logging()
    config = load_config()
    analysis = _report(config, _context(config), overwrite)
    typer.echo(f"contrasts reported: {len(analysis.contrasts)}")


@app.command(CliCommand.SMOKE.value)
def smoke(overwrite: bool = False) -> None:
    configure_logging()
    config = smoke_config(load_config())
    context = _context(config, smoke_run=True)
    policy = _overwrite(overwrite)
    prepare_dataset(config, context, DatasetId.NBAIOT, policy)
    for experiment in config.experiments:
        state = execute_experiment(config, context, experiment.id, policy)
        if state is not EvidenceState.VALID:
            raise typer.Exit(code=1)
    analysis = _report(config, context, overwrite)
    typer.echo(f"smoke contrasts: {len(analysis.contrasts)}")


def main() -> None:
    app()
