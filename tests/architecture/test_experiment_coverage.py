from __future__ import annotations

from pathlib import Path

from fedorbit.cli import app
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.dispatch import (
    ExperimentExecutionRequest,
    _registered_experiment_producers,
)
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import CliCommand, ExperimentName, OverwritePolicy


def test_experiment_enum_matches_catalogue() -> None:
    catalogue = build_catalogue()
    assert set(catalogue.registered_names()) == set(ExperimentName)


def test_catalogue_matches_registered_producers(tmp_path: Path) -> None:
    catalogue = build_catalogue()
    experiment = ExperimentName.STATISTICAL_SYNTHESIS
    producers = _registered_experiment_producers(
        ArtifactStore(tmp_path),
        build_layout(),
        ExperimentExecutionRequest(
            experiment=experiment,
            definition=catalogue.definition(experiment),
            overwrite_policy=OverwritePolicy.REUSE,
        ),
    )
    assert set(producers) == set(ExperimentName)


def test_cli_exposes_run_for_registered_experiments() -> None:
    command_names = {command.name for command in app.registered_commands}
    assert CliCommand.RUN.value in command_names
    assert set(build_catalogue().registered_names()) == set(ExperimentName)
