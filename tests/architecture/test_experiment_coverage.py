from __future__ import annotations

from pathlib import Path

from fedorbit.cli import app
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.dispatch import (
    ExperimentExecutionRequest,
    registered_experiment_producers,
)
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import CliCommand, ExperimentName, OverwritePolicy, SemanticCoordinate


def test_experiment_enum_matches_catalogue() -> None:
    catalogue = build_catalogue()
    assert set(catalogue.registered_names()) == set(ExperimentName)


def test_catalogue_matches_registered_producers(tmp_path: Path) -> None:
    catalogue = build_catalogue()
    experiment = ExperimentName.STATISTICAL_SYNTHESIS
    producers = registered_experiment_producers(
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


def test_every_registered_experiment_declares_cell_relevance_coordinates() -> None:
    from fedorbit.experiments.cells import CELL_RELEVANCE_BY_EXPERIMENT

    assert set(CELL_RELEVANCE_BY_EXPERIMENT) == set(ExperimentName)
    for experiment, relevance in CELL_RELEVANCE_BY_EXPERIMENT.items():
        assert relevance, f"{experiment.value} declares an empty cell relevance set"
        assert SemanticCoordinate.EXPERIMENT in relevance
        assert SemanticCoordinate.SEED in relevance
