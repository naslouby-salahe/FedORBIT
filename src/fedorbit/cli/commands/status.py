from __future__ import annotations

import typer

from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.artifacts.storage import ArtifactStore
from fedorbit.cli.errors import CliUsageError, exit_from_error
from fedorbit.cli.parsing import experiment_identifier
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.domain.enums import ArtifactState, ExperimentName
from fedorbit.execution.executor import execution_store
from fedorbit.execution.planner import build_plan


def _verified_manifest(
    store: ArtifactStore,
    experiment: ExperimentName,
) -> ReusableArtifactManifest | None:
    for manifest in store.all_manifests():
        if experiment.value not in manifest.semantic_producer_coordinates:
            continue
        try:
            resolved = store.resolve(manifest.artifact_id)
        except ValueError:
            continue
        if resolved.state == ArtifactState.COMPLETED:
            return resolved
    return None


def status(experiment_name: str | None = typer.Argument(None)) -> None:
    try:
        rows = build_plan(load_fedorbit_config())
        selected = (
            {experiment_identifier(experiment_name)}
            if experiment_name is not None
            else {row.experiment for row in rows}
        )
        store = execution_store()
        typer.echo(
            f"{'#':>2} {'Experiment':<50} {'Role':<22} {'Status':<10} {'Est-run':<8} {'Est-end':<8}"
        )
        index = 0
        for row in rows:
            if row.experiment not in selected:
                continue
            status_value = (
                "completed" if _verified_manifest(store, row.experiment) is not None else "pending"
            )
            typer.echo(
                f"{index:>2} {row.experiment.value:<50} {row.classification.value:<22} "
                f"{status_value:<10} {'-':<8} {'-':<8}"
            )
            index += 1
    except CliUsageError as error:
        exit_from_error(error)
