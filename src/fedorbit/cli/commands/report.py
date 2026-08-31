from __future__ import annotations

from collections import OrderedDict
from typing import cast

import typer

from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.artifacts.paths import build_layout
from fedorbit.artifacts.storage import ArtifactStore
from fedorbit.cli.errors import CliUsageError, exit_from_error
from fedorbit.cli.parsing import experiment_identifier
from fedorbit.domain.enums import ArtifactState, ExperimentName
from fedorbit.domain.records import ArtifactIdentifier
from fedorbit.domain.serialization import StableJsonPayload
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.reporting.export import VerifiedEvidenceWriter


def _verified_manifest(
    store: ArtifactStore,
    experiment: ExperimentName,
) -> ReusableArtifactManifest | None:
    for manifest in store.all_manifests():
        if experiment.value not in manifest.semantic_producer_coordinates:
            continue
        try:
            resolved = store.resolve(ArtifactIdentifier(manifest.artifact_id))
        except ValueError:
            continue
        if resolved.state == ArtifactState.COMPLETED:
            return resolved
    return None


def report(
    experiment_name: str | None = typer.Argument(None),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    try:
        catalogue = build_catalogue()
        selected = (
            (experiment_identifier(experiment_name),)
            if experiment_name is not None
            else catalogue.registered_names()
        )
        layout = build_layout()
        store = ArtifactStore(layout.execution_root)
        writer = VerifiedEvidenceWriter(store, layout)
        exported = 0
        for experiment in selected:
            manifest = _verified_manifest(store, experiment)
            if manifest is None:
                continue
            destination = writer.write(
                experiment,
                ArtifactIdentifier(manifest.artifact_id),
                cast(
                    StableJsonPayload,
                    OrderedDict(
                        experiment=experiment.value,
                        artifact_id=manifest.artifact_id,
                        state=manifest.state.value,
                        dependency_fingerprint_sha256=manifest.dependency_fingerprint_sha256,
                    ),
                ),
                overwrite=overwrite,
            )
            typer.echo(str(destination))
            exported += 1
        if exported == 0:
            typer.echo("no verified persisted evidence available for report generation")
    except CliUsageError as error:
        exit_from_error(error)
