from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import NoReturn, cast

import typer
from typer import Argument, Exit

from fedorbit.analysis.records import MetricRecord
from fedorbit.config.loading import active_config
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.infrastructure.environment import (
    EnvironmentMismatchError,
    environment_snapshot,
    reference_gpu_matches,
    validate_lockfile,
)
from fedorbit.infrastructure.execution import (
    ArtifactStore,
    DatasetPreparationRequest,
    ExecutionError,
    ExperimentExecutionRequest,
    execution_store,
    preprocess_datasets,
    run_experiment,
    run_smoke_validation,
)
from fedorbit.infrastructure.failures import validation_failure_outcome
from fedorbit.infrastructure.manifests import ReusableArtifactManifest
from fedorbit.infrastructure.planner import build_plan
from fedorbit.infrastructure.workspace import build_layout, safe_slug
from fedorbit.reporting import VerifiedEvidenceWriter
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactState,
    DatasetId,
    ExperimentName,
    OverwritePolicy,
    StableJsonPayload,
)

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2


class CliUsageError(ValueError):
    pass


def exit_from_error(error: BaseException) -> NoReturn:
    if isinstance(error, CliUsageError):
        raise Exit(EXIT_USAGE) from error
    if isinstance(error, ExecutionError):
        outcome = validation_failure_outcome(str(error), invalid=False)
        typer.echo(f"error [{outcome.terminal_state.value}]: {error}", err=True)
    else:
        typer.echo(f"error: {error}", err=True)
    raise Exit(EXIT_RUNTIME) from error


def dataset_identifier(name: str) -> DatasetId:
    for candidate in DatasetId:
        if candidate.value == name:
            return candidate
    raise CliUsageError(
        f"unknown dataset identifier {name!r}: use the exact registered "
        "identifier (display names, filesystem names, aliases, and source-dataset "
        "names such as Edge-IIoTset or ToN-IoT are not accepted)"
    )


def experiment_identifier(name: str) -> ExperimentName:
    for candidate in ExperimentName:
        if candidate.value == name:
            return candidate
    raise CliUsageError(
        f"unknown experiment name {name!r}: use the exact registered experiment name"
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


def plan() -> None:
    rows = build_plan()
    typer.echo(f"registered experiments: {len(rows)}")
    for row in rows:
        typer.echo(
            f"{row.experiment.value} | {row.classification.value} | "
            f"planned cells: {row.planned_cells}"
        )


def preprocess(
    dataset_name: str | None = Argument(None),
    overwrite: bool = False,
) -> None:
    try:
        selected = (
            (dataset_identifier(dataset_name),)
            if dataset_name is not None
            else _registered_datasets()
        )
        result = preprocess_datasets(
            DatasetPreparationRequest(
                datasets=selected,
                overwrite_policy=OverwritePolicy.REPLACE if overwrite else OverwritePolicy.REUSE,
            )
        )
        for observation in result.observations:
            event_time = observation.event_time
            state = "ready" if observation.valid_for_chronological_preprocessing else "blocked"
            typer.echo(
                f"{observation.dataset.value}: {state} | chronology={event_time.state.value} | "
                f"reason={event_time.reason}"
            )
    except (CliUsageError, ExecutionError) as error:
        exit_from_error(error)


def _registered_datasets() -> tuple[DatasetId, ...]:
    return tuple(active_config().scientific.datasets.clients.keys())


def _verified_manifest(
    store: ArtifactStore,
    experiment: ExperimentName,
) -> ReusableArtifactManifest | None:
    candidates: list[ReusableArtifactManifest] = []
    for manifest in store.all_manifests():
        if experiment.value not in manifest.semantic_producer_coordinates:
            continue
        try:
            resolved = store.resolve(ArtifactIdentifier(manifest.artifact_id))
        except ValueError:
            continue
        if resolved.state == ArtifactState.COMPLETED:
            candidates.append(resolved)
    if not candidates:
        return None
    return max(candidates, key=_manifest_payload_mtime_ns)


def _manifest_payload_mtime_ns(manifest: ReusableArtifactManifest) -> int:
    return max(Path(payload).stat().st_mtime_ns for payload in manifest.payload_paths)


def _blocked_experiment(layout_root: Path, experiment: ExperimentName) -> bool:
    return (
        layout_root
        / "experiments"
        / safe_slug(experiment.value)
        / "artifacts"
        / "derived"
        / "blocked.json"
    ).is_file()


def report(
    experiment_name: str | None = Argument(None),
    overwrite: bool = False,
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
        exported_manifests: list[ReusableArtifactManifest] = []
        exported_metrics: list[MetricRecord] = []
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
            for metric_path in writer.write_metric_exports(
                experiment,
                ArtifactIdentifier(manifest.artifact_id),
            ):
                typer.echo(str(metric_path))
            metric = writer.metric_record(ArtifactIdentifier(manifest.artifact_id))
            if metric is not None:
                exported_metrics.append(metric)
            exported_manifests.append(manifest)
            exported += 1
        if experiment_name is None:
            for summary_path in writer.write_project_summary(
                tuple(exported_manifests),
                tuple(exported_metrics),
            ):
                typer.echo(str(summary_path))
        if exported == 0:
            typer.echo("no verified persisted evidence available for report generation")
    except CliUsageError as error:
        exit_from_error(error)


def run(
    experiment_name: str,
    overwrite: bool = False,
) -> None:
    try:
        experiment = experiment_identifier(experiment_name)
        definition = build_catalogue().definition(experiment)
        run_experiment(
            ExperimentExecutionRequest(
                experiment=experiment,
                definition=definition,
                overwrite_policy=OverwritePolicy.REPLACE if overwrite else OverwritePolicy.REUSE,
            )
        )
    except (CliUsageError, ExecutionError) as error:
        exit_from_error(error)


def smoke(overwrite: bool = False) -> None:
    try:
        run_smoke_validation(OverwritePolicy.REPLACE if overwrite else OverwritePolicy.REUSE)
    except (CliUsageError, ExecutionError) as error:
        exit_from_error(error)


def status(experiment_name: str | None = Argument(None)) -> None:
    try:
        rows = build_plan()
        selected = (
            {experiment_identifier(experiment_name)}
            if experiment_name is not None
            else {row.experiment for row in rows}
        )
        store = execution_store()
        layout = build_layout()
        typer.echo(
            f"{'#':>2} {'Experiment':<50} {'Role':<22} {'Status':<10} {'Est-run':<8} {'Est-end':<8}"
        )
        index = 0
        for row in rows:
            if row.experiment not in selected:
                continue
            status_value = "pending"
            if _verified_manifest(store, row.experiment) is not None:
                status_value = "completed"
            elif _blocked_experiment(layout.execution_root, row.experiment):
                status_value = "blocked"
            typer.echo(
                f"{index:>2} {row.experiment.value:<50} {row.classification.value:<22} "
                f"{status_value:<10} {'-':<8} {'-':<8}"
            )
            index += 1
    except CliUsageError as error:
        exit_from_error(error)


app = typer.Typer(name="fedorbit", no_args_is_help=True)
app.command("doctor")(doctor)
app.command("preprocess")(preprocess)
app.command("plan")(plan)
app.command("smoke")(smoke)
app.command("run")(run)
app.command("status")(status)
app.command("report")(report)


def main() -> None:
    raise SystemExit(app())


if __name__ == "__main__":
    main()
