from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn, cast

import structlog
import typer
from typer import Argument, Exit

from fedorbit.analysis.records import MetricRecord
from fedorbit.config.loading import active_config, raw_dataset_root
from fedorbit.experiments.catalogue import ExperimentCatalogue, build_catalogue
from fedorbit.infrastructure.environment import (
    environment_snapshot,
    reference_gpu_matches,
)
from fedorbit.infrastructure.execution import (
    ArtifactStore,
    DatasetPreparationRequest,
    ExecutionError,
    ExperimentExecutionRequest,
    completed_primary_transfer_comparison_records,
    completed_primary_transfer_metric_records,
    execution_store,
    preprocess_datasets,
    run_experiment,
    run_smoke_validation,
)
from fedorbit.infrastructure.failures import validation_failure_outcome
from fedorbit.infrastructure.manifests import DatasetManifest, ReusableArtifactManifest
from fedorbit.infrastructure.runtime import current_code_revision
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
    build_layout,
    experiment_workspace,
    safe_slug,
)
from fedorbit.reporting import (
    FigureSeries,
    TableScalar,
    VerifiedEvidenceWriter,
    dataset_and_client_protocol_table,
    experiment_matrix_table,
    numerical_constants_and_seeds_table,
    primary_strict_transfer_results_table,
    real_transfer_gain_forest_plot,
)
from fedorbit.types import (
    ArtifactState,
    CliCommand,
    ClientRole,
    DatasetId,
    DatasetIdentifierText,
    DirectedPairName,
    ExitStatus,
    ExperimentIdentifierText,
    ExperimentName,
    FailureReason,
    OverwritePolicy,
    RelativeGain,
    ReportArtifactName,
    ReportSeriesName,
    StableJsonPayload,
    TransferMethod,
)


class CliUsageError(ValueError):
    pass


OPTIONAL_ARGUMENT = Argument(None)


def exit_from_error(error: BaseException) -> NoReturn:
    if isinstance(error, CliUsageError):
        raise Exit(ExitStatus.USAGE) from error
    if isinstance(error, ExecutionError):
        outcome = validation_failure_outcome(FailureReason(str(error)), invalid=False)
        typer.echo(f"error [{outcome.terminal_state.value}]: {error}", err=True)
    else:
        typer.echo(f"error: {error}", err=True)
    raise Exit(ExitStatus.RUNTIME) from error


def dataset_identifier(name: DatasetIdentifierText) -> DatasetId:
    for candidate in DatasetId:
        if candidate.value == name:
            return candidate
    raise CliUsageError(
        f"unknown dataset identifier {name!r}: use the exact registered "
        "identifier (display names, filesystem names, aliases, and source-dataset "
        "names such as Edge-IIoTset or ToN-IoT are not accepted)"
    )


def experiment_identifier(name: ExperimentIdentifierText) -> ExperimentName:
    for candidate in ExperimentName:
        if candidate.value == name:
            return candidate
    raise CliUsageError(
        f"unknown experiment name {name!r}: use the exact registered experiment name"
    )


def doctor() -> None:
    snapshot = environment_snapshot()
    gpu_ok = reference_gpu_matches()
    raw_root = raw_dataset_root()
    typer.echo(f"python: {snapshot.python_version}")
    typer.echo(f"dependencies: {len(snapshot.dependencies)} registered")
    typer.echo(f"reference gpu matches: {gpu_ok}")
    typer.echo(f"raw data root present: {raw_root.is_dir()}")
    if not gpu_ok or not raw_root.is_dir():
        raise Exit(ExitStatus.RUNTIME)


def plan() -> None:
    catalogue = build_catalogue()
    names = catalogue.registered_names()
    typer.echo(f"registered experiments: {len(names)}")
    for name in names:
        definition = catalogue.definition(name)
        typer.echo(
            f"{name.value} | {definition.classification.value} | "
            f"planned cells: {definition.derived_planned_cells}"
        )


def preprocess(
    dataset_name: DatasetId | None = OPTIONAL_ARGUMENT,
    overwrite: bool = False,
) -> None:
    try:
        selected = (dataset_name,) if dataset_name is not None else _registered_datasets()
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
        for dataset, reason in result.resource_blocked_datasets:
            typer.echo(f"{dataset.value}: blocked | reason={reason}")
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
            resolved = store.resolve(manifest.artifact_id)
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


def _experiment_state(
    store: ArtifactStore,
    layout_root: Path,
    experiment: ExperimentName,
) -> ArtifactState:
    if _blocked_experiment(layout_root, experiment):
        return ArtifactState.BLOCKED
    candidates = tuple(
        manifest
        for manifest in store.all_manifests()
        if experiment.value in manifest.semantic_producer_coordinates
    )
    if not candidates:
        return ArtifactState.MISSING
    latest = max(candidates, key=_manifest_payload_mtime_ns)
    try:
        resolved = store.resolve(latest.artifact_id)
    except ValueError:
        return ArtifactState.INVALID
    if resolved.created_git_commit != current_code_revision().commit:
        return ArtifactState.STALE
    return resolved.state


def _base_model_pilot_dataset_manifests(layout: WorkspaceLayout) -> tuple[DatasetManifest, ...]:
    workspace = experiment_workspace(layout, ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT)
    manifests: list[DatasetManifest] = []
    for dataset in DatasetId:
        path = workspace / "artifacts" / "derived" / f"dataset-manifest.{dataset.value}.json"
        if path.is_file():
            manifests.append(DatasetManifest.model_validate_json(path.read_text(encoding="utf-8")))
    return tuple(manifests)


def _dataset_client_roles() -> Mapping[str, ClientRole]:
    return OrderedDict(
        (dataset.value, client.role)
        for dataset, client in active_config().scientific.datasets.clients.items()
    )


def _experiment_matrix_rows(
    catalogue: ExperimentCatalogue,
) -> tuple[Mapping[str, TableScalar], ...]:
    rows: list[Mapping[str, TableScalar]] = []
    for name in catalogue.registered_names():
        definition = catalogue.definition(name)
        rows.append(
            OrderedDict(
                experiment=name.value,
                classification=definition.classification.value,
                datasets_or_pairs=", ".join(scope.value for scope in definition.datasets_or_pairs),
                methods=", ".join(str(method) for method in definition.methods),
                registered_seeds=", ".join(str(seed) for seed in definition.seeds),
                conditions=str(definition.conditions),
                derived_planned_cells=int(definition.derived_planned_cells),
                prerequisites=", ".join(
                    str(prerequisite) for prerequisite in definition.prerequisites
                ),
                evidence_relationship=None,
            )
        )
    return tuple(rows)


def _real_transfer_gain_series(
    comparisons: Mapping[DirectedPairName, RelativeGain],
) -> tuple[FigureSeries, ...]:
    if not comparisons:
        return ()
    pairs = sorted(comparisons)
    return (
        FigureSeries(
            name=ReportSeriesName(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value),
            x=tuple(comparisons[pair] for pair in pairs),
            y=tuple(float(index) for index in range(len(pairs))),
        ),
    )


def report(
    experiment_name: ExperimentName | None = OPTIONAL_ARGUMENT,
    overwrite: bool = False,
) -> None:
    try:
        catalogue = build_catalogue()
        selected = (
            (experiment_name,) if experiment_name is not None else catalogue.registered_names()
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
                manifest.artifact_id,
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
                manifest.artifact_id,
            ):
                typer.echo(str(metric_path))
            metric = writer.metric_record(manifest.artifact_id)
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
            for table, name in (
                (
                    numerical_constants_and_seeds_table(),
                    ReportArtifactName("numerical-constants-and-seeds"),
                ),
                (
                    experiment_matrix_table(_experiment_matrix_rows(catalogue)),
                    ReportArtifactName("experiment-matrix"),
                ),
                (
                    dataset_and_client_protocol_table(
                        _base_model_pilot_dataset_manifests(layout),
                        modality_by_dataset=OrderedDict(),
                        role_by_dataset=_dataset_client_roles(),
                        excluded_class_counts=OrderedDict(),
                    ),
                    ReportArtifactName("dataset-and-client-protocol"),
                ),
            ):
                typer.echo(str(writer.write_project_evidence_table(table, name)))
            primary_transfer_metrics = completed_primary_transfer_metric_records(store)
            primary_transfer_comparisons = completed_primary_transfer_comparison_records(store)
            typer.echo(
                str(
                    writer.write_project_evidence_table(
                        primary_strict_transfer_results_table(
                            primary_transfer_metrics, primary_transfer_comparisons
                        ),
                        ReportArtifactName("primary-strict-transfer-results"),
                    )
                )
            )
            gain_by_pair = OrderedDict(
                (comparison.pair, comparison.mean_difference)
                for comparison in primary_transfer_comparisons
                if comparison.method_a == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
                and comparison.method_b == TransferMethod.LOCAL_ONLY
                and comparison.mean_difference is not None
            )
            gain_series = _real_transfer_gain_series(gain_by_pair)
            if gain_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            real_transfer_gain_forest_plot(gain_series),
                            ReportArtifactName("real-transfer-gain-forest-plot"),
                        )
                    )
                )
        if exported == 0:
            typer.echo("no verified persisted evidence available for report generation")
    except CliUsageError as error:
        exit_from_error(error)


def run(
    experiment_name: ExperimentName,
    overwrite: bool = False,
) -> None:
    try:
        definition = build_catalogue().definition(experiment_name)
        run_experiment(
            ExperimentExecutionRequest(
                experiment=experiment_name,
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


def status(experiment_name: ExperimentName | None = OPTIONAL_ARGUMENT) -> None:
    try:
        catalogue = build_catalogue()
        names = catalogue.registered_names()
        selected = {experiment_name} if experiment_name is not None else set(names)
        store = execution_store()
        layout = build_layout()
        typer.echo(
            f"{'#':>2} {'Experiment':<50} {'Role':<22} {'Status':<10} {'Est-run':<8} {'Est-end':<8}"
        )
        index = 0
        for name in names:
            if name not in selected:
                continue
            definition = catalogue.definition(name)
            status_value = _experiment_state(store, layout.execution_root, name)
            typer.echo(
                f"{index:>2} {name.value:<50} {definition.classification.value:<22} "
                f"{status_value.value:<10} {'-':<8} {'-':<8}"
            )
            index += 1
    except CliUsageError as error:
        exit_from_error(error)


app = typer.Typer(name="fedorbit", no_args_is_help=True)
app.command(CliCommand.DOCTOR.value)(doctor)
app.command(CliCommand.PREPROCESS.value)(preprocess)
app.command(CliCommand.PLAN.value)(plan)
app.command(CliCommand.SMOKE.value)(smoke)
app.command(CliCommand.RUN.value)(run)
app.command(CliCommand.STATUS.value)(status)
app.command(CliCommand.REPORT.value)(report)


def _configure_execution_logging() -> None:
    structlog.configure(
        processors=(
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(sort_keys=True),
        ),
        wrapper_class=structlog.make_filtering_bound_logger(20),
    )


def main() -> None:
    _configure_execution_logging()
    raise SystemExit(app())


if __name__ == "__main__":
    main()
