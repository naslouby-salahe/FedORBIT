from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

import structlog
import typer
from typer import Argument, Exit

from fedorbit.analysis.records import MetricRecord
from fedorbit.config.loading import active_config, raw_dataset_root
from fedorbit.experiments.catalogue import (
    CataloguePrerequisiteValue,
    ExperimentDefinition,
    build_catalogue,
)
from fedorbit.experiments.classification import completed_evidence_status_rows
from fedorbit.experiments.dispatch import (
    ExperimentExecutionRequest,
    run_experiment,
    run_smoke_validation,
)
from fedorbit.experiments.report_rows import (
    ablation_results_rows,
    base_model_pilot_dataset_manifests,
    baseline_paired_difference_series,
    confirmation_figure_series,
    confirmation_results_rows,
    coupling_mechanism_results_rows,
    dataset_client_roles,
    dataset_modality_by_dataset,
    evidence_status_rows,
    exact_solver_results_rows,
    excluded_class_counts,
    experiment_matrix_rows,
    failure_boundary_figure_series,
    failure_boundary_results_rows,
    generalization_results_rows,
    map_value_bound_series,
    real_transfer_gain_series,
    scalability_figure_series,
    scalability_results_rows,
    semantic_sufficiency_series,
    sparsity_and_dense_results_rows,
    sparsity_figure_series,
    transfer_ontology_and_null_padding_rows,
)
from fedorbit.experiments.synthesis import (
    completed_experiment_metric_records,
    completed_experiment_metric_records_with_support,
    completed_primary_transfer_comparison_records,
    completed_primary_transfer_metric_records,
)
from fedorbit.experiments.training import training_protocol_rows
from fedorbit.infrastructure.artifacts import (
    ArtifactStateReport,
    ArtifactStore,
    ExecutionError,
    execution_store,
)
from fedorbit.infrastructure.environment import (
    environment_snapshot,
    reference_gpu_matches,
)
from fedorbit.infrastructure.evidence import recorded_execution_identity
from fedorbit.infrastructure.failures import validation_failure_outcome
from fedorbit.infrastructure.manifests import ReusableArtifactManifest, recorded_experiment
from fedorbit.infrastructure.preparation import DatasetPreparationRequest, preprocess_datasets
from fedorbit.infrastructure.runtime import (
    build_reproducibility_identity,
    compatible,
    execution_logger,
)
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
    build_layout,
    safe_slug,
)
from fedorbit.reporting import (
    VerifiedEvidenceWriter,
    ablation_results_table,
    baseline_paired_difference_plot,
    confirmation_results_table,
    confirmation_safety_coverage_figure,
    coupling_gap_phase_figure,
    coupling_mechanism_results_table,
    dataset_and_client_protocol_table,
    evidence_status_table,
    exact_solver_results_table,
    experiment_matrix_table,
    failure_boundary_figure,
    failure_boundary_results_table,
    generalization_results_table,
    information_resource_matrix_table,
    map_value_bound_figure,
    model_and_training_protocol_table,
    numerical_constants_and_seeds_table,
    predicted_vs_realized_transfer_figure,
    primary_strict_transfer_results_table,
    real_transfer_gain_forest_plot,
    scalability_figure,
    scalability_results_table,
    semantic_sufficiency_frontier_figure,
    sparsity_and_dense_results_table,
    sparsity_utility_efficiency_figure,
    transfer_ontology_and_null_padding_table,
)
from fedorbit.types import (
    ArtifactState,
    CliCommand,
    DatasetId,
    ExecutionEventName,
    ExitStatus,
    ExperimentName,
    FailureReason,
    FieldDescription,
    Index,
    OverwritePolicy,
    PreprocessingReadiness,
    ReportArtifactName,
    StableJsonPayload,
    StorageLayoutSegment,
)


class CliUsageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExperimentStateReport:
    state: ArtifactState
    reason: FieldDescription


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


def doctor() -> None:
    snapshot = environment_snapshot()
    gpu_ok = reference_gpu_matches()
    raw_root = raw_dataset_root()
    layout = build_layout()
    recorded_identity = recorded_execution_identity(layout)
    identity_compatible = recorded_identity is None or compatible(
        build_reproducibility_identity(snapshot), recorded_identity
    )
    typer.echo(f"python: {snapshot.python_version}")
    typer.echo(f"recorded execution identity compatible: {identity_compatible}")
    typer.echo(f"reference gpu matches: {gpu_ok}")
    typer.echo(f"raw data root present: {raw_root.is_dir()}")
    if not gpu_ok or not raw_root.is_dir():
        raise Exit(ExitStatus.RUNTIME)


def plan() -> None:
    catalogue = build_catalogue()
    names = catalogue.registered_names()
    store = execution_store()
    layout = build_layout()
    typer.echo(f"registered experiments: {len(names)}")
    for name in names:
        definition = catalogue.definition(name)
        prerequisite_states = _plan_prerequisite_states(store, layout, definition.prerequisites)
        resume_boundary = _plan_resume_boundary(name, prerequisite_states)
        typer.echo(
            f"{name.value} | {definition.classification.value} | "
            f"planned cells: {definition.derived_planned_cells}"
        )
        typer.echo(
            "  semantic scope: "
            f"pairs={','.join(scope.value for scope in definition.datasets_or_pairs) or 'none'}; "
            f"methods={','.join(method.value for method in definition.methods) or 'none'}; "
            f"conditions={_planned_condition_count(definition)}; seeds={len(definition.seeds)}"
        )
        typer.echo(
            "  prerequisites: "
            f"{'; '.join(prerequisite_states) or 'none'}; resume boundary: {resume_boundary}"
        )


def _plan_prerequisite_states(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    prerequisites: tuple[CataloguePrerequisiteValue, ...],
) -> tuple[str, ...]:
    states: list[str] = []
    for prerequisite in prerequisites:
        if isinstance(prerequisite, ExperimentName):
            state = _experiment_state(store, layout.execution_root, prerequisite)
            states.append(f"{prerequisite.value}={state.state.value}")
        else:
            states.append(str(prerequisite.value))
    return tuple(states)


def _planned_condition_count(definition: ExperimentDefinition) -> Index:
    if definition.name is ExperimentName.WEAK_SIGNAL_SUPPORT_AND_HETEROGENEITY_BOUNDARIES:
        weak_boundaries = (
            active_config().experiments.weak_signal_support_and_heterogeneity_boundaries
        )
        return weak_boundaries.distinct_condition_count()
    return len(definition.conditions.entries)


def _plan_resume_boundary(
    experiment: ExperimentName,
    prerequisite_states: tuple[str, ...],
) -> str:
    for state in prerequisite_states:
        if not state.endswith(f"={ArtifactState.COMPLETED.value}"):
            return state.split("=", maxsplit=1)[0]
    return experiment.value


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
        blocked_reasons = OrderedDict(result.resource_blocked_datasets)
        invalid_reasons = OrderedDict(result.invalid_datasets)
        for observation in result.observations:
            event_time = observation.event_time
            terminal_reason = invalid_reasons.get(observation.dataset) or blocked_reasons.get(
                observation.dataset
            )
            if terminal_reason is not None:
                typer.echo(
                    f"{observation.dataset.value}: {PreprocessingReadiness.BLOCKED} | "
                    f"reason={terminal_reason}"
                )
                continue
            state = (
                PreprocessingReadiness.READY
                if observation.valid_for_chronological_preprocessing
                else PreprocessingReadiness.BLOCKED
            )
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
    return store.current_completed_manifest(experiment.value)


def _blocked_experiment(layout_root: Path, experiment: ExperimentName) -> bool:
    return (
        layout_root
        / StorageLayoutSegment.EXPERIMENTS
        / safe_slug(experiment.value)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
        / StorageLayoutSegment.BLOCKED_JSON
    ).is_file()


def _experiment_state(
    store: ArtifactStore,
    layout_root: Path,
    experiment: ExperimentName,
) -> ExperimentStateReport:
    if _blocked_experiment(layout_root, experiment):
        return ExperimentStateReport(
            ArtifactState.BLOCKED, FieldDescription("blocked by an upstream prerequisite")
        )
    candidates = tuple(
        manifest
        for manifest in store.all_manifests()
        if recorded_experiment(manifest) is experiment
    )
    if not candidates:
        return ExperimentStateReport(
            ArtifactState.MISSING, FieldDescription("no recorded artifact")
        )
    for manifest in candidates:
        report = store.artifact_state(manifest.artifact_id)
        if report.state is not ArtifactState.COMPLETED:
            return ExperimentStateReport(
                report.state,
                FieldDescription(_state_reason(report)),
            )
    return ExperimentStateReport(
        ArtifactState.COMPLETED, FieldDescription("verified completed artifact")
    )


def _state_reason(report: ArtifactStateReport) -> str:
    parts: list[str] = [str(report.reason)]
    if report.first_changed_dependency is not None:
        parts.append(f"first changed dependency: {report.first_changed_dependency}")
    if report.nearest_reusable_ancestor is not None:
        parts.append(f"nearest reusable ancestor: {report.nearest_reusable_ancestor}")
    return "; ".join(part for part in parts if part)


def report(
    experiment_name: ExperimentName | None = OPTIONAL_ARGUMENT,
    overwrite: bool = False,
) -> None:
    try:
        catalogue = build_catalogue()
        execution_logger().event(
            ExecutionEventName.REPORT_START,
            experiment=None if experiment_name is None else experiment_name.value,
        )
        selected = (
            (experiment_name,) if experiment_name is not None else catalogue.registered_names()
        )
        layout = build_layout()
        store = ArtifactStore(layout.execution_root)
        writer = VerifiedEvidenceWriter(store, layout, overwrite=overwrite)
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
                    ReportArtifactName.NUMERICAL_CONSTANTS_AND_SEEDS,
                ),
                (
                    experiment_matrix_table(experiment_matrix_rows(catalogue)),
                    ReportArtifactName.EXPERIMENT_MATRIX,
                ),
                (
                    dataset_and_client_protocol_table(
                        base_model_pilot_dataset_manifests(layout),
                        modality_by_dataset=dataset_modality_by_dataset(),
                        role_by_dataset=dataset_client_roles(),
                        excluded_class_counts=excluded_class_counts(
                            base_model_pilot_dataset_manifests(layout)
                        ),
                    ),
                    ReportArtifactName.DATASET_AND_CLIENT_PROTOCOL,
                ),
                (
                    information_resource_matrix_table(),
                    ReportArtifactName.INFORMATION_RESOURCE_MATRIX,
                ),
            ):
                typer.echo(str(writer.write_project_evidence_table(table, name)))
            transfer_ontology_rows = transfer_ontology_and_null_padding_rows(store)
            if transfer_ontology_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            transfer_ontology_and_null_padding_table(transfer_ontology_rows),
                            ReportArtifactName.TRANSFER_ONTOLOGY_AND_NULL_PADDING,
                        )
                    )
                )
            model_training_rows = training_protocol_rows(store)
            if model_training_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            model_and_training_protocol_table(model_training_rows),
                            ReportArtifactName.MODEL_AND_TRAINING_PROTOCOL,
                        )
                    )
                )
            evidence_rows = completed_evidence_status_rows(store)
            if evidence_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            evidence_status_table(evidence_status_rows(evidence_rows)),
                            ReportArtifactName.EVIDENCE_STATUS,
                        )
                    )
                )
            primary_transfer_metrics = completed_primary_transfer_metric_records(store)
            primary_transfer_comparisons = completed_primary_transfer_comparison_records(store)
            primary_transfer_artifact_ids = tuple(
                identifier
                for record in primary_transfer_metrics
                for identifier in record.input_artifact_ids
            ) + tuple(
                identifier
                for record in primary_transfer_comparisons
                for identifier in record.input_metric_artifact_ids
            )
            if primary_transfer_metrics or primary_transfer_comparisons:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            primary_strict_transfer_results_table(
                                primary_transfer_metrics, primary_transfer_comparisons
                            ),
                            ReportArtifactName.PRIMARY_STRICT_TRANSFER_RESULTS,
                            primary_transfer_artifact_ids,
                        )
                    )
                )
            gain_figure = real_transfer_gain_series(primary_transfer_comparisons)
            if gain_figure.series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            real_transfer_gain_forest_plot(
                                gain_figure.series, gain_figure.pair_labels
                            ),
                            ReportArtifactName.REAL_TRANSFER_GAIN_FOREST_PLOT,
                        )
                    )
                )
            baseline_difference_series = baseline_paired_difference_series(primary_transfer_metrics)
            if baseline_difference_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            baseline_paired_difference_plot(baseline_difference_series),
                            ReportArtifactName.BASELINE_PAIRED_DIFFERENCE_PLOT,
                        )
                    )
                )
            solver_benchmark_rows = exact_solver_results_rows(
                completed_experiment_metric_records_with_support(
                    store, ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK
                )
            )
            if solver_benchmark_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            exact_solver_results_table(solver_benchmark_rows),
                            ReportArtifactName.EXACT_SOLVER_RESULTS,
                        )
                    )
                )
            scalability_rows = scalability_results_rows(
                completed_experiment_metric_records_with_support(
                    store, ExperimentName.SCALABILITY_AND_EFFICIENCY
                )
            )
            if scalability_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            scalability_results_table(scalability_rows),
                            ReportArtifactName.SCALABILITY_RESULTS,
                        )
                    )
                )
            ablation_rows = ablation_results_rows(
                completed_experiment_metric_records(store, ExperimentName.MECHANISM_ABLATIONS)
            )
            if ablation_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            ablation_results_table(ablation_rows),
                            ReportArtifactName.ABLATION_RESULTS,
                        )
                    )
                )
            sparsity_and_dense_rows = sparsity_and_dense_results_rows(
                completed_experiment_metric_records(
                    store, ExperimentName.SPARSITY_AND_DENSE_FALLBACK
                ),
                primary_transfer_metrics,
            )
            if sparsity_and_dense_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            sparsity_and_dense_results_table(sparsity_and_dense_rows),
                            ReportArtifactName.SPARSITY_AND_DENSE_RESULTS,
                        )
                    )
                )
            generalization_rows = generalization_results_rows(
                completed_experiment_metric_records(
                    store, ExperimentName.SECONDARY_CROSS_MODALITY_GENERALIZATION
                )
            )
            if generalization_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            generalization_results_table(generalization_rows),
                            ReportArtifactName.GENERALIZATION_RESULTS,
                        )
                    )
                )
            confirmation_rows = confirmation_results_rows(
                completed_experiment_metric_records(
                    store, ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY
                ),
                primary_transfer_comparisons,
            )
            if confirmation_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            confirmation_results_table(
                                confirmation_rows,
                                registered_pairs=tuple(
                                    spec.direction
                                    for spec in (
                                        active_config().scientific.datasets.primary_directed_pairs
                                    )
                                ),
                            ),
                            ReportArtifactName.CONFIRMATION_RESULTS,
                        )
                    )
                )
            coupling_rows = coupling_mechanism_results_rows(
                completed_experiment_metric_records_with_support(
                    store, ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION
                ),
                completed_experiment_metric_records(
                    store, ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION
                ),
                primary_transfer_comparisons,
            )
            if coupling_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            coupling_mechanism_results_table(coupling_rows),
                            ReportArtifactName.COUPLING_MECHANISM_RESULTS,
                        )
                    )
                )
            failure_boundary_rows = failure_boundary_results_rows(
                completed_experiment_metric_records(
                    store, ExperimentName.WEAK_SIGNAL_SUPPORT_AND_HETEROGENEITY_BOUNDARIES
                ),
                completed_experiment_metric_records(
                    store, ExperimentName.SEMANTIC_SUFFICIENCY_FRONTIER
                ),
                primary_transfer_metrics,
            )
            if failure_boundary_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            failure_boundary_results_table(failure_boundary_rows),
                            ReportArtifactName.FAILURE_BOUNDARY_RESULTS,
                        )
                    )
                )
                boundary_series = failure_boundary_figure_series(failure_boundary_rows)
                if boundary_series:
                    typer.echo(
                        str(
                            writer.write_project_evidence_figure(
                                failure_boundary_figure(boundary_series),
                                ReportArtifactName.FAILURE_BOUNDARY_FIGURE,
                            )
                        )
                    )
            if coupling_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            coupling_gap_phase_figure(factor_rows=coupling_rows),
                            ReportArtifactName.COUPLING_GAP_PHASE_FIGURE,
                        )
                    )
                )
            if primary_transfer_metrics:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            predicted_vs_realized_transfer_figure(
                                metric_records=primary_transfer_metrics
                            ),
                            ReportArtifactName.PREDICTED_VS_REALIZED_TRANSFER_FIGURE,
                            tuple(
                                identifier
                                for record in primary_transfer_metrics
                                for identifier in record.input_artifact_ids
                            ),
                        )
                    )
                )
            sparsity_series = sparsity_figure_series(sparsity_and_dense_rows)
            if sparsity_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            sparsity_utility_efficiency_figure(sparsity_series),
                            ReportArtifactName.SPARSITY_UTILITY_EFFICIENCY_FIGURE,
                        )
                    )
                )
            confirmation_series = confirmation_figure_series(confirmation_rows)
            if confirmation_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            confirmation_safety_coverage_figure(confirmation_series),
                            ReportArtifactName.CONFIRMATION_SAFETY_COVERAGE_FIGURE,
                        )
                    )
                )
            frontier_series = semantic_sufficiency_series(
                completed_experiment_metric_records(
                    store, ExperimentName.SEMANTIC_SUFFICIENCY_FRONTIER
                )
            )
            if frontier_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            semantic_sufficiency_frontier_figure(frontier_series),
                            ReportArtifactName.SEMANTIC_SUFFICIENCY_FRONTIER_FIGURE,
                        )
                    )
                )
            scalability_series = scalability_figure_series(scalability_rows)
            if scalability_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            scalability_figure(scalability_series),
                            ReportArtifactName.SCALABILITY_FIGURE,
                        )
                    )
                )
            map_bound_series = map_value_bound_series(
                completed_experiment_metric_records(
                    store, ExperimentName.EXACT_MAP_VALUE_BOUND_VALIDATION
                )
            )
            if map_bound_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            map_value_bound_figure(map_bound_series),
                            ReportArtifactName.MAP_VALUE_BOUND_FIGURE,
                        )
                    )
                )
        if exported == 0:
            typer.echo("no verified persisted evidence available for report generation")
        execution_logger().event(ExecutionEventName.REPORT_END, exported=exported)
    except (CliUsageError, ExecutionError, ValueError) as error:
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
            f"{'#':>2} {'Experiment':<50} {'Role':<22} {'Status':<10} {'Est-run':<8} "
            f"{'Est-end':<8} Reason"
        )
        index = 0
        for name in names:
            if name not in selected:
                continue
            definition = catalogue.definition(name)
            report = _experiment_state(store, layout.execution_root, name)
            typer.echo(
                f"{index:>2} {name.value:<50} {definition.classification.value:<22} "
                f"{report.state.value:<10} {'-':<8} {'-':<8} {report.reason}"
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
