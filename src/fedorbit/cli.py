from __future__ import annotations

import statistics
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn, cast

import structlog
import typer
from typer import Argument, Exit

from fedorbit.analysis.records import MetricRecord, PairedComparisonRecord
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
    completed_experiment_metric_records,
    completed_experiment_metric_records_with_support,
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
    ablation_results_table,
    confirmation_results_table,
    coupling_mechanism_results_table,
    dataset_and_client_protocol_table,
    exact_solver_results_table,
    experiment_matrix_table,
    generalization_results_table,
    numerical_constants_and_seeds_table,
    primary_strict_transfer_results_table,
    real_transfer_gain_forest_plot,
    scalability_results_table,
    sparsity_and_dense_results_table,
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
    ExperimentLocalMethod,
    FailureReason,
    MetricId,
    MultiplicityFamily,
    OverwritePolicy,
    RelativeGain,
    ReportArtifactName,
    ReportSeriesName,
    RiskReductionColumn,
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


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def _percentile_95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def _metric_values(
    records: Sequence[MetricRecord], method: TransferMethod, metric_name: MetricId
) -> list[float]:
    return [
        record.metric_value
        for record in records
        if record.method == method
        and record.metric_name == metric_name
        and record.valid
        and record.metric_value is not None
    ]


def _parse_k_pattern_condition(condition: str) -> tuple[int, str] | None:
    if not condition.startswith("k"):
        return None
    k_text, separator, pattern = condition[1:].partition("-")
    if not separator or not k_text.isdigit() or not pattern:
        return None
    return int(k_text), pattern


def _exact_solver_results_rows(
    records_with_support: Sequence[tuple[MetricRecord, int | None]],
) -> tuple[Mapping[str, TableScalar], ...]:
    groups: OrderedDict[tuple[int, str, int | None], list[MetricRecord]] = OrderedDict()
    for record, support in records_with_support:
        parsed = _parse_k_pattern_condition(record.condition)
        if parsed is None:
            continue
        groups.setdefault((*parsed, support), []).append(record)
    rows: list[Mapping[str, TableScalar]] = []
    for (k, pattern, support), entries in groups.items():
        errors = _metric_values(
            entries, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, MetricId.ABSOLUTE_OBJECTIVE_ERROR
        )
        validity = _metric_values(
            entries,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            MetricId.CORRESPONDENCE_CERTIFICATE_VALIDITY,
        )
        runtimes = _metric_values(
            entries, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, MetricId.WALL_TIME
        )
        memory = _metric_values(
            entries, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, MetricId.PEAK_HOST_RSS
        )
        active_images = _metric_values(
            entries, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, MetricId.ACTIVE_IMAGE_CANDIDATES
        )
        lap_calls = _metric_values(
            entries, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, MetricId.LAP_CALLS
        )
        qap_runtimes = _metric_values(entries, TransferMethod.GENERIC_EXACT_QAP, MetricId.WALL_TIME)
        qap_timeouts = _metric_values(
            entries, TransferMethod.GENERIC_EXACT_QAP, MetricId.TIMEOUT_INDICATOR
        )
        dense_runtimes = _metric_values(
            entries, TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK, MetricId.WALL_TIME
        )
        rows.append(
            OrderedDict(
                k=k,
                block_pattern=pattern,
                support=support,
                truth_availability=bool(errors),
                exact_mismatches=sum(1 for value in validity if value == 0.0),
                maximum_absolute_error=max(errors) if errors else None,
                runtime_median=_median(runtimes),
                runtime_p95=_percentile_95(runtimes),
                qap_runtime=_median(qap_runtimes),
                dense_runtime=_median(dense_runtimes),
                timeouts=sum(1 for value in qap_timeouts if value == 1.0),
                memory=_median(memory),
                active_images=_median(active_images),
                lap_calls=_median(lap_calls),
            )
        )
    return tuple(rows)


def _scalability_results_rows(
    records_with_support: Sequence[tuple[MetricRecord, int | None]],
) -> tuple[Mapping[str, TableScalar], ...]:
    groups: OrderedDict[tuple[int, str, int | None, TransferMethod], list[MetricRecord]] = (
        OrderedDict()
    )
    for record, support in records_with_support:
        parsed = _parse_k_pattern_condition(record.condition)
        if parsed is None:
            continue
        groups.setdefault((*parsed, support, record.method), []).append(record)
    rows: list[Mapping[str, TableScalar]] = []
    for (k, pattern, support, method), entries in groups.items():
        runtimes = _metric_values(entries, method, MetricId.WALL_TIME)
        rss = _metric_values(entries, method, MetricId.PEAK_HOST_RSS)
        cuda_memory = _metric_values(entries, method, MetricId.PEAK_CUDA_ALLOCATED_BYTES)
        active_images = _metric_values(entries, method, MetricId.ACTIVE_IMAGE_CANDIDATES)
        lap_calls = _metric_values(entries, method, MetricId.LAP_CALLS)
        timeouts = _metric_values(entries, method, MetricId.TIMEOUT_INDICATOR)
        rows.append(
            OrderedDict(
                k=k,
                block=pattern,
                support=support,
                method=method.value,
                n_s=_median(active_images),
                lap_calls=_median(lap_calls),
                cuts=None,
                runtime_median=_median(runtimes),
                runtime_p95=_percentile_95(runtimes),
                rss=_median(rss),
                cuda_memory=_median(cuda_memory),
                timeout=sum(1 for value in timeouts if value == 1.0),
                exactness_status=None,
            )
        )
    return tuple(rows)


def _ablation_results_rows(
    records: Sequence[MetricRecord],
) -> tuple[Mapping[str, TableScalar], ...]:
    pairs = sorted({record.pair for record in records})
    ablation_methods = sorted(
        {record.method for record in records if record.method != TransferMethod.LOCAL_ONLY},
        key=lambda method: method.value,
    )
    rows: list[Mapping[str, TableScalar]] = []
    for pair in pairs:
        full_ce = _metric_value_by_condition(
            records, pair, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, "principal"
        )
        for method in ablation_methods:
            ablation_ce = _metric_value_by_condition(records, pair, method, "principal")
            if ablation_ce is None:
                continue
            difference_vs_full = ablation_ce - full_ce if full_ce is not None else None
            rows.append(
                OrderedDict(
                    ablation=method.value,
                    pair=pair,
                    realized_gain=None,
                    difference_vs_full=difference_vs_full,
                    equivalence=None,
                    retained_gain=None,
                    confirmation_safety=None,
                )
            )
    return tuple(rows)


def _metric_value_by_condition(
    records: Sequence[MetricRecord],
    pair: str,
    method: TransferMethod,
    condition: str,
    metric_name: MetricId = MetricId.MACRO_CROSS_ENTROPY,
) -> float | None:
    matches = [
        record
        for record in records
        if record.pair == pair
        and record.method == method
        and record.condition == condition
        and record.metric_name == metric_name
        and record.valid
        and record.metric_value is not None
    ]
    if not matches:
        return None
    return statistics.fmean(
        record.metric_value for record in matches if record.metric_value is not None
    )


def _sparsity_and_dense_results_rows(
    sparsity_records: Sequence[MetricRecord],
    local_only_records: Sequence[MetricRecord],
) -> tuple[Mapping[str, TableScalar], ...]:
    pairs = sorted({record.pair for record in sparsity_records})
    conditions = sorted({record.condition for record in sparsity_records})
    rows: list[Mapping[str, TableScalar]] = []
    for pair in pairs:
        local_only_ce = _metric_value_by_condition(
            local_only_records, pair, TransferMethod.LOCAL_ONLY, "principal"
        )
        for condition in conditions:
            method = next(
                (
                    record.method
                    for record in sparsity_records
                    if record.pair == pair and record.condition == condition
                ),
                None,
            )
            if method is None:
                continue
            condition_ce = _metric_value_by_condition(sparsity_records, pair, method, condition)
            realized_gain = (
                (local_only_ce - condition_ce) / local_only_ce
                if local_only_ce is not None and condition_ce is not None and local_only_ce != 0.0
                else None
            )
            rows.append(
                OrderedDict(
                    support_or_dense_condition=condition,
                    pair=pair,
                    realized_gain=realized_gain,
                    certified_value=None,
                    runtime=None,
                    memory=None,
                    confirmation_coverage=None,
                    dense_minus_sparse_difference=None,
                )
            )
    return tuple(rows)


def _generalization_results_rows(
    metric_records: Sequence[MetricRecord],
) -> tuple[Mapping[str, TableScalar], ...]:
    pairs = sorted({record.pair for record in metric_records})
    method_order = (
        TransferMethod.LOCAL_ONLY,
        TransferMethod.LOCAL_SIR,
        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
    )
    rows: list[Mapping[str, TableScalar]] = []
    for pair in pairs:
        for method in method_order:
            valid_seeds = len(
                {
                    record.seed
                    for record in metric_records
                    if record.pair == pair
                    and record.method == method
                    and record.metric_name == MetricId.MACRO_CROSS_ENTROPY
                    and record.valid
                }
            )
            if valid_seeds == 0:
                continue
            rows.append(
                OrderedDict(
                    pair=pair,
                    method=method.value,
                    valid_seeds=valid_seeds,
                    test_macro_ce=_metric_value_by_condition(
                        metric_records, pair, method, "principal"
                    ),
                    macro_f1=_metric_value_by_condition(
                        metric_records, pair, method, "principal", metric_name=MetricId.MACRO_F1
                    ),
                    balanced_accuracy=_metric_value_by_condition(
                        metric_records,
                        pair,
                        method,
                        "principal",
                        metric_name=MetricId.BALANCED_ACCURACY,
                    ),
                    gain_vs_local=None,
                    bca_ci_low=None,
                    bca_ci_high=None,
                    raw_p=None,
                    holm_p=None,
                    strict_validity=None,
                    confirmation_coverage=None,
                    is_secondary_pair=True,
                )
            )
    return tuple(rows)


def _confirmation_results_rows(
    records: Sequence[MetricRecord],
) -> tuple[Mapping[str, TableScalar], ...]:
    pairs = sorted({record.pair for record in records})
    rows: list[Mapping[str, TableScalar]] = []
    for pair in pairs:
        verdicts = [
            record.metric_value
            for record in records
            if record.pair == pair
            and record.method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
            and record.metric_name == MetricId.PROPOSAL_ACCEPTANCE_RATE
            and record.valid
            and record.metric_value is not None
        ]
        if not verdicts:
            continue
        proposals = len(verdicts)
        accepted = sum(1 for value in verdicts if value == 1.0)
        rows.append(
            OrderedDict(
                pair=pair,
                proposals=proposals,
                accepted=accepted,
                harmful_accepted_rate=None,
                useful_accepted_rate=None,
                beneficial_rejected_rate=None,
                coverage=accepted / proposals,
                no_confirm_harmful_rate=None,
                **{
                    RiskReductionColumn.ABSOLUTE_RISK_REDUCTION.value: None,
                    RiskReductionColumn.RELATIVE_RISK_REDUCTION.value: None,
                },
                ci=None,
                p=None,
            )
        )
    return tuple(rows)


def _coupling_gap_row(
    condition_or_pair: str,
    gap_values: Sequence[float],
    fixed_action_values: Sequence[float],
    ci: str | None,
    holm_p: float | None,
) -> Mapping[str, TableScalar]:
    materiality = active_config().scientific.materiality.coupling_objective_units
    above_materiality = sum(1 for value in gap_values if value > materiality)
    return OrderedDict(
        condition_or_pair=condition_or_pair,
        valid_units=len(gap_values),
        fixed_action_gap=statistics.fmean(fixed_action_values) if fixed_action_values else None,
        robust_coupling_gap=statistics.fmean(gap_values) if gap_values else None,
        fraction_above_materiality=above_materiality / len(gap_values) if gap_values else None,
        ci=ci,
        holm_p=holm_p,
        coupling_destruction_retained_gain_fraction=None,
    )


def _coupling_mechanism_results_rows(
    synthetic_records: Sequence[MetricRecord],
    real_packet_records: Sequence[MetricRecord],
    comparison_records: Sequence[PairedComparisonRecord],
) -> tuple[Mapping[str, TableScalar], ...]:
    rows: list[Mapping[str, TableScalar]] = []
    conditions = sorted({record.condition for record in synthetic_records})
    for condition in conditions:
        gap_values = [
            float(record.metric_value)
            for record in synthetic_records
            if record.condition == condition
            and record.method == TransferMethod.MATCHED_RESOURCE_RECTANGULAR
            and record.metric_name == MetricId.ROBUST_COUPLING_VALUE_GAP
            and record.valid
            and record.metric_value is not None
        ]
        fixed_action_values = [
            float(record.metric_value)
            for record in synthetic_records
            if record.condition == condition
            and record.method == TransferMethod.MATCHED_RESOURCE_RECTANGULAR
            and record.metric_name == MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP
            and record.valid
            and record.metric_value is not None
        ]
        if not gap_values:
            continue
        rows.append(_coupling_gap_row(condition, gap_values, fixed_action_values, None, None))
    pairs = sorted({record.pair for record in real_packet_records})
    comparisons_by_pair = {
        comparison.pair: comparison
        for comparison in comparison_records
        if comparison.family == MultiplicityFamily.COUPLING_MECHANISM
        and comparison.method_a == ExperimentLocalMethod.EXACT_ORBIT
    }
    for pair in pairs:
        gap_values = [
            float(record.metric_value)
            for record in real_packet_records
            if record.pair == pair
            and record.method == TransferMethod.MATCHED_RESOURCE_RECTANGULAR
            and record.metric_name == MetricId.ROBUST_COUPLING_VALUE_GAP
            and record.valid
            and record.metric_value is not None
        ]
        fixed_action_values = [
            float(record.metric_value)
            for record in real_packet_records
            if record.pair == pair
            and record.method == TransferMethod.MATCHED_RESOURCE_RECTANGULAR
            and record.metric_name == MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP
            and record.valid
            and record.metric_value is not None
        ]
        if not gap_values:
            continue
        comparison = comparisons_by_pair.get(DirectedPairName(pair))
        ci = (
            f"[{comparison.bca_ci_low:.4g}, {comparison.bca_ci_high:.4g}]"
            if comparison is not None
            and comparison.bca_ci_low is not None
            and comparison.bca_ci_high is not None
            else None
        )
        holm_p = comparison.holm_p if comparison is not None else None
        rows.append(_coupling_gap_row(pair, gap_values, fixed_action_values, ci, holm_p))
    return tuple(rows)


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
            solver_benchmark_rows = _exact_solver_results_rows(
                completed_experiment_metric_records_with_support(
                    store, ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK
                )
            )
            if solver_benchmark_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            exact_solver_results_table(solver_benchmark_rows),
                            ReportArtifactName("exact-solver-results"),
                        )
                    )
                )
            scalability_rows = _scalability_results_rows(
                completed_experiment_metric_records_with_support(
                    store, ExperimentName.SCALABILITY_AND_EFFICIENCY
                )
            )
            if scalability_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            scalability_results_table(scalability_rows),
                            ReportArtifactName("scalability-results"),
                        )
                    )
                )
            ablation_rows = _ablation_results_rows(
                completed_experiment_metric_records(store, ExperimentName.MECHANISM_ABLATIONS)
            )
            if ablation_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            ablation_results_table(ablation_rows),
                            ReportArtifactName("ablation-results"),
                        )
                    )
                )
            sparsity_and_dense_rows = _sparsity_and_dense_results_rows(
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
                            ReportArtifactName("sparsity-and-dense-results"),
                        )
                    )
                )
            generalization_rows = _generalization_results_rows(
                completed_experiment_metric_records(
                    store, ExperimentName.SECONDARY_CROSS_MODALITY_GENERALIZATION
                )
            )
            if generalization_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            generalization_results_table(generalization_rows),
                            ReportArtifactName("generalization-results"),
                        )
                    )
                )
            confirmation_rows = _confirmation_results_rows(
                completed_experiment_metric_records(
                    store, ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY
                )
            )
            if confirmation_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            confirmation_results_table(confirmation_rows),
                            ReportArtifactName("confirmation-results"),
                        )
                    )
                )
            coupling_rows = _coupling_mechanism_results_rows(
                completed_experiment_metric_records(
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
                            ReportArtifactName("coupling-mechanism-results"),
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
