from __future__ import annotations

import math
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
from fedorbit.experiments.classification import EvidenceStatusRow, completed_evidence_status_rows
from fedorbit.experiments.dispatch import (
    ExperimentExecutionRequest,
    run_experiment,
    run_smoke_validation,
)
from fedorbit.experiments.synthesis import (
    completed_experiment_metric_records,
    completed_experiment_metric_records_with_support,
    completed_primary_transfer_comparison_records,
    completed_primary_transfer_metric_records,
    transfer_ontology_and_null_padding_rows,
)
from fedorbit.experiments.training import training_protocol_rows
from fedorbit.infrastructure.artifacts import ArtifactStore, ExecutionError, execution_store
from fedorbit.infrastructure.environment import (
    environment_snapshot,
    reference_gpu_matches,
)
from fedorbit.infrastructure.failures import validation_failure_outcome
from fedorbit.infrastructure.manifests import DatasetManifest, ReusableArtifactManifest
from fedorbit.infrastructure.preparation import DatasetPreparationRequest, preprocess_datasets
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
    PRINCIPAL_EVALUATION_CONDITION,
    ArtifactState,
    CliCommand,
    ClientRole,
    DatasetId,
    DatasetIdentifierText,
    DatasetModality,
    DirectedPairName,
    EvaluationConditionName,
    ExitStatus,
    ExperimentIdentifierText,
    ExperimentLocalMethod,
    ExperimentName,
    FailureReason,
    Index,
    MetricId,
    MultiplicityFamily,
    OverwritePolicy,
    PreprocessingReadiness,
    RandomSeed,
    ReportArtifactName,
    ReportColumnName,
    ReportSeriesName,
    StableJsonPayload,
    StorageLayoutSegment,
    TransferMethod,
    WeakSignalBoundaryDimension,
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
        selected = (
            (dataset_identifier(dataset_name.value),)
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
            state = (
                PreprocessingReadiness.READY
                if observation.valid_for_chronological_preprocessing
                else PreprocessingReadiness.BLOCKED
            )
            typer.echo(
                f"{observation.dataset.value}: {state} | chronology={event_time.state.value} | "
                f"reason={event_time.reason}"
            )
        for dataset, reason in result.resource_blocked_datasets:
            typer.echo(f"{dataset.value}: {PreprocessingReadiness.BLOCKED} | reason={reason}")
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
    times = tuple(
        Path(payload).stat().st_mtime_ns
        for payload in manifest.payload_paths
        if Path(payload).is_file()
    )
    return max(times) if times else 0


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
    return resolved.state


def _base_model_pilot_dataset_manifests(layout: WorkspaceLayout) -> tuple[DatasetManifest, ...]:
    workspace = experiment_workspace(layout, ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT)
    manifests: list[DatasetManifest] = []
    for dataset in DatasetId:
        path = (
            workspace
            / StorageLayoutSegment.ARTIFACTS
            / StorageLayoutSegment.DERIVED
            / f"dataset-manifest.{dataset.value}.json"
        )
        if path.is_file():
            manifests.append(DatasetManifest.model_validate_json(path.read_text(encoding="utf-8")))
    return tuple(manifests)


def _dataset_client_roles() -> Mapping[DatasetId, ClientRole]:
    return OrderedDict(
        (dataset, client.role)
        for dataset, client in active_config().scientific.datasets.clients.items()
    )


def _dataset_modality_by_dataset() -> Mapping[DatasetId, DatasetModality]:
    modalities: OrderedDict[DatasetId, DatasetModality] = OrderedDict()
    for dataset in DatasetId:
        modalities[dataset] = (
            DatasetModality.NETWORK
            if dataset in {DatasetId.EDGE_IIOTSET_NETWORK, DatasetId.TON_IOT_NETWORK}
            else DatasetModality.HOST
        )
    return modalities


def _excluded_class_counts(
    manifests: Sequence[DatasetManifest],
) -> Mapping[DatasetId, Index]:
    return OrderedDict(
        (manifest.dataset, max(0, manifest.feature_quality.dropped_feature_count))
        for manifest in manifests
    )


def _experiment_matrix_rows(
    catalogue: ExperimentCatalogue,
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for name in catalogue.registered_names():
        definition = catalogue.definition(name)
        rows.append(
            OrderedDict(
                (
                    (ReportColumnName.EXPERIMENT, name.value),
                    (ReportColumnName.CLASSIFICATION, definition.classification.value),
                    (
                        ReportColumnName.DATASETS_OR_PAIRS,
                        ", ".join(scope.value for scope in definition.datasets_or_pairs),
                    ),
                    (
                        ReportColumnName.METHODS,
                        ", ".join(str(method) for method in definition.methods),
                    ),
                    (
                        ReportColumnName.REGISTERED_SEEDS,
                        ", ".join(str(seed) for seed in definition.seeds),
                    ),
                    (ReportColumnName.CONDITIONS, str(definition.conditions)),
                    (ReportColumnName.DERIVED_PLANNED_CELLS, int(definition.derived_planned_cells)),
                    (
                        ReportColumnName.PREREQUISITES,
                        ", ".join(str(prerequisite) for prerequisite in definition.prerequisites),
                    ),
                    (
                        ReportColumnName.EVIDENCE_RELATIONSHIP,
                        (
                            ", ".join(consumer.value for consumer in definition.evidence_consumers)
                            if definition.evidence_consumers
                            else "report export"
                        ),
                    ),
                )
            )
        )
    return tuple(rows)


def _real_transfer_gain_series(
    comparisons: Sequence[PairedComparisonRecord],
) -> tuple[tuple[FigureSeries, ...], tuple[ReportSeriesName, ...]]:
    selected = [
        record
        for record in comparisons
        if record.method_a == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
        and record.method_b == TransferMethod.LOCAL_ONLY
        and record.mean_difference is not None
    ]
    if not selected:
        return (), ()
    selected.sort(key=lambda record: record.pair)
    mids: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    pair_labels: list[ReportSeriesName] = []
    for record in selected:
        mid = record.mean_difference
        if mid is None:
            continue
        mids.append(float(mid))
        lows.append(float(record.bca_ci_low) if record.bca_ci_low is not None else float(mid))
        highs.append(float(record.bca_ci_high) if record.bca_ci_high is not None else float(mid))
        pair_labels.append(ReportSeriesName(record.pair))
    if not mids:
        return (), ()
    series = (
        FigureSeries(
            name=ReportSeriesName(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value),
            x=tuple(mids),
            y=tuple(float(index) for index in range(len(mids))),
            x_low=tuple(lows),
            x_high=tuple(highs),
        ),
    )
    return series, tuple(pair_labels)


def _baseline_paired_difference_series(
    metric_records: Sequence[MetricRecord],
) -> tuple[FigureSeries, ...]:
    pairs = sorted({record.pair for record in metric_records})
    local_only_by_pair_seed: Mapping[tuple[DirectedPairName, RandomSeed], float] = OrderedDict(
        ((record.pair, record.seed), record.metric_value)
        for record in metric_records
        if record.method == TransferMethod.LOCAL_ONLY
        and record.condition == PRINCIPAL_EVALUATION_CONDITION.name
        and record.metric_name == MetricId.MACRO_CROSS_ENTROPY
        and record.valid
        and record.metric_value is not None
    )
    baseline_methods = (
        TransferMethod.LOCAL_SIR,
        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
    )
    series: list[FigureSeries] = []
    for pair in pairs:
        for method in baseline_methods:
            x_values: list[float] = []
            y_values: list[float] = []
            for record in metric_records:
                if (
                    record.pair != pair
                    or record.method != method
                    or record.condition != PRINCIPAL_EVALUATION_CONDITION.name
                    or record.metric_name != MetricId.MACRO_CROSS_ENTROPY
                    or not record.valid
                    or record.metric_value is None
                ):
                    continue
                local_only_value = local_only_by_pair_seed.get((record.pair, record.seed))
                if local_only_value is None:
                    continue
                x_values.append(float(record.seed))
                y_values.append(record.metric_value - local_only_value)
            if x_values:
                series.append(
                    FigureSeries(
                        name=ReportSeriesName(method.value),
                        x=tuple(x_values),
                        y=tuple(y_values),
                        panel=ReportSeriesName(pair),
                    )
                )
    return tuple(series)


def _median(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def _percentile_95(
    values: Sequence[float],
) -> float | None:
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


def _parse_k_pattern_condition(
    condition: str,
) -> tuple[int, str] | None:
    if not condition.startswith("k"):
        return None
    k_text, separator, pattern = condition[1:].partition("-")
    if not separator or not k_text.isdigit() or not pattern:
        return None
    return int(k_text), pattern


def _exact_solver_results_rows(
    records_with_support: Sequence[tuple[MetricRecord, int | None]],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    groups: OrderedDict[tuple[int, str, int | None], list[MetricRecord]] = OrderedDict()
    for record, support in records_with_support:
        parsed = _parse_k_pattern_condition(record.condition)
        if parsed is None:
            continue
        groups.setdefault((*parsed, support), []).append(record)
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
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
                (
                    (
                        ReportColumnName.K,
                        k,
                    ),
                    (
                        ReportColumnName.BLOCK_PATTERN,
                        pattern,
                    ),
                    (
                        ReportColumnName.SUPPORT,
                        support,
                    ),
                    (
                        ReportColumnName.TRUTH_AVAILABILITY,
                        bool(errors),
                    ),
                    (
                        ReportColumnName.EXACT_MISMATCHES,
                        sum(1 for value in validity if value == 0.0),
                    ),
                    (
                        ReportColumnName.MAXIMUM_ABSOLUTE_ERROR,
                        max(errors) if errors else None,
                    ),
                    (
                        ReportColumnName.RUNTIME_MEDIAN,
                        _median(runtimes),
                    ),
                    (
                        ReportColumnName.RUNTIME_P95,
                        _percentile_95(runtimes),
                    ),
                    (
                        ReportColumnName.QAP_RUNTIME,
                        _median(qap_runtimes),
                    ),
                    (
                        ReportColumnName.DENSE_RUNTIME,
                        _median(dense_runtimes),
                    ),
                    (
                        ReportColumnName.TIMEOUTS,
                        sum(1 for value in qap_timeouts if value == 1.0),
                    ),
                    (
                        ReportColumnName.MEMORY,
                        _median(memory),
                    ),
                    (
                        ReportColumnName.ACTIVE_IMAGES,
                        _median(active_images),
                    ),
                    (
                        ReportColumnName.LAP_CALLS,
                        _median(lap_calls),
                    ),
                )
            )
        )
    return tuple(rows)


def _scalability_results_rows(
    records_with_support: Sequence[tuple[MetricRecord, int | None]],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    groups: OrderedDict[tuple[int, str, int | None, TransferMethod], list[MetricRecord]] = (
        OrderedDict()
    )
    for record, support in records_with_support:
        parsed = _parse_k_pattern_condition(record.condition)
        if parsed is None:
            continue
        groups.setdefault((*parsed, support, record.method), []).append(record)
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for (k, pattern, support, method), entries in groups.items():
        runtimes = _metric_values(entries, method, MetricId.WALL_TIME)
        rss = _metric_values(entries, method, MetricId.PEAK_HOST_RSS)
        cuda_memory = _metric_values(entries, method, MetricId.PEAK_CUDA_ALLOCATED_BYTES)
        active_images = _metric_values(entries, method, MetricId.ACTIVE_IMAGE_CANDIDATES)
        lap_calls = _metric_values(entries, method, MetricId.LAP_CALLS)
        timeouts = _metric_values(entries, method, MetricId.TIMEOUT_INDICATOR)
        predicted_work = _metric_values(entries, method, MetricId.PREDICTED_WORK_COORDINATE)
        rows.append(
            OrderedDict(
                (
                    (
                        ReportColumnName.K,
                        k,
                    ),
                    (
                        ReportColumnName.BLOCK,
                        pattern,
                    ),
                    (
                        ReportColumnName.SUPPORT,
                        support,
                    ),
                    (
                        ReportColumnName.METHOD,
                        method.value,
                    ),
                    (
                        ReportColumnName.N_S,
                        _median(active_images),
                    ),
                    (
                        ReportColumnName.LAP_CALLS,
                        _median(lap_calls),
                    ),
                    (
                        ReportColumnName.CUTS,
                        None,
                    ),
                    (
                        ReportColumnName.RUNTIME_MEDIAN,
                        _median(runtimes),
                    ),
                    (
                        ReportColumnName.RUNTIME_P95,
                        _percentile_95(runtimes),
                    ),
                    (
                        ReportColumnName.RSS,
                        _median(rss),
                    ),
                    (
                        ReportColumnName.CUDA_MEMORY,
                        _median(cuda_memory),
                    ),
                    (
                        ReportColumnName.TIMEOUT,
                        sum(1 for value in timeouts if value == 1.0),
                    ),
                    (
                        ReportColumnName.EXACTNESS_STATUS,
                        None,
                    ),
                    (
                        ReportColumnName.PREDICTED_WORK,
                        _median(predicted_work),
                    ),
                )
            )
        )
    return tuple(rows)


def _ablation_results_rows(
    records: Sequence[MetricRecord],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    pairs = sorted({record.pair for record in records})
    ablation_methods = sorted(
        {record.method for record in records if record.method != TransferMethod.LOCAL_ONLY},
        key=lambda method: method.value,
    )
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for pair in pairs:
        full_ce = _metric_value_by_condition(
            records,
            pair,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            PRINCIPAL_EVALUATION_CONDITION.name,
        )
        for method in ablation_methods:
            ablation_ce = _metric_value_by_condition(
                records, pair, method, PRINCIPAL_EVALUATION_CONDITION.name
            )
            if ablation_ce is None:
                continue
            difference_vs_full = ablation_ce - full_ce if full_ce is not None else None
            rows.append(
                OrderedDict(
                    (
                        (
                            ReportColumnName.ABLATION,
                            method.value,
                        ),
                        (
                            ReportColumnName.PAIR,
                            pair,
                        ),
                        (
                            ReportColumnName.REALIZED_GAIN,
                            None,
                        ),
                        (
                            ReportColumnName.DIFFERENCE_VS_FULL,
                            difference_vs_full,
                        ),
                        (
                            ReportColumnName.EQUIVALENCE,
                            None,
                        ),
                        (
                            ReportColumnName.RETAINED_GAIN,
                            None,
                        ),
                        (
                            ReportColumnName.CONFIRMATION_SAFETY,
                            None,
                        ),
                    )
                )
            )
    return tuple(rows)


def _metric_value_by_condition(
    records: Sequence[MetricRecord],
    pair: DirectedPairName,
    method: TransferMethod,
    condition: EvaluationConditionName,
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
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    pairs = sorted({record.pair for record in sparsity_records})
    conditions = sorted({record.condition for record in sparsity_records})
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for pair in pairs:
        local_only_ce = _metric_value_by_condition(
            local_only_records,
            pair,
            TransferMethod.LOCAL_ONLY,
            PRINCIPAL_EVALUATION_CONDITION.name,
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
            runtime = _metric_value_by_condition(
                sparsity_records,
                pair,
                method,
                condition,
                metric_name=MetricId.WALL_TIME,
            )
            memory = _metric_value_by_condition(
                sparsity_records,
                pair,
                method,
                condition,
                metric_name=MetricId.PEAK_HOST_RSS,
            )
            coverage = _metric_value_by_condition(
                sparsity_records,
                pair,
                method,
                condition,
                metric_name=MetricId.PROPOSAL_ACCEPTANCE_RATE,
            )
            certified = _metric_value_by_condition(
                sparsity_records,
                pair,
                method,
                condition,
                metric_name=MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE,
            )
            rows.append(
                OrderedDict(
                    (
                        (
                            ReportColumnName.SUPPORT_OR_DENSE_CONDITION,
                            condition,
                        ),
                        (
                            ReportColumnName.PAIR,
                            pair,
                        ),
                        (
                            ReportColumnName.REALIZED_GAIN,
                            realized_gain,
                        ),
                        (
                            ReportColumnName.CERTIFIED_VALUE,
                            certified,
                        ),
                        (
                            ReportColumnName.RUNTIME,
                            runtime,
                        ),
                        (
                            ReportColumnName.MEMORY,
                            memory,
                        ),
                        (
                            ReportColumnName.CONFIRMATION_COVERAGE,
                            coverage,
                        ),
                        (
                            ReportColumnName.DENSE_MINUS_SPARSE_DIFFERENCE,
                            None,
                        ),
                    )
                )
            )
    return tuple(rows)


def _generalization_results_rows(
    metric_records: Sequence[MetricRecord],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    pairs = sorted({record.pair for record in metric_records})
    method_order = (
        TransferMethod.LOCAL_ONLY,
        TransferMethod.LOCAL_SIR,
        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
    )
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
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
                    (
                        (ReportColumnName.PAIR, pair),
                        (ReportColumnName.METHOD, method.value),
                        (ReportColumnName.VALID_SEEDS, valid_seeds),
                        (
                            ReportColumnName.TEST_MACRO_CE,
                            _metric_value_by_condition(
                                metric_records, pair, method, PRINCIPAL_EVALUATION_CONDITION.name
                            ),
                        ),
                        (
                            ReportColumnName.MACRO_F1,
                            _metric_value_by_condition(
                                metric_records,
                                pair,
                                method,
                                PRINCIPAL_EVALUATION_CONDITION.name,
                                metric_name=MetricId.MACRO_F1,
                            ),
                        ),
                        (
                            ReportColumnName.BALANCED_ACCURACY,
                            _metric_value_by_condition(
                                metric_records,
                                pair,
                                method,
                                PRINCIPAL_EVALUATION_CONDITION.name,
                                metric_name=MetricId.BALANCED_ACCURACY,
                            ),
                        ),
                        (ReportColumnName.GAIN_VS_LOCAL, None),
                        (ReportColumnName.BCA_CI_LOW, None),
                        (ReportColumnName.BCA_CI_HIGH, None),
                        (ReportColumnName.RAW_P, None),
                        (ReportColumnName.HOLM_P, None),
                        (ReportColumnName.STRICT_VALIDITY, None),
                        (ReportColumnName.CONFIRMATION_COVERAGE, None),
                        (ReportColumnName.IS_SECONDARY_PAIR, True),
                    )
                )
            )
    return tuple(rows)


def _confirmation_results_rows(
    records: Sequence[MetricRecord],
    comparisons: Sequence[PairedComparisonRecord] = (),
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    pairs = sorted({record.pair for record in records})
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
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

        def _mean(metric_name: MetricId, pair_name: DirectedPairName = pair) -> float | None:
            values = [
                record.metric_value
                for record in records
                if record.pair == pair_name
                and record.method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
                and record.metric_name == metric_name
                and record.valid
                and record.metric_value is not None
            ]
            if not values:
                return None
            return sum(values) / len(values)

        rows.append(
            OrderedDict(
                (
                    (
                        ReportColumnName.PAIR,
                        pair,
                    ),
                    (
                        ReportColumnName.PROPOSALS,
                        proposals,
                    ),
                    (
                        ReportColumnName.ACCEPTED,
                        accepted,
                    ),
                    (
                        ReportColumnName.HARMFUL_ACCEPTED_RATE,
                        _mean(MetricId.HARMFUL_ACCEPTED_RATE),
                    ),
                    (
                        ReportColumnName.USEFUL_ACCEPTED_RATE,
                        _mean(MetricId.USEFUL_ACCEPTED_RATE),
                    ),
                    (
                        ReportColumnName.BENEFICIAL_REJECTED_RATE,
                        _mean(MetricId.BENEFICIAL_REJECTED_RATE),
                    ),
                    (
                        ReportColumnName.COVERAGE,
                        _mean(MetricId.COVERAGE_CONFIRM) or (accepted / proposals),
                    ),
                    (
                        ReportColumnName.NO_CONFIRM_HARMFUL_RATE,
                        _mean(MetricId.HARM_RATE_NO_CONFIRM),
                    ),
                    (
                        ReportColumnName.ARR,
                        _mean(MetricId.ABSOLUTE_RISK_REDUCTION),
                    ),
                    (
                        ReportColumnName.RRR,
                        _mean(MetricId.RELATIVE_RISK_REDUCTION),
                    ),
                    (
                        ReportColumnName.CI,
                        _confirmation_ci(comparisons, pair),
                    ),
                    (
                        ReportColumnName.P,
                        _confirmation_p(comparisons, pair),
                    ),
                )
            )
        )
    return tuple(rows)


def _confirmation_contrast(
    comparisons: Sequence[PairedComparisonRecord], pair: DirectedPairName
) -> PairedComparisonRecord | None:
    for record in comparisons:
        if record.family == MultiplicityFamily.CONFIRMATION_SAFETY and record.pair == pair:
            return record
    return None


def _confirmation_ci(
    comparisons: Sequence[PairedComparisonRecord], pair: DirectedPairName
) -> str | None:
    record = _confirmation_contrast(comparisons, pair)
    if record is None or record.bca_ci_low is None or record.bca_ci_high is None:
        return None
    return f"[{record.bca_ci_low}, {record.bca_ci_high}]"


def _confirmation_p(
    comparisons: Sequence[PairedComparisonRecord], pair: DirectedPairName
) -> float | None:
    record = _confirmation_contrast(comparisons, pair)
    if record is None:
        return None
    return record.holm_p


def _coupling_gap_row(
    condition_or_pair: str,
    gap_values: Sequence[float],
    fixed_action_values: Sequence[float],
    ci: str | None,
    holm_p: float | None,
) -> Mapping[ReportColumnName, TableScalar]:
    materiality = active_config().scientific.materiality.coupling_objective_units
    above_materiality = sum(1 for value in gap_values if value > materiality)
    return OrderedDict(
        (
            (
                ReportColumnName.CONDITION_OR_PAIR,
                condition_or_pair,
            ),
            (
                ReportColumnName.VALID_UNITS,
                len(gap_values),
            ),
            (
                ReportColumnName.FIXED_ACTION_GAP,
                statistics.fmean(fixed_action_values) if fixed_action_values else None,
            ),
            (
                ReportColumnName.ROBUST_COUPLING_GAP,
                statistics.fmean(gap_values) if gap_values else None,
            ),
            (
                ReportColumnName.FRACTION_ABOVE_MATERIALITY,
                above_materiality / len(gap_values) if gap_values else None,
            ),
            (
                ReportColumnName.CI,
                ci,
            ),
            (
                ReportColumnName.HOLM_P,
                holm_p,
            ),
            (
                ReportColumnName.COUPLING_DESTRUCTION_RETAINED_GAIN_FRACTION,
                None,
            ),
        )
    )


def _coupling_mechanism_results_rows(
    synthetic_records: Sequence[MetricRecord],
    real_packet_records: Sequence[MetricRecord],
    comparison_records: Sequence[PairedComparisonRecord],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
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
    comparisons_by_pair = OrderedDict(
        (comparison.pair, comparison)
        for comparison in comparison_records
        if comparison.family == MultiplicityFamily.COUPLING_MECHANISM
        and comparison.method_a == ExperimentLocalMethod.EXACT_ORBIT
    )
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


_WEAK_SIGNAL_BOUNDARY_DIMENSIONS = tuple(WeakSignalBoundaryDimension)


def _boundary_dimension_and_setting(
    condition: EvaluationConditionName,
) -> tuple[str, str]:
    for dimension in _WEAK_SIGNAL_BOUNDARY_DIMENSIONS:
        prefix = f"{dimension}-"
        if condition.startswith(prefix):
            return dimension, condition[len(prefix) :]
    return "semantic-sufficiency-partition", condition


def _failure_boundary_results_rows(
    weak_signal_records: Sequence[MetricRecord],
    semantic_records: Sequence[MetricRecord],
    primary_transfer_records: Sequence[MetricRecord],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for experiment_records in (weak_signal_records, semantic_records):
        cells: set[tuple[DirectedPairName, EvaluationConditionName, TransferMethod]] = {
            (record.pair, record.condition, record.method)
            for record in experiment_records
            if record.method != TransferMethod.LOCAL_ONLY
        }
        for pair, condition, method in sorted(
            cells, key=lambda cell: (cell[0], cell[1], cell[2].value)
        ):
            dimension, setting = _boundary_dimension_and_setting(condition)
            local_only_ce = _metric_value_by_condition(
                experiment_records, pair, TransferMethod.LOCAL_ONLY, condition
            )
            if local_only_ce is None:
                local_only_ce = _metric_value_by_condition(
                    primary_transfer_records,
                    pair,
                    TransferMethod.LOCAL_ONLY,
                    PRINCIPAL_EVALUATION_CONDITION.name,
                )
            method_ce = _metric_value_by_condition(experiment_records, pair, method, condition)
            realized_gain = (
                (local_only_ce - method_ce) / local_only_ce
                if local_only_ce is not None and method_ce is not None and local_only_ce != 0.0
                else None
            )
            certified_value = _metric_value_by_condition(
                experiment_records,
                pair,
                method,
                condition,
                metric_name=MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE,
            )
            abstention = _metric_value_by_condition(
                experiment_records,
                pair,
                method,
                condition,
                metric_name=MetricId.ABSTENTION_INDICATOR,
            )
            null_node_count = _metric_value_by_condition(
                experiment_records,
                pair,
                method,
                condition,
                metric_name=MetricId.NULL_NODE_COUNT,
            )
            confirmation_coverage = _metric_value_by_condition(
                experiment_records,
                pair,
                method,
                condition,
                metric_name=MetricId.PROPOSAL_ACCEPTANCE_RATE,
            )
            rows.append(
                OrderedDict(
                    (
                        (
                            ReportColumnName.BOUNDARY_DIMENSION,
                            dimension,
                        ),
                        (
                            ReportColumnName.SETTING,
                            setting,
                        ),
                        (
                            ReportColumnName.PAIR,
                            pair,
                        ),
                        (
                            ReportColumnName.METHOD,
                            method.value,
                        ),
                        (
                            ReportColumnName.CERTIFIED_VALUE,
                            certified_value,
                        ),
                        (
                            ReportColumnName.REALIZED_GAIN,
                            realized_gain,
                        ),
                        (
                            ReportColumnName.ABSTENTION,
                            abstention,
                        ),
                        (
                            ReportColumnName.NULL_NODE_COUNT,
                            null_node_count,
                        ),
                        (
                            ReportColumnName.CONFIRMATION_COVERAGE,
                            confirmation_coverage,
                        ),
                        (
                            ReportColumnName.STATE,
                            "Completed",
                        ),
                    )
                )
            )
    return tuple(rows)


def _numeric_row_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
    x_key: ReportColumnName,
    y_key: ReportColumnName,
) -> tuple[FigureSeries, ...]:
    xs: list[float] = []
    ys: list[float] = []
    for index, row in enumerate(rows):
        y_value = row.get(y_key)
        if not isinstance(y_value, int | float):
            continue
        x_value = row.get(x_key)
        xs.append(float(x_value) if isinstance(x_value, int | float) else float(index))
        ys.append(float(y_value))
    if not xs:
        return ()
    return (FigureSeries(name=ReportSeriesName(y_key), x=tuple(xs), y=tuple(ys)),)


def _sparsity_figure_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[FigureSeries, ...]:
    xs: list[float] = []
    ys: list[float] = []
    sizes: list[float] = []
    for row in rows:
        runtime = row.get(ReportColumnName.RUNTIME)
        gain = row.get(ReportColumnName.REALIZED_GAIN)
        memory = row.get(ReportColumnName.MEMORY)
        if not isinstance(runtime, int | float) or not isinstance(gain, int | float):
            continue
        xs.append(float(runtime))
        ys.append(float(gain))
        sizes.append(float(memory) if isinstance(memory, int | float) else 1.0)
    if not xs:
        return ()
    return (
        FigureSeries(
            name=ReportSeriesName("sparsity"),
            x=tuple(xs),
            y=tuple(ys),
            marker_sizes=tuple(sizes),
        ),
    )


def _confirmation_figure_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[FigureSeries, ...]:
    starts_x: list[float] = []
    starts_y: list[float] = []
    ends_x: list[float] = []
    ends_y: list[float] = []
    for row in rows:
        coverage = row.get(ReportColumnName.COVERAGE)
        harm = row.get(ReportColumnName.HARMFUL_ACCEPTED_RATE)
        no_confirm_harm = row.get(ReportColumnName.NO_CONFIRM_HARMFUL_RATE)
        if not isinstance(coverage, int | float):
            continue
        if not isinstance(harm, int | float) or not isinstance(no_confirm_harm, int | float):
            continue
        starts_x.append(float(coverage))
        starts_y.append(float(no_confirm_harm))
        ends_x.append(float(coverage))
        ends_y.append(float(harm))
    if not starts_x:
        return ()
    return (
        FigureSeries(
            name=ReportSeriesName("no-confirm to confirm"),
            x=tuple(starts_x),
            y=tuple(starts_y),
            arrow_x=tuple(ends_x),
            arrow_y=tuple(ends_y),
        ),
    )


def _semantic_sufficiency_series(
    records: Sequence[MetricRecord],
) -> tuple[FigureSeries, ...]:
    orbit: OrderedDict[tuple[DirectedPairName, RandomSeed, TransferMethod, str], float] = (
        OrderedDict()
    )
    local_ce: OrderedDict[tuple[DirectedPairName, RandomSeed, str], float] = OrderedDict()
    method_ce: OrderedDict[tuple[DirectedPairName, RandomSeed, TransferMethod, str], float] = (
        OrderedDict()
    )
    for record in records:
        if not record.valid or record.metric_value is None:
            continue
        if record.metric_name == MetricId.ORBIT_SIZE:
            orbit[(record.pair, record.seed, record.method, record.condition)] = record.metric_value
        if record.metric_name == MetricId.MACRO_CROSS_ENTROPY:
            if record.method == TransferMethod.LOCAL_ONLY:
                local_ce[(record.pair, record.seed, record.condition)] = record.metric_value
            else:
                method_ce[(record.pair, record.seed, record.method, record.condition)] = (
                    record.metric_value
                )
    by_method: OrderedDict[TransferMethod, list[tuple[float, float]]] = OrderedDict()
    for key, ce in method_ce.items():
        pair, seed, method, condition = key
        baseline = local_ce.get((pair, seed, condition))
        size = orbit.get(key)
        if baseline is None or baseline == 0.0 or size is None or size <= 0.0:
            continue
        by_method.setdefault(method, []).append((math.log(size), (baseline - ce) / baseline))
    return tuple(
        FigureSeries(
            name=ReportSeriesName(method.value),
            x=tuple(point[0] for point in points),
            y=tuple(point[1] for point in points),
        )
        for method, points in by_method.items()
        if points
    )


def _failure_boundary_figure_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[FigureSeries, ...]:
    grouped: OrderedDict[str, list[tuple[float, float]]] = OrderedDict()
    for index, row in enumerate(rows):
        dimension = row.get(ReportColumnName.BOUNDARY_DIMENSION)
        gain = row.get(ReportColumnName.REALIZED_GAIN)
        if not isinstance(dimension, str) or not isinstance(gain, int | float):
            continue
        setting = row.get(ReportColumnName.SETTING)
        x_value = float(setting) if isinstance(setting, int | float) else float(index)
        grouped.setdefault(dimension, []).append((x_value, float(gain)))
    return tuple(
        FigureSeries(
            name=ReportSeriesName(dimension),
            x=tuple(point[0] for point in points),
            y=tuple(point[1] for point in points),
        )
        for dimension, points in grouped.items()
        if points
    )


def _predicted_vs_realized_series(
    records: Sequence[MetricRecord],
) -> tuple[FigureSeries, ...]:
    from scipy.stats import spearmanr

    certified: OrderedDict[tuple[DirectedPairName, RandomSeed, TransferMethod], float] = (
        OrderedDict()
    )
    realized: OrderedDict[tuple[DirectedPairName, RandomSeed, TransferMethod], float] = (
        OrderedDict()
    )
    for record in records:
        if not record.valid or record.metric_value is None:
            continue
        key = (record.pair, record.seed, record.method)
        if record.metric_name == MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE:
            certified[key] = record.metric_value
        if record.metric_name == MetricId.RELATIVE_MACRO_CE_GAIN:
            realized[key] = record.metric_value
    by_pair: OrderedDict[DirectedPairName, list[tuple[float, float]]] = OrderedDict()
    for key, certified_value in certified.items():
        realized_value = realized.get(key)
        if realized_value is None:
            continue
        by_pair.setdefault(key[0], []).append((certified_value, realized_value))
    series: list[FigureSeries] = []
    for pair, points in by_pair.items():
        xs = tuple(point[0] for point in points)
        ys = tuple(point[1] for point in points)
        name = pair
        if len(points) >= active_config().scientific.statistics.spearman_minimum_valid_points:
            correlation = float(spearmanr(list(xs), list(ys)).statistic)
            name = DirectedPairName(f"{pair} Spearman={correlation:.3f}")
        series.append(FigureSeries(name=ReportSeriesName(name), x=xs, y=ys))
    return tuple(series)


def _scalability_figure_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[FigureSeries, ...]:
    return _numeric_row_series(
        rows, ReportColumnName.PREDICTED_WORK, ReportColumnName.RUNTIME_MEDIAN
    )


def _map_value_bound_series(
    records: Sequence[MetricRecord],
) -> tuple[FigureSeries, ...]:
    bounds: OrderedDict[tuple[str, RandomSeed], float] = OrderedDict()
    values: OrderedDict[tuple[str, RandomSeed], float] = OrderedDict()
    for record in records:
        if not record.valid or record.metric_value is None:
            continue
        key = (record.condition, record.seed)
        if record.metric_name == MetricId.ORBIT_RADIUS_MAP_BOUND:
            bounds[key] = record.metric_value
        if record.metric_name == MetricId.EXACT_MAP_ACTION_VALUE:
            values[key] = record.metric_value
    xs: list[float] = []
    ys: list[float] = []
    for key, bound in bounds.items():
        value = values.get(key)
        if value is None:
            continue
        xs.append(bound)
        ys.append(value)
    if not xs:
        return ()
    return (
        FigureSeries(
            name=ReportSeriesName(MetricId.EXACT_MAP_ACTION_VALUE.value),
            x=tuple(xs),
            y=tuple(ys),
        ),
    )


def _evidence_status_rows(
    rows: Sequence[EvidenceStatusRow],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    return tuple(
        OrderedDict(
            (
                (
                    ReportColumnName.QUESTION,
                    row.question,
                ),
                (
                    ReportColumnName.FINAL_STATE,
                    row.final_state,
                ),
                (
                    ReportColumnName.MATERIALITY_RESULT,
                    row.materiality_result,
                ),
                (
                    ReportColumnName.STATISTICAL_RESULT,
                    row.statistical_result,
                ),
                (
                    ReportColumnName.EVIDENCE_COMPLETENESS,
                    row.evidence_completeness,
                ),
                (
                    ReportColumnName.SCOPE,
                    row.scope,
                ),
                (
                    ReportColumnName.SUPPORTING_TABLE,
                    row.supporting_table,
                ),
                (
                    ReportColumnName.SUPPORTING_FIGURE,
                    row.supporting_figure,
                ),
                (
                    ReportColumnName.FORBIDDEN_WORDING,
                    row.forbidden_wording,
                ),
            )
        )
        for row in rows
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
                    ReportArtifactName.NUMERICAL_CONSTANTS_AND_SEEDS,
                ),
                (
                    experiment_matrix_table(_experiment_matrix_rows(catalogue)),
                    ReportArtifactName.EXPERIMENT_MATRIX,
                ),
                (
                    dataset_and_client_protocol_table(
                        _base_model_pilot_dataset_manifests(layout),
                        modality_by_dataset=_dataset_modality_by_dataset(),
                        role_by_dataset=_dataset_client_roles(),
                        excluded_class_counts=_excluded_class_counts(
                            _base_model_pilot_dataset_manifests(layout)
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
                            evidence_status_table(_evidence_status_rows(evidence_rows)),
                            ReportArtifactName.EVIDENCE_STATUS,
                        )
                    )
                )
            primary_transfer_metrics = completed_primary_transfer_metric_records(store)
            primary_transfer_comparisons = completed_primary_transfer_comparison_records(store)
            typer.echo(
                str(
                    writer.write_project_evidence_table(
                        primary_strict_transfer_results_table(
                            primary_transfer_metrics, primary_transfer_comparisons
                        ),
                        ReportArtifactName.PRIMARY_STRICT_TRANSFER_RESULTS,
                    )
                )
            )
            gain_series, gain_pair_labels = _real_transfer_gain_series(primary_transfer_comparisons)
            if gain_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            real_transfer_gain_forest_plot(gain_series, gain_pair_labels),
                            ReportArtifactName.REAL_TRANSFER_GAIN_FOREST_PLOT,
                        )
                    )
                )
            baseline_difference_series = _baseline_paired_difference_series(
                primary_transfer_metrics
            )
            if baseline_difference_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            baseline_paired_difference_plot(baseline_difference_series),
                            ReportArtifactName.BASELINE_PAIRED_DIFFERENCE_PLOT,
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
                            ReportArtifactName.EXACT_SOLVER_RESULTS,
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
                            ReportArtifactName.SCALABILITY_RESULTS,
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
                            ReportArtifactName.ABLATION_RESULTS,
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
                            ReportArtifactName.SPARSITY_AND_DENSE_RESULTS,
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
                            ReportArtifactName.GENERALIZATION_RESULTS,
                        )
                    )
                )
            confirmation_rows = _confirmation_results_rows(
                completed_experiment_metric_records(
                    store, ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY
                ),
                primary_transfer_comparisons,
            )
            if confirmation_rows:
                typer.echo(
                    str(
                        writer.write_project_evidence_table(
                            confirmation_results_table(confirmation_rows),
                            ReportArtifactName.CONFIRMATION_RESULTS,
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
                            ReportArtifactName.COUPLING_MECHANISM_RESULTS,
                        )
                    )
                )
            failure_boundary_rows = _failure_boundary_results_rows(
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
                boundary_series = _failure_boundary_figure_series(failure_boundary_rows)
                if boundary_series:
                    typer.echo(
                        str(
                            writer.write_project_evidence_figure(
                                failure_boundary_figure(boundary_series),
                                ReportArtifactName.FAILURE_BOUNDARY_FIGURE,
                            )
                        )
                    )
            coupling_series = _numeric_row_series(
                coupling_rows,
                ReportColumnName.CONDITION_OR_PAIR,
                ReportColumnName.FIXED_ACTION_GAP,
            )
            if coupling_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            coupling_gap_phase_figure(coupling_series),
                            ReportArtifactName.COUPLING_GAP_PHASE_FIGURE,
                        )
                    )
                )
            predicted_series = _predicted_vs_realized_series(primary_transfer_metrics)
            if predicted_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            predicted_vs_realized_transfer_figure(predicted_series),
                            ReportArtifactName.PREDICTED_VS_REALIZED_TRANSFER_FIGURE,
                        )
                    )
                )
            sparsity_series = _sparsity_figure_series(sparsity_and_dense_rows)
            if sparsity_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            sparsity_utility_efficiency_figure(sparsity_series),
                            ReportArtifactName.SPARSITY_UTILITY_EFFICIENCY_FIGURE,
                        )
                    )
                )
            confirmation_series = _confirmation_figure_series(confirmation_rows)
            if confirmation_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            confirmation_safety_coverage_figure(confirmation_series),
                            ReportArtifactName.CONFIRMATION_SAFETY_COVERAGE_FIGURE,
                        )
                    )
                )
            frontier_series = _semantic_sufficiency_series(
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
            scalability_series = _scalability_figure_series(scalability_rows)
            if scalability_series:
                typer.echo(
                    str(
                        writer.write_project_evidence_figure(
                            scalability_figure(scalability_series),
                            ReportArtifactName.SCALABILITY_FIGURE,
                        )
                    )
                )
            map_bound_series = _map_value_bound_series(
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
    except CliUsageError as error:
        exit_from_error(error)


def run(
    experiment_name: ExperimentName,
    overwrite: bool = False,
) -> None:
    try:
        resolved = experiment_identifier(experiment_name.value)
        definition = build_catalogue().definition(resolved)
        run_experiment(
            ExperimentExecutionRequest(
                experiment=resolved,
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
