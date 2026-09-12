from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import JsonValue

from fedorbit.analysis.records import MetricRecord, PairedComparisonRecord
from fedorbit.analysis.resources import information_resource_catalogue
from fedorbit.config.loading import active_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.infrastructure.evidence import EvidenceExportError as EvidenceExportError
from fedorbit.infrastructure.evidence import EvidenceFigure as EvidenceFigure
from fedorbit.infrastructure.evidence import EvidenceTable as EvidenceTable
from fedorbit.infrastructure.evidence import EvidenceTablePayload as EvidenceTablePayload
from fedorbit.infrastructure.evidence import FigureError as FigureError
from fedorbit.infrastructure.evidence import FigureSeries as FigureSeries
from fedorbit.infrastructure.evidence import MetricArtifactPayload as MetricArtifactPayload
from fedorbit.infrastructure.evidence import TableError as TableError
from fedorbit.infrastructure.evidence import TableScalar as TableScalar
from fedorbit.infrastructure.evidence import VerifiedEvidenceWriter as VerifiedEvidenceWriter
from fedorbit.infrastructure.manifests import DatasetManifest
from fedorbit.types import (
    ClientRole,
    FedorbitConfigSection,
    MetricId,
    ReportAxisLabel,
    ReportColumnName,
    ReportColumns,
    ReportSeriesName,
    RiskReductionColumn,
    TransferMethod,
)


def _report_columns(columns: Sequence[str]) -> ReportColumns: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    return tuple(ReportColumnName(column) for column in columns)


def _leaf_scalars(prefix: str, value: JsonValue) -> tuple[tuple[str, str], ...]: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    if isinstance(value, Mapping):
        rows: list[tuple[str, str]] = []
        for key, item in value.items():
            rows.extend(_leaf_scalars(f"{prefix}.{key}", item))
        return tuple(rows)
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_leaf_scalars(f"{prefix}[{index}]", item))
        return tuple(rows)
    return ((prefix, str(value)),)


def numerical_constants_and_seeds_table(
    config: FedorbitConfig | None = None,
) -> EvidenceTable:
    resolved = config if config is not None else active_config()
    dumped = cast(Mapping[str, JsonValue], resolved.model_dump(mode="json"))
    rows = tuple(
        sorted(
            _leaf_scalars(
                FedorbitConfigSection.SCIENTIFIC,
                dumped.get(FedorbitConfigSection.SCIENTIFIC, OrderedDict()),
            )
        )
        + sorted(
            _leaf_scalars(
                FedorbitConfigSection.SOLVERS,
                dumped.get(FedorbitConfigSection.SOLVERS, OrderedDict()),
            )
        )
    )
    return EvidenceTable(
        columns=_report_columns(("configuration_path", "value")), #TODO: should be enums not hardcoded strings
        rows=tuple((path, value) for path, value in rows),
    )


def experiment_matrix_table(
    rows: Sequence[Mapping[str, TableScalar]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return _rows_table(
        (
            "experiment", #TODO: should be enums not hardcoded strings
            "classification", #TODO: should be enums not hardcoded strings
            "datasets_or_pairs", #TODO: should be enums not hardcoded strings
            "methods", #TODO: should be enums not hardcoded strings
            "registered_seeds", #TODO: should be enums not hardcoded strings
            "conditions", #TODO: should be enums not hardcoded strings
            "derived_planned_cells", #TODO: should be enums not hardcoded strings
            "prerequisites", #TODO: should be enums not hardcoded strings
            "evidence_relationship", #TODO: should be enums not hardcoded strings
        ),
        rows,
    )


def dataset_and_client_protocol_table(
    manifests: Sequence[DatasetManifest],
    modality_by_dataset: Mapping[str, str], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    role_by_dataset: Mapping[str, ClientRole], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    excluded_class_counts: Mapping[str, int], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    columns = (
        "dataset_component", #TODO: should be enums not hardcoded strings
        "modality", #TODO: should be enums not hardcoded strings
        "observed_raw_rows", #TODO: should be enums not hardcoded strings
        "retained_rows", #TODO: should be enums not hardcoded strings
        "timestamp_range", #TODO: should be enums not hardcoded strings
        "local_prediction_classes", #TODO: should be enums not hardcoded strings
        "feature_count", #TODO: should be enums not hardcoded strings
        "transfer_candidates", #TODO: should be enums not hardcoded strings
        "exclusions", #TODO: should be enums not hardcoded strings
        "scientific_role", #TODO: should be enums not hardcoded strings
        "raw_manifest_hash", #TODO: should be enums not hardcoded strings
    )
    rows = tuple(
        (
            manifest.component,
            modality_by_dataset.get(manifest.dataset.value, ""),
            sum(manifest.raw_counts.values()),
            sum(manifest.local_class_counts.values()),
            f"{manifest.timestamp_range[0]}..{manifest.timestamp_range[1]}",
            len(manifest.local_class_counts),
            len(manifest.adapter_feature_order),
            len(manifest.transfer_candidate_counts),
            excluded_class_counts.get(manifest.dataset.value, 0),
            role_by_dataset.get(manifest.dataset.value, ClientRole.PRIMARY).value,
            manifest.raw_sha256,
        )
        for manifest in manifests
    )
    return EvidenceTable(columns=_report_columns(columns), rows=rows)


def _metric_value(
    records: Sequence[MetricRecord],
    pair: str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    method: TransferMethod,
    metric_name: MetricId,
) -> float | None: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    matches = [
        record
        for record in records
        if record.pair == pair
        and record.method == method
        and record.metric_name == metric_name
        and record.valid
    ]
    if not matches:
        return None
    return sum(record.metric_value for record in matches if record.metric_value is not None) / len(
        matches
    )


def primary_strict_transfer_results_table(
    metric_records: Sequence[MetricRecord],
    comparison_records: Sequence[PairedComparisonRecord],
) -> EvidenceTable:
    columns = (
        "pair", #TODO: should be enums not hardcoded strings
        "method", #TODO: should be enums not hardcoded strings
        "valid_seeds", #TODO: should be enums not hardcoded strings
        "test_macro_ce", #TODO: should be enums not hardcoded strings
        "macro_f1", #TODO: should be enums not hardcoded strings
        "balanced_accuracy", #TODO: should be enums not hardcoded strings
        "gain_vs_local", #TODO: should be enums not hardcoded strings
        "bca_ci_low", #TODO: should be enums not hardcoded strings
        "bca_ci_high", #TODO: should be enums not hardcoded strings
        "raw_p", #TODO: should be enums not hardcoded strings
        "holm_p", #TODO: should be enums not hardcoded strings
        "strict_validity", #TODO: should be enums not hardcoded strings
        "confirmation_coverage", #TODO: should be enums not hardcoded strings
    )
    pairs = sorted({record.pair for record in metric_records})
    method_order = (
        TransferMethod.LOCAL_ONLY,
        TransferMethod.LOCAL_SIR,
        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
        TransferMethod.GENERIC_EXACT_QAP,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.EXACT_MAP_ORACLE,
    )
    rows: list[tuple[TableScalar, ...]] = []
    for pair in pairs:
        for method in method_order:
            comparison = next(
                (
                    record
                    for record in comparison_records
                    if record.pair == pair and record.method_a == method
                ),
                None,
            )
            rows.append(
                (
                    pair,
                    method.value,
                    comparison.paired_seed_count if comparison is not None else None,
                    _metric_value(metric_records, pair, method, MetricId.MACRO_CROSS_ENTROPY),
                    _metric_value(metric_records, pair, method, MetricId.MACRO_F1),
                    _metric_value(metric_records, pair, method, MetricId.BALANCED_ACCURACY),
                    comparison.mean_difference if comparison is not None else None,
                    comparison.bca_ci_low if comparison is not None else None,
                    comparison.bca_ci_high if comparison is not None else None,
                    comparison.raw_p if comparison is not None else None,
                    comparison.holm_p if comparison is not None else None,
                    comparison.decision.value if comparison is not None else None,
                    _metric_value(metric_records, pair, method, MetricId.COVERAGE_CONFIRM),
                )
            )
    return EvidenceTable(columns=_report_columns(columns), rows=tuple(rows))


def _rows_table(
    columns: tuple[str, ...], rows: Sequence[Mapping[str, TableScalar]] #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return EvidenceTable(
        columns=_report_columns(columns),
        rows=tuple(tuple(row[column] for column in columns) for row in rows),
    )


def transfer_ontology_and_null_padding_table(
    rows: Sequence[Mapping[str, TableScalar]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return _rows_table(
        (
            "candidate_concept", #TODO: should be enums not hardcoded strings
            "pair", #TODO: should be enums not hardcoded strings
            "coarse_group", #TODO: should be enums not hardcoded strings
            "source_real_or_null", #TODO: should be enums not hardcoded strings
            "target_real_or_null", #TODO: should be enums not hardcoded strings
            "support_counts", #TODO: should be enums not hardcoded strings
            "action_eligibility", #TODO: should be enums not hardcoded strings
            "null_reason", #TODO: should be enums not hardcoded strings
        ),
        rows,
    )


def model_and_training_protocol_table(
    rows: Sequence[Mapping[str, TableScalar]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return _rows_table(
        (
            "model", #TODO: should be enums not hardcoded strings
            "architecture", #TODO: should be enums not hardcoded strings
            "normalization", #TODO: should be enums not hardcoded strings
            "activation", #TODO: should be enums not hardcoded strings
            "initialization", #TODO: should be enums not hardcoded strings
            "optimizer", #TODO: should be enums not hardcoded strings
            "batch", #TODO: should be enums not hardcoded strings
            "selected_learning_rate", #TODO: should be enums not hardcoded strings
            "selected_weight_decay", #TODO: should be enums not hardcoded strings
            "selected_dropout", #TODO: should be enums not hardcoded strings
            "stopping_rule", #TODO: should be enums not hardcoded strings
        ),
        rows,
    )


def information_resource_matrix_table() -> EvidenceTable:
    columns = (
        "method", #TODO: should be enums not hardcoded strings
        "target_raw_data", #TODO: should be enums not hardcoded strings
        "anonymous_source_nodes", #TODO: should be enums not hardcoded strings
        "coarse_groups", #TODO: should be enums not hardcoded strings
        "source_response", #TODO: should be enums not hardcoded strings
        "target_local_response", #TODO: should be enums not hardcoded strings
        "fine_names", #TODO: should be enums not hardcoded strings
        "exact_map", #TODO: should be enums not hardcoded strings
        "confirmation", #TODO: should be enums not hardcoded strings
        "predecision_test_access", #TODO: should be enums not hardcoded strings
        "strict_compatibility", #TODO: should be enums not hardcoded strings
    )
    rows = tuple(
        (
            method.value,
            facts.target_raw_data,
            facts.anonymous_source_nodes,
            facts.coarse_groups,
            facts.source_response,
            facts.target_local_response,
            facts.fine_names,
            facts.exact_map,
            facts.confirmation,
            facts.predecision_test_access,
            facts.strict_compatibility,
        )
        for method, facts in information_resource_catalogue().items()
    )
    return EvidenceTable(columns=_report_columns(columns), rows=rows)


def coupling_mechanism_results_table(
    rows: Sequence[Mapping[str, TableScalar]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return _rows_table(
        (
            "condition_or_pair", #TODO: should be enums not hardcoded strings
            "valid_units", #TODO: should be enums not hardcoded strings
            "fixed_action_gap", #TODO: should be enums not hardcoded strings
            "robust_coupling_gap", #TODO: should be enums not hardcoded strings
            "fraction_above_materiality", #TODO: should be enums not hardcoded strings
            "ci", #TODO: should be enums not hardcoded strings
            "holm_p", #TODO: should be enums not hardcoded strings
            "coupling_destruction_retained_gain_fraction", #TODO: should be enums not hardcoded strings
        ),
        rows,
    )


def exact_solver_results_table(
    rows: Sequence[Mapping[str, TableScalar]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return _rows_table(
        (
            "k", #TODO: should be enums not hardcoded strings
            "block_pattern", #TODO: should be enums not hardcoded strings
            "support", #TODO: should be enums not hardcoded strings
            "truth_availability", #TODO: should be enums not hardcoded strings
            "exact_mismatches", #TODO: should be enums not hardcoded strings
            "maximum_absolute_error", #TODO: should be enums not hardcoded strings
            "runtime_median", #TODO: should be enums not hardcoded strings
            "runtime_p95", #TODO: should be enums not hardcoded strings
            "qap_runtime", #TODO: should be enums not hardcoded strings
            "dense_runtime", #TODO: should be enums not hardcoded strings
            "timeouts", #TODO: should be enums not hardcoded strings
            "memory", #TODO: should be enums not hardcoded strings
            "active_images", #TODO: should be enums not hardcoded strings
            "lap_calls", #TODO: should be enums not hardcoded strings
        ),
        rows,
    )


def ablation_results_table(
    rows: Sequence[Mapping[str, TableScalar]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return _rows_table(
        (
            "ablation", #TODO: should be enums not hardcoded strings
            "pair", #TODO: should be enums not hardcoded strings
            "realized_gain", #TODO: should be enums not hardcoded strings
            "difference_vs_full", #TODO: should be enums not hardcoded strings
            "equivalence", #TODO: should be enums not hardcoded strings
            "retained_gain", #TODO: should be enums not hardcoded strings
            "confirmation_safety", #TODO: should be enums not hardcoded strings
        ),
        rows,
    )


def sparsity_and_dense_results_table(
    rows: Sequence[Mapping[str, TableScalar]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return _rows_table(
        (
            "support_or_dense_condition", #TODO: should be enums not hardcoded strings
            "pair", #TODO: should be enums not hardcoded strings
            "realized_gain", #TODO: should be enums not hardcoded strings
            "certified_value", #TODO: should be enums not hardcoded strings
            "runtime", #TODO: should be enums not hardcoded strings
            "memory", #TODO: should be enums not hardcoded strings
            "confirmation_coverage", #TODO: should be enums not hardcoded strings
            "dense_minus_sparse_difference", #TODO: should be enums not hardcoded strings
        ),
        rows,
    )


def evidence_status_table(
    rows: Sequence[Mapping[str, TableScalar]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return _rows_table(
        (
            "question", #TODO: should be enums not hardcoded strings
            "final_state", #TODO: should be enums not hardcoded strings
            "materiality_result", #TODO: should be enums not hardcoded strings
            "statistical_result", #TODO: should be enums not hardcoded strings
            "evidence_completeness", #TODO: should be enums not hardcoded strings
            "scope", #TODO: should be enums not hardcoded strings
            "supporting_table", #TODO: should be enums not hardcoded strings
            "supporting_figure", #TODO: should be enums not hardcoded strings
            "forbidden_wording", #TODO: should be enums not hardcoded strings
        ),
        rows,
    )


def confirmation_results_table(
    rows: Sequence[Mapping[str, TableScalar]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return _rows_table(
        (
            "pair", #TODO: should be enums not hardcoded strings
            "proposals", #TODO: should be enums not hardcoded strings
            "accepted", #TODO: should be enums not hardcoded strings
            "harmful_accepted_rate", #TODO: should be enums not hardcoded strings
            "useful_accepted_rate", #TODO: should be enums not hardcoded strings
            "beneficial_rejected_rate", #TODO: should be enums not hardcoded strings
            "coverage", #TODO: should be enums not hardcoded strings
            "no_confirm_harmful_rate", #TODO: should be enums not hardcoded strings
            RiskReductionColumn.ABSOLUTE_RISK_REDUCTION,
            RiskReductionColumn.RELATIVE_RISK_REDUCTION,
            "ci", #TODO: should be enums not hardcoded strings
            "p", #TODO: should be enums not hardcoded strings
        ),
        rows,
    )


def generalization_results_table(
    rows: Sequence[Mapping[str, TableScalar]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return _rows_table(
        (
            "pair", #TODO: should be enums not hardcoded strings
            "method", #TODO: should be enums not hardcoded strings
            "valid_seeds", #TODO: should be enums not hardcoded strings
            "test_macro_ce", #TODO: should be enums not hardcoded strings
            "macro_f1", #TODO: should be enums not hardcoded strings
            "balanced_accuracy", #TODO: should be enums not hardcoded strings
            "gain_vs_local", #TODO: should be enums not hardcoded strings
            "bca_ci_low", #TODO: should be enums not hardcoded strings
            "bca_ci_high", #TODO: should be enums not hardcoded strings
            "raw_p", #TODO: should be enums not hardcoded strings
            "holm_p", #TODO: should be enums not hardcoded strings
            "strict_validity", #TODO: should be enums not hardcoded strings
            "confirmation_coverage", #TODO: should be enums not hardcoded strings
            "is_secondary_pair", #TODO: should be enums not hardcoded strings
        ),
        rows,
    )


def failure_boundary_results_table(
    rows: Sequence[Mapping[str, TableScalar]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return _rows_table(
        (
            "boundary_dimension", #TODO: should be enums not hardcoded strings
            "setting", #TODO: should be enums not hardcoded strings
            "pair", #TODO: should be enums not hardcoded strings
            "method", #TODO: should be enums not hardcoded strings
            "certified_value", #TODO: should be enums not hardcoded strings
            "realized_gain", #TODO: should be enums not hardcoded strings
            "abstention", #TODO: should be enums not hardcoded strings
            "null_node_count", #TODO: should be enums not hardcoded strings
            "confirmation_coverage", #TODO: should be enums not hardcoded strings
            "state", #TODO: should be enums not hardcoded strings
        ),
        rows,
    )


def scalability_results_table(
    rows: Sequence[Mapping[str, TableScalar]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceTable:
    return _rows_table(
        (
            "k", #TODO: should be enums not hardcoded strings
            "block", #TODO: should be enums not hardcoded strings
            "support", #TODO: should be enums not hardcoded strings
            "method", #TODO: should be enums not hardcoded strings
            "n_s", #TODO: should be enums not hardcoded strings
            "lap_calls", #TODO: should be enums not hardcoded strings
            "cuts", #TODO: should be enums not hardcoded strings
            "runtime_median", #TODO: should be enums not hardcoded strings
            "runtime_p95", #TODO: should be enums not hardcoded strings
            "rss", #TODO: should be enums not hardcoded strings
            "cuda_memory", #TODO: should be enums not hardcoded strings
            "timeout", #TODO: should be enums not hardcoded strings
            "exactness_status", #TODO: should be enums not hardcoded strings
            "predicted_work", #TODO: should be enums not hardcoded strings
        ),
        rows,
    )


def _figure(
    x_label: str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    y_label: str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    series: Sequence[FigureSeries],
    *,
    vertical_reference_lines: tuple[float, ...] = (), #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    horizontal_reference_lines: tuple[float, ...] = (), #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    log_x: bool = False,
    log_y: bool = False,
    draw_unit_diagonal: bool = False,
    separate_panels: bool = False,
    y_tick_labels: tuple[ReportSeriesName, ...] | None = None,
) -> EvidenceFigure:
    return EvidenceFigure(
        x_label=ReportAxisLabel(x_label),
        y_label=ReportAxisLabel(y_label),
        series=tuple(series),
        vertical_reference_lines=vertical_reference_lines,
        horizontal_reference_lines=horizontal_reference_lines,
        log_x=log_x,
        log_y=log_y,
        draw_unit_diagonal=draw_unit_diagonal,
        separate_panels=separate_panels,
        y_tick_labels=y_tick_labels,
    )


def real_transfer_gain_forest_plot(
    series: Sequence[FigureSeries],
    pair_labels: tuple[ReportSeriesName, ...] = (),
) -> EvidenceFigure:
    material = active_config().scientific.materiality.realized_relative_macro_ce
    return _figure(
        "paired mean relative macro-CE gain vs local",
        "primary directed pair",
        series,
        vertical_reference_lines=(0.0, material),
        y_tick_labels=pair_labels or None,
    )


def baseline_paired_difference_plot(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        "seed", #TODO: should be enums not hardcoded strings
        "seed-level paired difference",
        series,
        horizontal_reference_lines=(0.0,),
        separate_panels=True,
    )


def coupling_gap_phase_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure("coupling factor combination", "predicted structural zero/strict state", series)


def predicted_vs_realized_transfer_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure("certified robust predicted value", "TEST relative macro-CE gain", series)


def sparsity_utility_efficiency_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure("runtime", "realized gain", series)


def confirmation_safety_coverage_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure("confirmation coverage", "harmful accepted rate", series)


def semantic_sufficiency_frontier_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure("log|orbit|", "realized gain", series)


def failure_boundary_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        "boundary setting",
        "certified value / realized gain",
        series,
        separate_panels=True,
    )


def scalability_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        "N_S * sum(n_g^3) (log scale)",
        "runtime (log scale)",
        series,
        log_x=True,
        log_y=True,
    )


def map_value_bound_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        "orbit-radius bound",
        "exact map action value",
        series,
        draw_unit_diagonal=True,
    )
