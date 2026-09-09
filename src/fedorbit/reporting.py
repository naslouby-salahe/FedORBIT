from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from pydantic import JsonValue

from fedorbit.analysis.records import MetricRecord, PairedComparisonRecord
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
    RiskReductionColumn,
    TransferMethod,
)


def _report_columns(columns: Sequence[str]) -> ReportColumns:
    return tuple(ReportColumnName(column) for column in columns)


def _leaf_scalars(prefix: str, value: JsonValue) -> tuple[tuple[str, str], ...]:
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
        columns=_report_columns(("configuration_path", "value")),
        rows=tuple((path, value) for path, value in rows),
    )


def experiment_matrix_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "experiment",
            "classification",
            "datasets_or_pairs",
            "methods",
            "registered_seeds",
            "conditions",
            "derived_planned_cells",
            "prerequisites",
            "evidence_relationship",
        ),
        rows,
    )


def dataset_and_client_protocol_table(
    manifests: Sequence[DatasetManifest],
    modality_by_dataset: Mapping[str, str],
    role_by_dataset: Mapping[str, ClientRole],
    excluded_class_counts: Mapping[str, int],
) -> EvidenceTable:
    columns = (
        "dataset_component",
        "modality",
        "observed_raw_rows",
        "retained_rows",
        "timestamp_range",
        "local_prediction_classes",
        "feature_count",
        "transfer_candidates",
        "exclusions",
        "scientific_role",
        "raw_manifest_hash",
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
    pair: str,
    method: TransferMethod,
    metric_name: MetricId,
) -> float | None:
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
        "pair",
        "method",
        "valid_seeds",
        "test_macro_ce",
        "macro_f1",
        "balanced_accuracy",
        "gain_vs_local",
        "bca_ci_low",
        "bca_ci_high",
        "raw_p",
        "holm_p",
        "strict_validity",
        "confirmation_coverage",
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
    columns: tuple[str, ...], rows: Sequence[Mapping[str, TableScalar]]
) -> EvidenceTable:
    return EvidenceTable(
        columns=_report_columns(columns),
        rows=tuple(tuple(row[column] for column in columns) for row in rows),
    )


def transfer_ontology_and_null_padding_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "candidate_concept",
            "pair",
            "coarse_group",
            "source_real_or_null",
            "target_real_or_null",
            "support_counts",
            "action_eligibility",
            "null_reason",
        ),
        rows,
    )


def model_and_training_protocol_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "model",
            "architecture",
            "normalization",
            "activation",
            "initialization",
            "optimizer",
            "batch",
            "selected_learning_rate",
            "selected_weight_decay",
            "selected_dropout",
            "stopping_rule",
        ),
        rows,
    )


@dataclass(frozen=True, slots=True)
class _InformationResourceFacts:
    target_raw_data: bool
    anonymous_source_nodes: bool
    coarse_groups: bool
    source_response: bool
    target_local_response: bool
    fine_names: bool
    exact_map: bool
    confirmation: bool
    predecision_test_access: bool
    strict_compatibility: bool


_INFORMATION_RESOURCE_MATRIX_FACTS: Mapping[TransferMethod, _InformationResourceFacts] = (
    OrderedDict(
        (
            (
                TransferMethod.LOCAL_ONLY,
                _InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=False,
                    coarse_groups=False,
                    source_response=False,
                    target_local_response=False,
                    fine_names=False,
                    exact_map=False,
                    confirmation=False,
                    predecision_test_access=False,
                    strict_compatibility=True,
                ),
            ),
            (
                TransferMethod.LOCAL_SIR,
                _InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=False,
                    coarse_groups=True,
                    source_response=False,
                    target_local_response=True,
                    fine_names=False,
                    exact_map=False,
                    confirmation=True,
                    predecision_test_access=False,
                    strict_compatibility=True,
                ),
            ),
            (
                TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                _InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=True,
                    coarse_groups=True,
                    source_response=True,
                    target_local_response=False,
                    fine_names=False,
                    exact_map=False,
                    confirmation=True,
                    predecision_test_access=False,
                    strict_compatibility=True,
                ),
            ),
            (
                TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
                _InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=True,
                    coarse_groups=True,
                    source_response=True,
                    target_local_response=False,
                    fine_names=False,
                    exact_map=True,
                    confirmation=True,
                    predecision_test_access=False,
                    strict_compatibility=True,
                ),
            ),
            (
                TransferMethod.GENERIC_EXACT_QAP,
                _InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=True,
                    coarse_groups=True,
                    source_response=True,
                    target_local_response=False,
                    fine_names=False,
                    exact_map=True,
                    confirmation=True,
                    predecision_test_access=False,
                    strict_compatibility=True,
                ),
            ),
            (
                TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                _InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=True,
                    coarse_groups=True,
                    source_response=True,
                    target_local_response=False,
                    fine_names=False,
                    exact_map=False,
                    confirmation=True,
                    predecision_test_access=False,
                    strict_compatibility=True,
                ),
            ),
            (
                TransferMethod.EXACT_MAP_ORACLE,
                _InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=True,
                    coarse_groups=True,
                    source_response=True,
                    target_local_response=False,
                    fine_names=True,
                    exact_map=True,
                    confirmation=True,
                    predecision_test_access=False,
                    strict_compatibility=False,
                ),
            ),
        )
    )
)


def information_resource_matrix_table() -> EvidenceTable:
    columns = (
        "method",
        "target_raw_data",
        "anonymous_source_nodes",
        "coarse_groups",
        "source_response",
        "target_local_response",
        "fine_names",
        "exact_map",
        "confirmation",
        "predecision_test_access",
        "strict_compatibility",
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
        for method, facts in _INFORMATION_RESOURCE_MATRIX_FACTS.items()
    )
    return EvidenceTable(columns=_report_columns(columns), rows=rows)


def coupling_mechanism_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "condition_or_pair",
            "valid_units",
            "fixed_action_gap",
            "robust_coupling_gap",
            "fraction_above_materiality",
            "ci",
            "holm_p",
            "coupling_destruction_retained_gain_fraction",
        ),
        rows,
    )


def exact_solver_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "k",
            "block_pattern",
            "support",
            "truth_availability",
            "exact_mismatches",
            "maximum_absolute_error",
            "runtime_median",
            "runtime_p95",
            "qap_runtime",
            "dense_runtime",
            "timeouts",
            "memory",
            "active_images",
            "lap_calls",
        ),
        rows,
    )


def ablation_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "ablation",
            "pair",
            "realized_gain",
            "difference_vs_full",
            "equivalence",
            "retained_gain",
            "confirmation_safety",
        ),
        rows,
    )


def sparsity_and_dense_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "support_or_dense_condition",
            "pair",
            "realized_gain",
            "certified_value",
            "runtime",
            "memory",
            "confirmation_coverage",
            "dense_minus_sparse_difference",
        ),
        rows,
    )


def confirmation_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "pair",
            "proposals",
            "accepted",
            "harmful_accepted_rate",
            "useful_accepted_rate",
            "beneficial_rejected_rate",
            "coverage",
            "no_confirm_harmful_rate",
            RiskReductionColumn.ABSOLUTE_RISK_REDUCTION,
            RiskReductionColumn.RELATIVE_RISK_REDUCTION,
            "ci",
            "p",
        ),
        rows,
    )


def generalization_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "pair",
            "method",
            "valid_seeds",
            "test_macro_ce",
            "macro_f1",
            "balanced_accuracy",
            "gain_vs_local",
            "bca_ci_low",
            "bca_ci_high",
            "raw_p",
            "holm_p",
            "strict_validity",
            "confirmation_coverage",
            "is_secondary_pair",
        ),
        rows,
    )


def failure_boundary_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "boundary_dimension",
            "setting",
            "pair",
            "method",
            "certified_value",
            "realized_gain",
            "abstention",
            "null_node_count",
            "confirmation_coverage",
            "state",
        ),
        rows,
    )


def scalability_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "k",
            "block",
            "support",
            "method",
            "n_s",
            "lap_calls",
            "cuts",
            "runtime_median",
            "runtime_p95",
            "rss",
            "cuda_memory",
            "timeout",
            "exactness_status",
        ),
        rows,
    )


def _figure(x_label: str, y_label: str, series: Sequence[FigureSeries]) -> EvidenceFigure:
    return EvidenceFigure(
        x_label=ReportAxisLabel(x_label), y_label=ReportAxisLabel(y_label), series=tuple(series)
    )


def real_transfer_gain_forest_plot(series: Sequence[FigureSeries]) -> EvidenceFigure:
    return _figure("paired mean relative macro-CE gain vs local", "primary directed pair", series)


def baseline_paired_difference_plot(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure("primary directed pair", "seed-level paired difference", series)


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
    return _figure("boundary setting", "certified value / realized gain", series)


def scalability_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        "N_S * sum(n_g^3) (log scale)",
        "runtime (log scale)",
        series,
    )


def map_value_bound_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure("orbit-radius bound", "exact map action value", series)
