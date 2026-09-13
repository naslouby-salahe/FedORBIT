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
    DatasetId,
    DatasetModality,
    DirectedPairName,
    FedorbitConfigSection,
    Index,
    MetricId,
    ReferenceLineCoordinate,
    ReportAxisLabel,
    ReportColumnName,
    ReportColumns,
    ReportSeriesName,
    TransferMethod,
)


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
        columns=(ReportColumnName.CONFIGURATION_PATH, ReportColumnName.VALUE),
        rows=tuple((path, value) for path, value in rows),
    )


def experiment_matrix_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            ReportColumnName.EXPERIMENT,
            ReportColumnName.CLASSIFICATION,
            ReportColumnName.DATASETS_OR_PAIRS,
            ReportColumnName.METHODS,
            ReportColumnName.REGISTERED_SEEDS,
            ReportColumnName.CONDITIONS,
            ReportColumnName.DERIVED_PLANNED_CELLS,
            ReportColumnName.PREREQUISITES,
            ReportColumnName.EVIDENCE_RELATIONSHIP,
        ),
        rows,
    )


def dataset_and_client_protocol_table(
    manifests: Sequence[DatasetManifest],
    modality_by_dataset: Mapping[DatasetId, DatasetModality],
    role_by_dataset: Mapping[DatasetId, ClientRole],
    excluded_class_counts: Mapping[DatasetId, Index],
) -> EvidenceTable:
    columns = (
        ReportColumnName.DATASET_COMPONENT,
        ReportColumnName.MODALITY,
        ReportColumnName.OBSERVED_RAW_ROWS,
        ReportColumnName.RETAINED_ROWS,
        ReportColumnName.TIMESTAMP_RANGE,
        ReportColumnName.LOCAL_PREDICTION_CLASSES,
        ReportColumnName.FEATURE_COUNT,
        ReportColumnName.TRANSFER_CANDIDATES,
        ReportColumnName.EXCLUSIONS,
        ReportColumnName.SCIENTIFIC_ROLE,
        ReportColumnName.RAW_MANIFEST_HASH,
    )
    rows = tuple(
        (
            manifest.component,
            modality_by_dataset[manifest.dataset],
            sum(manifest.raw_counts.values()),
            sum(manifest.local_class_counts.values()),
            f"{manifest.timestamp_range[0]}..{manifest.timestamp_range[1]}",
            len(manifest.local_class_counts),
            len(manifest.adapter_feature_order),
            len(manifest.transfer_candidate_counts),
            excluded_class_counts.get(manifest.dataset, 0),
            role_by_dataset.get(manifest.dataset, ClientRole.PRIMARY),
            manifest.raw_sha256,
        )
        for manifest in manifests
    )
    return EvidenceTable(columns=columns, rows=rows)


def _metric_value(
    records: Sequence[MetricRecord],
    pair: DirectedPairName,
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
        ReportColumnName.PAIR,
        ReportColumnName.METHOD,
        ReportColumnName.VALID_SEEDS,
        ReportColumnName.TEST_MACRO_CE,
        ReportColumnName.MACRO_F1,
        ReportColumnName.BALANCED_ACCURACY,
        ReportColumnName.GAIN_VS_LOCAL,
        ReportColumnName.BCA_CI_LOW,
        ReportColumnName.BCA_CI_HIGH,
        ReportColumnName.RAW_P,
        ReportColumnName.HOLM_P,
        ReportColumnName.STRICT_VALIDITY,
        ReportColumnName.CONFIRMATION_COVERAGE,
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
    return EvidenceTable(columns=columns, rows=tuple(rows))


def _rows_table(
    columns: ReportColumns,
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return EvidenceTable(
        columns=columns,
        rows=tuple(tuple(row[column] for column in columns) for row in rows),
    )


def transfer_ontology_and_null_padding_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            ReportColumnName.CANDIDATE_CONCEPT,
            ReportColumnName.PAIR,
            ReportColumnName.COARSE_GROUP,
            ReportColumnName.SOURCE_REAL_OR_NULL,
            ReportColumnName.TARGET_REAL_OR_NULL,
            ReportColumnName.SUPPORT_COUNTS,
            ReportColumnName.ACTION_ELIGIBILITY,
            ReportColumnName.NULL_REASON,
        ),
        rows,
    )


def model_and_training_protocol_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            ReportColumnName.MODEL,
            ReportColumnName.ARCHITECTURE,
            ReportColumnName.NORMALIZATION,
            ReportColumnName.ACTIVATION,
            ReportColumnName.INITIALIZATION,
            ReportColumnName.OPTIMIZER,
            ReportColumnName.BATCH,
            ReportColumnName.SELECTED_LEARNING_RATE,
            ReportColumnName.SELECTED_WEIGHT_DECAY,
            ReportColumnName.SELECTED_DROPOUT,
            ReportColumnName.STOPPING_RULE,
        ),
        rows,
    )


def information_resource_matrix_table() -> EvidenceTable:
    columns = (
        ReportColumnName.METHOD,
        ReportColumnName.TARGET_RAW_DATA,
        ReportColumnName.ANONYMOUS_SOURCE_NODES,
        ReportColumnName.COARSE_GROUPS,
        ReportColumnName.SOURCE_RESPONSE,
        ReportColumnName.TARGET_LOCAL_RESPONSE,
        ReportColumnName.FINE_NAMES,
        ReportColumnName.EXACT_MAP,
        ReportColumnName.CONFIRMATION,
        ReportColumnName.PREDECISION_TEST_ACCESS,
        ReportColumnName.STRICT_COMPATIBILITY,
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
    return EvidenceTable(columns=columns, rows=rows)


def coupling_mechanism_results_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            ReportColumnName.CONDITION_OR_PAIR,
            ReportColumnName.VALID_UNITS,
            ReportColumnName.FIXED_ACTION_GAP,
            ReportColumnName.ROBUST_COUPLING_GAP,
            ReportColumnName.FRACTION_ABOVE_MATERIALITY,
            ReportColumnName.CI,
            ReportColumnName.HOLM_P,
            ReportColumnName.COUPLING_DESTRUCTION_RETAINED_GAIN_FRACTION,
        ),
        rows,
    )


def exact_solver_results_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            ReportColumnName.K,
            ReportColumnName.BLOCK_PATTERN,
            ReportColumnName.SUPPORT,
            ReportColumnName.TRUTH_AVAILABILITY,
            ReportColumnName.EXACT_MISMATCHES,
            ReportColumnName.MAXIMUM_ABSOLUTE_ERROR,
            ReportColumnName.RUNTIME_MEDIAN,
            ReportColumnName.RUNTIME_P95,
            ReportColumnName.QAP_RUNTIME,
            ReportColumnName.DENSE_RUNTIME,
            ReportColumnName.TIMEOUTS,
            ReportColumnName.MEMORY,
            ReportColumnName.ACTIVE_IMAGES,
            ReportColumnName.LAP_CALLS,
        ),
        rows,
    )


def ablation_results_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            ReportColumnName.ABLATION,
            ReportColumnName.PAIR,
            ReportColumnName.REALIZED_GAIN,
            ReportColumnName.DIFFERENCE_VS_FULL,
            ReportColumnName.EQUIVALENCE,
            ReportColumnName.RETAINED_GAIN,
            ReportColumnName.CONFIRMATION_SAFETY,
        ),
        rows,
    )


def sparsity_and_dense_results_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            ReportColumnName.SUPPORT_OR_DENSE_CONDITION,
            ReportColumnName.PAIR,
            ReportColumnName.REALIZED_GAIN,
            ReportColumnName.CERTIFIED_VALUE,
            ReportColumnName.RUNTIME,
            ReportColumnName.MEMORY,
            ReportColumnName.CONFIRMATION_COVERAGE,
            ReportColumnName.DENSE_MINUS_SPARSE_DIFFERENCE,
        ),
        rows,
    )


def evidence_status_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            ReportColumnName.QUESTION,
            ReportColumnName.FINAL_STATE,
            ReportColumnName.MATERIALITY_RESULT,
            ReportColumnName.STATISTICAL_RESULT,
            ReportColumnName.EVIDENCE_COMPLETENESS,
            ReportColumnName.SCOPE,
            ReportColumnName.SUPPORTING_TABLE,
            ReportColumnName.SUPPORTING_FIGURE,
            ReportColumnName.FORBIDDEN_WORDING,
        ),
        rows,
    )


def confirmation_results_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            ReportColumnName.PAIR,
            ReportColumnName.PROPOSALS,
            ReportColumnName.ACCEPTED,
            ReportColumnName.HARMFUL_ACCEPTED_RATE,
            ReportColumnName.USEFUL_ACCEPTED_RATE,
            ReportColumnName.BENEFICIAL_REJECTED_RATE,
            ReportColumnName.COVERAGE,
            ReportColumnName.NO_CONFIRM_HARMFUL_RATE,
            ReportColumnName.ARR,
            ReportColumnName.RRR,
            ReportColumnName.CI,
            ReportColumnName.P,
        ),
        rows,
    )


def generalization_results_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            ReportColumnName.PAIR,
            ReportColumnName.METHOD,
            ReportColumnName.VALID_SEEDS,
            ReportColumnName.TEST_MACRO_CE,
            ReportColumnName.MACRO_F1,
            ReportColumnName.BALANCED_ACCURACY,
            ReportColumnName.GAIN_VS_LOCAL,
            ReportColumnName.BCA_CI_LOW,
            ReportColumnName.BCA_CI_HIGH,
            ReportColumnName.RAW_P,
            ReportColumnName.HOLM_P,
            ReportColumnName.STRICT_VALIDITY,
            ReportColumnName.CONFIRMATION_COVERAGE,
            ReportColumnName.IS_SECONDARY_PAIR,
        ),
        rows,
    )


def failure_boundary_results_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            ReportColumnName.BOUNDARY_DIMENSION,
            ReportColumnName.SETTING,
            ReportColumnName.PAIR,
            ReportColumnName.METHOD,
            ReportColumnName.CERTIFIED_VALUE,
            ReportColumnName.REALIZED_GAIN,
            ReportColumnName.ABSTENTION,
            ReportColumnName.NULL_NODE_COUNT,
            ReportColumnName.CONFIRMATION_COVERAGE,
            ReportColumnName.STATE,
        ),
        rows,
    )


def scalability_results_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            ReportColumnName.K,
            ReportColumnName.BLOCK,
            ReportColumnName.SUPPORT,
            ReportColumnName.METHOD,
            ReportColumnName.N_S,
            ReportColumnName.LAP_CALLS,
            ReportColumnName.CUTS,
            ReportColumnName.RUNTIME_MEDIAN,
            ReportColumnName.RUNTIME_P95,
            ReportColumnName.RSS,
            ReportColumnName.CUDA_MEMORY,
            ReportColumnName.TIMEOUT,
            ReportColumnName.EXACTNESS_STATUS,
            ReportColumnName.PREDICTED_WORK,
        ),
        rows,
    )


def _figure(
    x_label: ReportAxisLabel,
    y_label: ReportAxisLabel,
    series: Sequence[FigureSeries],
    *,
    vertical_reference_lines: tuple[ReferenceLineCoordinate, ...] = (),
    horizontal_reference_lines: tuple[ReferenceLineCoordinate, ...] = (),
    log_x: bool = False,
    log_y: bool = False,
    draw_unit_diagonal: bool = False,
    separate_panels: bool = False,
    y_tick_labels: tuple[ReportSeriesName, ...] | None = None,
) -> EvidenceFigure:
    return EvidenceFigure(
        x_label=x_label,
        y_label=y_label,
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
        ReportAxisLabel("paired mean relative macro-CE gain vs local"),
        ReportAxisLabel("primary directed pair"),
        series,
        vertical_reference_lines=(0.0, material),
        y_tick_labels=pair_labels or None,
    )


def baseline_paired_difference_plot(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        ReportAxisLabel("seed"),
        ReportAxisLabel("seed-level paired difference"),
        series,
        horizontal_reference_lines=(0.0,),
        separate_panels=True,
    )


def coupling_gap_phase_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        ReportAxisLabel("coupling factor combination"),
        ReportAxisLabel("predicted structural zero/strict state"),
        series,
    )


def predicted_vs_realized_transfer_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        ReportAxisLabel("certified robust predicted value"),
        ReportAxisLabel("TEST relative macro-CE gain"),
        series,
    )


def sparsity_utility_efficiency_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        ReportAxisLabel("runtime"),
        ReportAxisLabel("realized gain"),
        series,
    )


def confirmation_safety_coverage_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        ReportAxisLabel("confirmation coverage"),
        ReportAxisLabel("harmful accepted rate"),
        series,
    )


def semantic_sufficiency_frontier_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        ReportAxisLabel("log|orbit|"),
        ReportAxisLabel("realized gain"),
        series,
    )


def failure_boundary_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        ReportAxisLabel("boundary setting"),
        ReportAxisLabel("certified value / realized gain"),
        series,
        separate_panels=True,
    )


def scalability_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        ReportAxisLabel("N_S * sum(n_g^3) (log scale)"),
        ReportAxisLabel("runtime (log scale)"),
        series,
        log_x=True,
        log_y=True,
    )


def map_value_bound_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:
    return _figure(
        ReportAxisLabel("orbit-radius bound"),
        ReportAxisLabel("exact map action value"),
        series,
        draw_unit_diagonal=True,
    )
