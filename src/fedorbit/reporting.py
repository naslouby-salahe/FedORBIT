from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import cast

from pydantic import JsonValue

from fedorbit.analysis.records import MetricRecord, PairedComparisonRecord
from fedorbit.analysis.resources import information_resource_catalogue
from fedorbit.config.loading import active_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.infrastructure.evidence import UNAVAILABLE_CELL_TEXT as UNAVAILABLE_CELL_TEXT
from fedorbit.infrastructure.evidence import EvidenceExportError as EvidenceExportError
from fedorbit.infrastructure.evidence import EvidenceFigure as EvidenceFigure
from fedorbit.infrastructure.evidence import EvidenceTable as EvidenceTable
from fedorbit.infrastructure.evidence import FigureError as FigureError
from fedorbit.infrastructure.evidence import FigureSeries as FigureSeries
from fedorbit.infrastructure.evidence import MetricArtifactPayload as MetricArtifactPayload
from fedorbit.infrastructure.evidence import TableError as TableError
from fedorbit.infrastructure.evidence import TableScalar as TableScalar
from fedorbit.infrastructure.evidence import VerifiedEvidenceWriter as VerifiedEvidenceWriter
from fedorbit.infrastructure.evidence import (
    format_balanced_accuracy,
    format_integer,
    format_interval_estimate,
    format_macro_f1,
    format_memory_mib,
    format_p_value,
    format_runtime_seconds,
    format_scientific_metric,
)
from fedorbit.infrastructure.manifests import DatasetManifest
from fedorbit.types import (
    PRINCIPAL_EVALUATION_CONDITION,
    BoundaryReportState,
    ClientRole,
    DatasetId,
    DatasetModality,
    DirectedPairName,
    EvaluationCondition,
    EvaluationConditionKind,
    EvaluationConditionName,
    EvidenceHypothesis,
    FedorbitConfigSection,
    Index,
    MetricId,
    MultiplicityFamily,
    ReferenceLineCoordinate,
    RelativeGain,
    ReportArtifactName,
    ReportAxisLabel,
    ReportColumnName,
    ReportColumns,
    ReportSeriesName,
    SupportSize,
    TransferMethod,
)

ColumnFormatter = Callable[[float], str]


COLUMN_FORMATTERS: Mapping[ReportColumnName, ColumnFormatter] = OrderedDict(
    (
        (ReportColumnName.TEST_MACRO_CE, format_scientific_metric),
        (ReportColumnName.GAIN_VS_LOCAL, format_scientific_metric),
        (ReportColumnName.REALIZED_GAIN, format_scientific_metric),
        (ReportColumnName.RETAINED_GAIN, format_scientific_metric),
        (ReportColumnName.DIFFERENCE_VS_FULL, format_scientific_metric),
        (ReportColumnName.DENSE_MINUS_SPARSE_DIFFERENCE, format_scientific_metric),
        (ReportColumnName.CERTIFIED_VALUE, format_scientific_metric),
        (ReportColumnName.FIXED_ACTION_GAP, format_scientific_metric),
        (ReportColumnName.ROBUST_COUPLING_GAP, format_scientific_metric),
        (ReportColumnName.FRACTION_ABOVE_MATERIALITY, format_scientific_metric),
        (ReportColumnName.COUPLING_DESTRUCTION_RETAINED_GAIN_FRACTION, format_scientific_metric),
        (ReportColumnName.MAXIMUM_ABSOLUTE_ERROR, format_scientific_metric),
        (ReportColumnName.CONFIRMATION_SAFETY, format_scientific_metric),
        (ReportColumnName.EQUIVALENCE, format_scientific_metric),
        (ReportColumnName.CONFIRMATION_COVERAGE, format_scientific_metric),
        (ReportColumnName.COVERAGE, format_scientific_metric),
        (ReportColumnName.HARMFUL_ACCEPTED_RATE, format_scientific_metric),
        (ReportColumnName.USEFUL_ACCEPTED_RATE, format_scientific_metric),
        (ReportColumnName.BENEFICIAL_REJECTED_RATE, format_scientific_metric),
        (ReportColumnName.NO_CONFIRM_HARMFUL_RATE, format_scientific_metric),
        (ReportColumnName.ARR, format_scientific_metric),
        (ReportColumnName.RRR, format_scientific_metric),
        (ReportColumnName.MACRO_F1, format_macro_f1),
        (ReportColumnName.BALANCED_ACCURACY, format_balanced_accuracy),
        (ReportColumnName.RAW_P, format_p_value),
        (ReportColumnName.HOLM_P, format_p_value),
        (ReportColumnName.P, format_p_value),
        (ReportColumnName.RUNTIME, format_runtime_seconds),
        (ReportColumnName.RUNTIME_MEDIAN, format_runtime_seconds),
        (ReportColumnName.RUNTIME_P95, format_runtime_seconds),
        (ReportColumnName.QAP_RUNTIME, format_runtime_seconds),
        (ReportColumnName.DENSE_RUNTIME, format_runtime_seconds),
        (ReportColumnName.MEMORY, format_memory_mib),
        (ReportColumnName.RSS, format_memory_mib),
    )
)

COUNT_COLUMNS: frozenset[ReportColumnName] = frozenset(
    {
        ReportColumnName.VALID_SEEDS,
        ReportColumnName.VALID_UNITS,
        ReportColumnName.PROPOSALS,
        ReportColumnName.ACCEPTED,
        ReportColumnName.EXACT_MISMATCHES,
        ReportColumnName.ACTIVE_IMAGES,
        ReportColumnName.LAP_CALLS,
        ReportColumnName.N_S,
        ReportColumnName.TIMEOUTS,
        ReportColumnName.TIMEOUT,
        ReportColumnName.CUTS,
        ReportColumnName.NULL_NODE_COUNT,
        ReportColumnName.OBSERVED_RAW_ROWS,
        ReportColumnName.RETAINED_ROWS,
        ReportColumnName.FEATURE_COUNT,
        ReportColumnName.TRANSFER_CANDIDATES,
        ReportColumnName.LOCAL_PREDICTION_CLASSES,
        ReportColumnName.EXCLUSIONS,
        ReportColumnName.DERIVED_PLANNED_CELLS,
        ReportColumnName.K,
        ReportColumnName.SUPPORT,
        ReportColumnName.DUPLICATE_ROW_COUNT,
        ReportColumnName.OCCURRENCE_COUNT,
    }
)


def _rendered_cell(column: ReportColumnName, value: TableScalar) -> TableScalar:
    if value is None or isinstance(value, str | bool):
        return value
    if column in COUNT_COLUMNS:
        return format_integer(float(value))
    formatter = COLUMN_FORMATTERS.get(column)
    if formatter is None:
        return value
    return formatter(float(value))


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
        ReportColumnName.CI,
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
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
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
                OrderedDict(
                    (
                        (ReportColumnName.PAIR, pair),
                        (ReportColumnName.METHOD, method.value),
                        (
                            ReportColumnName.VALID_SEEDS,
                            comparison.paired_seed_count if comparison is not None else None,
                        ),
                        (
                            ReportColumnName.TEST_MACRO_CE,
                            _metric_value(
                                metric_records, pair, method, MetricId.MACRO_CROSS_ENTROPY
                            ),
                        ),
                        (
                            ReportColumnName.MACRO_F1,
                            _metric_value(metric_records, pair, method, MetricId.MACRO_F1),
                        ),
                        (
                            ReportColumnName.BALANCED_ACCURACY,
                            _metric_value(metric_records, pair, method, MetricId.BALANCED_ACCURACY),
                        ),
                        (
                            ReportColumnName.GAIN_VS_LOCAL,
                            comparison.mean_difference if comparison is not None else None,
                        ),
                        (
                            ReportColumnName.CI,
                            None
                            if comparison is None or comparison.mean_difference is None
                            else format_interval_estimate(
                                comparison.mean_difference,
                                comparison.bca_ci_low,
                                comparison.bca_ci_high,
                            ),
                        ),
                        (
                            ReportColumnName.RAW_P,
                            comparison.raw_p if comparison is not None else None,
                        ),
                        (
                            ReportColumnName.HOLM_P,
                            comparison.holm_p if comparison is not None else None,
                        ),
                        (
                            ReportColumnName.STRICT_VALIDITY,
                            comparison.decision.value if comparison is not None else None,
                        ),
                        (
                            ReportColumnName.CONFIRMATION_COVERAGE,
                            _metric_value(metric_records, pair, method, MetricId.COVERAGE_CONFIRM),
                        ),
                    )
                )
            )
    return _rows_table(columns, rows)


def _rows_table(
    columns: ReportColumns,
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> EvidenceTable:
    return EvidenceTable(
        columns=columns,
        rows=tuple(
            tuple(_rendered_cell(column, row.get(column)) for column in columns) for row in rows
        ),
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


def _mean_stored_metric(
    records: Sequence[MetricRecord],
    metric_name: MetricId,
    pair: DirectedPairName,
    method: TransferMethod,
    condition: EvaluationConditionName,
) -> float | None:
    values = [
        float(record.metric_value)
        for record in records
        if record.pair == pair
        and record.method == method
        and record.condition == condition
        and record.metric_name == metric_name
        and record.valid
        and record.metric_value is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _stored_equivalence_decision(
    comparisons: Sequence[PairedComparisonRecord],
    pair: DirectedPairName,
    method: TransferMethod,
    family: MultiplicityFamily,
) -> str | None:
    for comparison in comparisons:
        if (
            comparison.family == family
            and comparison.pair == pair
            and comparison.method_b == method
            and comparison.equivalence_margin_low is not None
        ):
            return comparison.decision.value
    return None


def _row_with_filled_columns(
    row: Mapping[ReportColumnName, TableScalar],
    filling: Mapping[ReportColumnName, TableScalar],
) -> Mapping[ReportColumnName, TableScalar]:
    completed = OrderedDict(row)
    for column, value in filling.items():
        if completed.get(column) is None:
            completed[column] = value
    return completed


def ablation_results_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
    metric_records: Sequence[MetricRecord] = (),
    comparison_records: Sequence[PairedComparisonRecord] = (),
) -> EvidenceTable:
    filled = tuple(
        _row_with_filled_columns(
            row,
            _ablation_filling(row, metric_records, comparison_records),
        )
        for row in rows
    )
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
        filled,
    )


def _ablation_filling(
    row: Mapping[ReportColumnName, TableScalar],
    metric_records: Sequence[MetricRecord],
    comparison_records: Sequence[PairedComparisonRecord],
) -> Mapping[ReportColumnName, TableScalar]:
    pair = row.get(ReportColumnName.PAIR)
    ablation = row.get(ReportColumnName.ABLATION)
    if not isinstance(pair, str) or not isinstance(ablation, str):
        return OrderedDict()
    method = next(
        (candidate for candidate in TransferMethod if candidate.value == ablation),
        None,
    )
    if method is None:
        return OrderedDict()
    pair_name = DirectedPairName(pair)
    return OrderedDict(
        (
            (
                ReportColumnName.REALIZED_GAIN,
                _mean_stored_metric(
                    metric_records,
                    MetricId.RELATIVE_MACRO_CE_GAIN,
                    pair_name,
                    method,
                    PRINCIPAL_EVALUATION_CONDITION.name,
                ),
            ),
            (
                ReportColumnName.EQUIVALENCE,
                _stored_equivalence_decision(
                    comparison_records,
                    pair_name,
                    method,
                    MultiplicityFamily.MECHANISM_ABLATIONS,
                ),
            ),
        )
    )


def sparsity_and_dense_results_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
    metric_records: Sequence[MetricRecord] = (),
) -> EvidenceTable:
    gains = _sparsity_stored_gains(rows, metric_records)
    filled = tuple(
        _row_with_filled_columns(
            row,
            OrderedDict(
                (
                    (
                        ReportColumnName.DENSE_MINUS_SPARSE_DIFFERENCE,
                        _dense_minus_sparse_difference(row, gains),
                    ),
                )
            ),
        )
        for row in rows
    )
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
        filled,
    )


def _sparsity_stored_gains(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
    metric_records: Sequence[MetricRecord],
) -> Mapping[tuple[DirectedPairName, TransferMethod, EvaluationConditionName], RelativeGain]:
    dense_condition = EvaluationCondition(EvaluationConditionKind.DENSE_CCP).name
    sparse_conditions = tuple(
        EvaluationCondition.exact_sparse(SupportSize(size)).name
        for size in (
            *active_config().scientific.action.sparse_support_sensitivity,
            active_config().scientific.action.principal_sparse_support,
        )
    )
    candidate_methods = (
        (TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK, (dense_condition,)),
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, sparse_conditions),
    )
    gains: OrderedDict[
        tuple[DirectedPairName, TransferMethod, EvaluationConditionName], RelativeGain
    ] = OrderedDict()
    for row in rows:
        pair = row.get(ReportColumnName.PAIR)
        if not isinstance(pair, str):
            continue
        for method, conditions in candidate_methods:
            for condition in conditions:
                value = _mean_stored_metric(
                    metric_records,
                    MetricId.RELATIVE_MACRO_CE_GAIN,
                    DirectedPairName(pair),
                    method,
                    condition,
                )
                if value is not None:
                    gains[(DirectedPairName(pair), method, condition)] = value
    return gains


def _dense_minus_sparse_difference(
    row: Mapping[ReportColumnName, TableScalar],
    gains: Mapping[tuple[DirectedPairName, TransferMethod, EvaluationConditionName], RelativeGain],
) -> RelativeGain | None:
    pair = row.get(ReportColumnName.PAIR)
    condition = row.get(ReportColumnName.SUPPORT_OR_DENSE_CONDITION)
    if not isinstance(pair, str) or not isinstance(condition, str) or not condition:
        return None
    dense = gains.get(
        (
            DirectedPairName(pair),
            TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
            EvaluationConditionName(condition),
        )
    )
    sparse = gains.get(
        (
            DirectedPairName(pair),
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            EvaluationCondition.exact_sparse(
                SupportSize(active_config().scientific.action.principal_sparse_support)
            ).name,
        )
    )
    if dense is None or sparse is None:
        return None
    return dense - sparse


def _registered_confirmation_rows(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
    registered_pairs: Sequence[DirectedPairName],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    recorded = {row.get(ReportColumnName.PAIR) for row in rows}
    completed: list[Mapping[ReportColumnName, TableScalar]] = list(rows)
    for pair in registered_pairs:
        if pair in recorded:
            continue
        completed.append(
            OrderedDict(
                (
                    (ReportColumnName.PAIR, pair),
                    (ReportColumnName.PROPOSALS, 0),
                    (ReportColumnName.ACCEPTED, 0),
                )
            )
        )
    return tuple(completed)


def confirmation_results_table(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
    registered_pairs: Sequence[DirectedPairName] = (),
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
        _registered_confirmation_rows(rows, registered_pairs),
    )


EVIDENCE_SUPPORT_ARTIFACTS: Mapping[
    EvidenceHypothesis, tuple[ReportArtifactName, ReportArtifactName]
] = OrderedDict(
    (
        (
            EvidenceHypothesis.EXACT_SPARSE_SEPARATOR_EXACTNESS,
            (ReportArtifactName.EXACT_SOLVER_RESULTS, ReportArtifactName.SCALABILITY_FIGURE),
        ),
        (
            EvidenceHypothesis.JOINT_CORRESPONDENCE_AVOIDS_RECTANGULAR_PESSIMISM,
            (
                ReportArtifactName.COUPLING_MECHANISM_RESULTS,
                ReportArtifactName.COUPLING_GAP_PHASE_FIGURE,
            ),
        ),
        (
            EvidenceHypothesis.ACTION_CERTIFICATION_WITHOUT_FINE_MAP_IDENTIFICATION,
            (ReportArtifactName.EXACT_SOLVER_RESULTS, ReportArtifactName.MAP_VALUE_BOUND_FIGURE),
        ),
        (
            EvidenceHypothesis.STRICT_CROSS_TELEMETRY_TRANSFER_UTILITY,
            (
                ReportArtifactName.PRIMARY_STRICT_TRANSFER_RESULTS,
                ReportArtifactName.REAL_TRANSFER_GAIN_FOREST_PLOT,
            ),
        ),
        (
            EvidenceHypothesis.VALUE_OF_EXTERNAL_PROCEDURAL_EVIDENCE,
            (
                ReportArtifactName.PRIMARY_STRICT_TRANSFER_RESULTS,
                ReportArtifactName.BASELINE_PAIRED_DIFFERENCE_PLOT,
            ),
        ),
        (
            EvidenceHypothesis.OPERATIONAL_RELEVANCE_OF_SPARSE_SUPPORT,
            (
                ReportArtifactName.SPARSITY_AND_DENSE_RESULTS,
                ReportArtifactName.SPARSITY_UTILITY_EFFICIENCY_FIGURE,
            ),
        ),
        (
            EvidenceHypothesis.TARGET_CONFIRMATION_SAFETY,
            (
                ReportArtifactName.CONFIRMATION_RESULTS,
                ReportArtifactName.CONFIRMATION_SAFETY_COVERAGE_FIGURE,
            ),
        ),
        (
            EvidenceHypothesis.SPARSE_SOLVER_WORK_STRUCTURE_AGREEMENT,
            (ReportArtifactName.SCALABILITY_RESULTS, ReportArtifactName.SCALABILITY_FIGURE),
        ),
    )
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
        _evidence_support_rows(rows),
    )


def _evidence_support_rows(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    recorded: OrderedDict[str, Mapping[ReportColumnName, TableScalar]] = OrderedDict(
        (str(row.get(ReportColumnName.QUESTION)), row) for row in rows
    )
    completed: list[Mapping[ReportColumnName, TableScalar]] = []
    for question in EvidenceHypothesis:
        row = recorded.pop(question.value, None)
        completed.append(_evidence_support_row(question, row))
    completed.extend(recorded.values())
    return tuple(completed)


def _evidence_support_row(
    question: EvidenceHypothesis,
    recorded: Mapping[ReportColumnName, TableScalar] | None,
) -> Mapping[ReportColumnName, TableScalar]:
    completed: OrderedDict[ReportColumnName, TableScalar] = OrderedDict()
    if recorded is not None:
        completed.update(recorded)
    completed[ReportColumnName.QUESTION] = question.value
    if completed.get(ReportColumnName.FINAL_STATE) is None:
        completed[ReportColumnName.FINAL_STATE] = UNAVAILABLE_CELL_TEXT
    artifacts = EVIDENCE_SUPPORT_ARTIFACTS.get(question)
    if artifacts is not None:
        completed[ReportColumnName.SUPPORTING_TABLE] = artifacts[0]
        completed[ReportColumnName.SUPPORTING_FIGURE] = artifacts[1]
    return completed


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
        tuple(
            OrderedDict(
                (
                    *(row.items()),
                    (ReportColumnName.STATE, _boundary_state(row)),
                )
            )
            for row in rows
        ),
    )


def _boundary_state(row: Mapping[ReportColumnName, TableScalar]) -> str:
    abstention = _numeric_cell(row, ReportColumnName.ABSTENTION)
    if abstention is not None and abstention == 1.0:
        return BoundaryReportState.ABSTAINED.value
    if _numeric_cell(row, ReportColumnName.REALIZED_GAIN) is None:
        return BoundaryReportState.UNAVAILABLE.value
    return BoundaryReportState.COMPLETED.value


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


def _numeric_cell(
    row: Mapping[ReportColumnName, TableScalar],
    column: ReportColumnName,
) -> float | None:
    value = row.get(column)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _numeric_text(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _setting_coordinate(value: TableScalar) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return _numeric_text(value)


def _unavailable_series(name: str) -> FigureSeries:
    return FigureSeries(name=ReportSeriesName(name), x=(), y=())


def coupling_gap_factor_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[FigureSeries, ...]:
    structure = active_config().generators.coupling_structure
    compatibility_levels = tuple(level.value for level in structure.compatibility)
    numeric_vocabularies: tuple[tuple[str, tuple[float, ...]], ...] = (
        (
            "response heterogeneity",
            tuple(float(level) for level in structure.response_heterogeneity),
        ),
        ("directed asymmetry", tuple(float(level) for level in structure.directed_asymmetry)),
        ("response sparsity", tuple(float(level) for level in structure.response_sparsity)),
    )
    grouped: OrderedDict[str, list[tuple[float, float]]] = OrderedDict()
    supports: list[tuple[float, float]] = []
    unparsed = 0
    for row in rows:
        condition = row.get(ReportColumnName.CONDITION_OR_PAIR)
        gap = _numeric_cell(row, ReportColumnName.FIXED_ACTION_GAP)
        if not isinstance(condition, str) or gap is None:
            unparsed += 1
            continue
        parts = condition.split("-")
        if len(parts) != 5 or parts[0] not in compatibility_levels:
            unparsed += 1
            continue
        levels: list[tuple[str, float]] = [
            ("compatibility", float(compatibility_levels.index(parts[0])))
        ]
        registered = True
        for (factor, vocabulary), text in zip(numeric_vocabularies, parts[1:4], strict=True):
            observed = _numeric_text(text)
            level = next((value for value in vocabulary if observed == value), None)
            if level is None:
                registered = False
                break
            levels.append((factor, level))
        if not registered:
            unparsed += 1
            continue
        for factor, level in levels:
            grouped.setdefault(f"{factor} | compatibility={parts[0]}", []).append((level, gap))
        support = _numeric_cell(row, ReportColumnName.SUPPORT)
        if support is not None:
            supports.append((support, gap))
    series = tuple(
        FigureSeries(
            name=ReportSeriesName(factor),
            x=tuple(point[0] for point in points),
            y=tuple(point[1] for point in points),
        )
        for factor, points in grouped.items()
    )
    if supports:
        series += (
            FigureSeries(
                name=ReportSeriesName("support budget"),
                x=tuple(point[0] for point in supports),
                y=tuple(point[1] for point in supports),
            ),
        )
    else:
        series += (_unavailable_series("support budget | unavailable in rendered rows"),)
    if unparsed:
        series += (_unavailable_series(f"unparsed coupling conditions | unavailable n={unparsed}"),)
    return series


def coupling_gap_phase_figure(
    series: Sequence[FigureSeries] = (),
    factor_rows: Sequence[Mapping[ReportColumnName, TableScalar]] = (),
) -> EvidenceFigure:
    structure = active_config().generators.coupling_structure
    resolved = coupling_gap_factor_series(factor_rows) if factor_rows else tuple(series)
    return _figure(
        ReportAxisLabel("coupling factor level"),
        ReportAxisLabel("fixed-action rectangularization gap"),
        resolved,
        horizontal_reference_lines=(
            0.0,
            structure.incompatible_fixed_action_gap_strictly_greater_than,
        ),
    )


def sparsity_utility_efficiency_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[FigureSeries, ...]:
    grouped: OrderedDict[str, list[tuple[float, float, float]]] = OrderedDict()
    unavailable: list[str] = []
    for row in rows:
        condition = row.get(ReportColumnName.SUPPORT_OR_DENSE_CONDITION)
        if not isinstance(condition, str) or not condition:
            continue
        runtime = _numeric_cell(row, ReportColumnName.RUNTIME)
        gain = _numeric_cell(row, ReportColumnName.REALIZED_GAIN)
        if runtime is None or gain is None:
            unavailable.append(condition)
            continue
        memory = _numeric_cell(row, ReportColumnName.MEMORY)
        grouped.setdefault(condition, []).append(
            (runtime, gain, memory if memory is not None else 0.0)
        )
    series = tuple(
        FigureSeries(
            name=ReportSeriesName(condition),
            x=tuple(point[0] for point in points),
            y=tuple(point[1] for point in points),
            marker_sizes=tuple(point[2] for point in points),
        )
        for condition, points in grouped.items()
    )
    if not series:
        series = (_unavailable_series("sparsity cells | unavailable in stored rows"),)
    if unavailable:
        series += (_unavailable_series(f"unavailable sparsity conditions | n={len(unavailable)}"),)
    return series


def sparsity_utility_efficiency_figure(
    series: Sequence[FigureSeries] = (),
    sparsity_rows: Sequence[Mapping[ReportColumnName, TableScalar]] = (),
) -> EvidenceFigure:
    resolved = sparsity_utility_efficiency_series(sparsity_rows) if sparsity_rows else tuple(series)
    return _figure(
        ReportAxisLabel("runtime"),
        ReportAxisLabel("realized gain"),
        resolved,
    )


def scalability_trend_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[FigureSeries, ...]:
    grouped: OrderedDict[str, list[tuple[float, float]]] = OrderedDict()
    timeouts: list[tuple[float, float]] = []
    unavailable = 0
    for row in rows:
        method = row.get(ReportColumnName.METHOD)
        work = _numeric_cell(row, ReportColumnName.PREDICTED_WORK)
        runtime = _numeric_cell(row, ReportColumnName.RUNTIME_MEDIAN)
        if not isinstance(method, str) or work is None or runtime is None:
            unavailable += 1
            continue
        grouped.setdefault(method, []).append((work, runtime))
        timeout_count = _numeric_cell(row, ReportColumnName.TIMEOUT)
        if timeout_count is not None and timeout_count > 0.0:
            timeouts.append((work, runtime))
    series = tuple(
        FigureSeries(
            name=ReportSeriesName(method),
            x=tuple(point[0] for point in points),
            y=tuple(point[1] for point in points),
        )
        for method, points in grouped.items()
    )
    if timeouts:
        series += (
            FigureSeries(
                name=ReportSeriesName("timeout limit reached"),
                x=tuple(point[0] for point in timeouts),
                y=tuple(point[1] for point in timeouts),
                marker_sizes=tuple(0.0 for _ in timeouts),
            ),
        )
    if not series:
        series = (_unavailable_series("scalability cells | unavailable in stored rows"),)
    if unavailable:
        series += (_unavailable_series(f"unavailable scalability cells | n={unavailable}"),)
    return series


def scalability_figure(
    series: Sequence[FigureSeries] = (),
    scalability_rows: Sequence[Mapping[ReportColumnName, TableScalar]] = (),
) -> EvidenceFigure:
    resolved = scalability_trend_series(scalability_rows) if scalability_rows else tuple(series)
    return _figure(
        ReportAxisLabel("N_S * sum(n_g^3) (log scale)"),
        ReportAxisLabel("runtime (log scale)"),
        resolved,
        log_x=True,
        log_y=True,
    )


def confirmation_safety_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
    registered_pairs: Sequence[DirectedPairName] = (),
) -> tuple[FigureSeries, ...]:
    recorded: OrderedDict[str, tuple[float, float, float]] = OrderedDict()
    unavailable: list[str] = []
    for row in rows:
        pair = row.get(ReportColumnName.PAIR)
        if not isinstance(pair, str) or not pair:
            continue
        coverage = _numeric_cell(row, ReportColumnName.COVERAGE)
        harm = _numeric_cell(row, ReportColumnName.HARMFUL_ACCEPTED_RATE)
        no_confirm_harm = _numeric_cell(row, ReportColumnName.NO_CONFIRM_HARMFUL_RATE)
        if coverage is None or harm is None or no_confirm_harm is None:
            unavailable.append(pair)
            continue
        recorded[pair] = (coverage, no_confirm_harm, harm)
    series = tuple(
        FigureSeries(
            name=ReportSeriesName(pair),
            x=(coverage,),
            y=(no_confirm_harm,),
            arrow_x=(coverage,),
            arrow_y=(harm,),
        )
        for pair, (coverage, no_confirm_harm, harm) in recorded.items()
    )
    for pair in registered_pairs:
        if pair not in recorded:
            series += (
                _unavailable_series(
                    f"{pair} | no proposal state: coverage and safety rates undefined"
                ),
            )
    for pair in unavailable:
        series += (_unavailable_series(f"{pair} | confirmation rates unavailable in stored rows"),)
    return series


def confirmation_safety_coverage_figure(
    series: Sequence[FigureSeries] = (),
    confirmation_rows: Sequence[Mapping[ReportColumnName, TableScalar]] = (),
    registered_pairs: Sequence[DirectedPairName] = (),
) -> EvidenceFigure:
    resolved = (
        confirmation_safety_series(confirmation_rows, registered_pairs)
        if confirmation_rows
        else tuple(series)
    )
    return _figure(
        ReportAxisLabel("confirmation coverage"),
        ReportAxisLabel("harmful accepted rate"),
        resolved,
    )


def failure_boundary_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[FigureSeries, ...]:
    grouped: OrderedDict[str, list[tuple[float, float]]] = OrderedDict()
    unavailable: list[str] = []
    for row in rows:
        dimension = row.get(ReportColumnName.BOUNDARY_DIMENSION)
        if not isinstance(dimension, str) or not dimension:
            continue
        gain = _numeric_cell(row, ReportColumnName.REALIZED_GAIN)
        if gain is None:
            unavailable.append(dimension)
            continue
        x_value = _setting_coordinate(row.get(ReportColumnName.SETTING))
        if x_value is None:
            unavailable.append(dimension)
            continue
        grouped.setdefault(dimension, []).append((x_value, gain))
    series = tuple(
        FigureSeries(
            name=ReportSeriesName(dimension),
            x=tuple(point[0] for point in points),
            y=tuple(point[1] for point in points),
        )
        for dimension, points in grouped.items()
    )
    if unavailable:
        series += (
            _unavailable_series(
                f"abstained or ineligible boundary settings | unavailable n={len(unavailable)}"
            ),
        )
    if not series:
        series = (_unavailable_series("boundary settings | unavailable in stored rows"),)
    return series


def failure_boundary_figure(
    series: Sequence[FigureSeries] = (),
    boundary_rows: Sequence[Mapping[ReportColumnName, TableScalar]] = (),
) -> EvidenceFigure:
    resolved = failure_boundary_series(boundary_rows) if boundary_rows else tuple(series)
    return _figure(
        ReportAxisLabel("boundary setting"),
        ReportAxisLabel("certified value / realized gain"),
        resolved,
        separate_panels=True,
    )


def predicted_vs_realized_series(
    metric_records: Sequence[MetricRecord] = (),
) -> tuple[FigureSeries, ...]:
    correlations = _stored_correlations(metric_records)
    realized: OrderedDict[tuple[DirectedPairName, int], float] = OrderedDict()
    predicted: set[DirectedPairName] = set()
    points: OrderedDict[DirectedPairName, list[tuple[float, float]]] = OrderedDict()
    for record in metric_records:
        if not record.valid or record.metric_value is None:
            continue
        if record.metric_name == MetricId.RELATIVE_MACRO_CE_GAIN:
            realized[(record.pair, record.seed)] = float(record.metric_value)
    for record in metric_records:
        if not record.valid or record.metric_value is None:
            continue
        if record.metric_name != MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE:
            continue
        predicted.add(record.pair)
        gain = realized.get((record.pair, record.seed))
        if gain is None:
            continue
        points.setdefault(record.pair, []).append((float(record.metric_value), gain))
    rendered: list[FigureSeries] = []
    for pair, pair_points in points.items():
        correlation = correlations.get(pair)
        if correlation is None:
            name = f"{pair} | Spearman unavailable; n={len(pair_points)}"
        else:
            name = f"{pair} | Spearman rho={correlation:.3f}; n={len(pair_points)}"
        rendered.append(
            FigureSeries(
                name=ReportSeriesName(name),
                x=tuple(point[0] for point in pair_points),
                y=tuple(point[1] for point in pair_points),
            )
        )
    unmatched = len(predicted - set(points))
    if unmatched or not rendered:
        rendered.append(
            _unavailable_series(f"predicted-to-realized pairs | unavailable n={unmatched}")
        )
    return tuple(rendered)


def _stored_correlations(
    metric_records: Sequence[MetricRecord],
) -> Mapping[DirectedPairName, float]:
    correlations: OrderedDict[DirectedPairName, float] = OrderedDict()
    for record in metric_records:
        if (
            record.metric_name == MetricId.PREDICTED_REALIZED_SPEARMAN
            and record.valid
            and record.metric_value is not None
        ):
            correlations[record.pair] = float(record.metric_value)
    return correlations


def predicted_vs_realized_transfer_figure(
    series: Sequence[FigureSeries] = (),
    metric_records: Sequence[MetricRecord] = (),
) -> EvidenceFigure:
    resolved = predicted_vs_realized_series(metric_records) if metric_records else tuple(series)
    return _figure(
        ReportAxisLabel("certified robust predicted value"),
        ReportAxisLabel("TEST relative macro-CE gain"),
        resolved,
    )


def map_value_bound_series(
    metric_records: Sequence[MetricRecord] = (),
) -> tuple[FigureSeries, ...]:
    bounds: OrderedDict[tuple[DirectedPairName, EvaluationConditionName, int], float] = (
        OrderedDict()
    )
    values: OrderedDict[tuple[DirectedPairName, EvaluationConditionName, int], float] = (
        OrderedDict()
    )
    for record in metric_records:
        if not record.valid or record.metric_value is None:
            continue
        key = (record.pair, record.condition, record.seed)
        if record.metric_name == MetricId.ORBIT_RADIUS_MAP_BOUND:
            bounds[key] = float(record.metric_value)
        if record.metric_name == MetricId.EXACT_MAP_ACTION_VALUE:
            values[key] = float(record.metric_value)
    xs: list[float] = []
    ys: list[float] = []
    for key, bound in bounds.items():
        value = values.get(key)
        if value is None:
            continue
        xs.append(bound)
        ys.append(value)
    unavailable = len(set(bounds) | set(values)) - len(xs)
    series: tuple[FigureSeries, ...] = ()
    if xs:
        series += (
            FigureSeries(
                name=ReportSeriesName(MetricId.EXACT_MAP_ACTION_VALUE.value),
                x=tuple(xs),
                y=tuple(ys),
            ),
        )
    if unavailable or not series:
        series += (
            _unavailable_series(f"ineligible map-value bound cells | unavailable n={unavailable}"),
        )
    return series


def map_value_bound_figure(
    series: Sequence[FigureSeries] = (),
    metric_records: Sequence[MetricRecord] = (),
) -> EvidenceFigure:
    resolved = map_value_bound_series(metric_records) if metric_records else tuple(series)
    return _figure(
        ReportAxisLabel("orbit-radius bound"),
        ReportAxisLabel("exact map action value"),
        resolved,
        draw_unit_diagonal=True,
    )


def semantic_sufficiency_frontier_series(
    series: Sequence[FigureSeries],
) -> tuple[FigureSeries, ...]:
    return tuple(
        replace(item, name=ReportSeriesName(f"{item.name} | oracle/diagnostic-only"))
        if _diagnostic_only(item)
        else item
        for item in series
    )


def _diagnostic_only(series: FigureSeries) -> bool:
    if TransferMethod.EXACT_MAP_ORACLE.value in str(series.name):
        return True
    return any(value == 0.0 for value in series.x)


def semantic_sufficiency_frontier_figure(
    series: Sequence[FigureSeries] = (),
) -> EvidenceFigure:
    return _figure(
        ReportAxisLabel("log|orbit|"),
        ReportAxisLabel("realized gain"),
        semantic_sufficiency_frontier_series(series),
    )
