from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import fedorbit.infrastructure.evidence as evidence_module
import fedorbit.reporting as reporting_module
from fedorbit.analysis.records import (
    ComparisonDecision,
    MetricDirection,
    MetricRecord,
    PairedComparisonRecord,
)
from fedorbit.config.loading import active_config
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.evidence import (
    EXPORT_LEDGER_KEY,
    UNAVAILABLE_CELL_TEXT,
    EvidenceFigure,
    EvidenceTable,
    FigureSeries,
    TableScalar,
    VerifiedEvidenceWriter,
)
from fedorbit.infrastructure.manifests import DatasetManifest
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.reporting import (
    ablation_results_table,
    baseline_paired_difference_plot,
    confirmation_results_table,
    confirmation_safety_coverage_figure,
    coupling_gap_factor_series,
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
    ArtifactIdentifier,
    ClientRole,
    ContrastName,
    DatasetId,
    DatasetModality,
    DirectedPairName,
    EvaluationCondition,
    EvaluationConditionKind,
    EvaluationConditionName,
    EvidenceHypothesis,
    ExperimentName,
    MetricId,
    MetricUnit,
    MultiplicityFamily,
    ReportArtifactName,
    ReportColumnName,
    ReportSeriesName,
    Sha256Digest,
    SupportSize,
    TransferMethod,
)

FINGERPRINT = Sha256Digest("a" * 64)
PAIR = DirectedPairName("edge_iiotset_network -> ton_iot_network")
OTHER_PAIR = DirectedPairName("ton_iot_network -> edge_iiotset_network")
UNSEEN_PAIR = DirectedPairName("ton_iot_windows10_host -> ton_iot_network")
COUPLING_CONDITION = "jointly_realizable-0.5-0.0-0.25-2x2"


def _metric_record(
    metric_name: MetricId,
    metric_value: float,
    metric_unit: MetricUnit = MetricUnit.NATS,
    pair: DirectedPairName = PAIR,
    method: TransferMethod = TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
    condition: EvaluationConditionName = PRINCIPAL_EVALUATION_CONDITION.name,
) -> MetricRecord:
    return MetricRecord(
        experiment=ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        pair=pair,
        method=method,
        condition=condition,
        seed=1103,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        direction=MetricDirection.LOWER_IS_BETTER,
        evaluation_class_set_sha256=FINGERPRINT,
        input_artifact_ids=(ArtifactIdentifier("upstream-metric"),),
        dependency_fingerprint_sha256=FINGERPRINT,
        valid=True,
        invalid_reason=None,
    )


def _comparison(
    mean_difference: float,
    bca_ci_low: float,
    bca_ci_high: float,
    raw_p: float,
    holm_p: float,
    pair: DirectedPairName = PAIR,
) -> PairedComparisonRecord:
    return PairedComparisonRecord(
        contrast_name=ContrastName(f"FedORBIT vs Local-Only: {pair}"),
        family=MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
        pair=pair,
        method_a=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        method_b=TransferMethod.LOCAL_ONLY,
        metric=MetricId.MACRO_CROSS_ENTROPY,
        paired_seed_count=10,
        mean_difference=mean_difference,
        median_difference=mean_difference,
        bca_ci_low=bca_ci_low,
        bca_ci_high=bca_ci_high,
        raw_p=raw_p,
        holm_p=holm_p,
        materiality_threshold=None,
        equivalence_margin_low=None,
        equivalence_margin_high=None,
        input_metric_artifact_ids=(ArtifactIdentifier("upstream-metric"),),
        dependency_fingerprint_sha256=FINGERPRINT,
        decision=ComparisonDecision.SUPERIOR,
    )


def _series(name: str = "s=2") -> tuple[FigureSeries, ...]:
    return (
        FigureSeries(
            name=ReportSeriesName(name),
            x=(1.0, 2.0),
            y=(3.0, 4.0),
            y_low=(2.5, 3.5),
            y_high=(3.5, 4.5),
            marker_sizes=(10.0, 20.0),
        ),
    )


def _dataset_manifest() -> DatasetManifest:
    return DatasetManifest.model_validate(
        {
            "dataset": "edge_iiotset_network",
            "component": "network",
            "raw_files": ("ML-EdgeIIoT-dataset.csv",),
            "raw_sha256": "a" * 64,
            "raw_counts": {"ML-EdgeIIoT-dataset.csv": 100},
            "schema": "1.0",
            "adapter_feature_order": ("tcp.ack",),
            "adapter_feature_roles": {"tcp.ack": "behavioral_numeric"},
            "accepted_schema_aliases": (),
            "adapter_adaptations": (),
            "timestamp_field": "frame.time",
            "timestamp_range": (1577836800.0, 1577923200.0),
            "duplicate_counts": {"total": 0},
            "conflicting_duplicate_counts": {"total": 0},
            "local_class_counts": {"normal": 50, "ddos_tcp": 50},
            "transfer_candidate_counts": {"DDoS": 50},
            "feature_quality": {
                "dropped_feature_count": 0,
                "candidate_count_before_filtering": 1,
                "client_invalid": False,
            },
            "preprocessing_state": "materialized",
            "dependency_fingerprint_sha256": "b" * 64,
            "producer_code_sha256": "c" * 64,
        }
    )


def _generic_rows() -> tuple[OrderedDict[ReportColumnName, TableScalar], ...]:
    return (
        OrderedDict(
            (
                (ReportColumnName.ABLATION, "FedORBIT Without Confirmation"),
                (ReportColumnName.PAIR, PAIR),
                (ReportColumnName.METHOD, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value),
                (ReportColumnName.REALIZED_GAIN, 0.123456),
                (ReportColumnName.DIFFERENCE_VS_FULL, -0.05),
                (ReportColumnName.EQUIVALENCE, 0.0125),
                (ReportColumnName.RETAINED_GAIN, 0.5),
                (ReportColumnName.CONFIRMATION_SAFETY, 1.0),
                (ReportColumnName.RUNTIME_MEDIAN, 1.23456),
                (ReportColumnName.RSS, 64.25),
                (ReportColumnName.LAP_CALLS, 7.0),
                (ReportColumnName.TIMEOUTS, 1.0),
                (ReportColumnName.RAW_P, 0.00005),
                (ReportColumnName.SUPPORT, 2),
                (ReportColumnName.K, 6),
                (ReportColumnName.COVERAGE, 0.0),
                (ReportColumnName.HARMFUL_ACCEPTED_RATE, 0.1),
                (ReportColumnName.NO_CONFIRM_HARMFUL_RATE, 0.25),
            )
        ),
        OrderedDict(
            (
                (ReportColumnName.ABLATION, "Coupling-Destroyed FedORBIT"),
                (ReportColumnName.PAIR, OTHER_PAIR),
                (ReportColumnName.METHOD, TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK.value),
                (ReportColumnName.REALIZED_GAIN, None),
                (ReportColumnName.DIFFERENCE_VS_FULL, None),
                (ReportColumnName.EQUIVALENCE, None),
                (ReportColumnName.RETAINED_GAIN, None),
                (ReportColumnName.CONFIRMATION_SAFETY, None),
                (ReportColumnName.SUPPORT_OR_DENSE_CONDITION, "dense CCP"),
                (ReportColumnName.ABSTENTION, 1.0),
                (ReportColumnName.STATE, "Completed"),
                (ReportColumnName.BOUNDARY_DIMENSION, "semantic-sufficiency-partition"),
                (ReportColumnName.SETTING, "partition-unavailable"),
                (ReportColumnName.RUNTIME_MEDIAN, None),
                (ReportColumnName.TIMEOUT, 0.0),
                (ReportColumnName.PREDICTED_WORK, None),
                (ReportColumnName.HARMFUL_ACCEPTED_RATE, None),
                (ReportColumnName.NO_CONFIRM_HARMFUL_RATE, None),
            )
        ),
    )


def _registered_tables() -> OrderedDict[ReportArtifactName, EvidenceTable]:
    dataset = _dataset_manifest()
    dataset_id = DatasetId("edge_iiotset_network")
    metric_records = (
        _metric_record(MetricId.MACRO_CROSS_ENTROPY, 1.234567),
        _metric_record(MetricId.MACRO_F1, 0.987654, MetricUnit.FRACTION),
        _metric_record(MetricId.BALANCED_ACCURACY, 0.5, MetricUnit.FRACTION),
        _metric_record(MetricId.COVERAGE_CONFIRM, 0.75, MetricUnit.FRACTION),
    )
    comparison_records = (_comparison(0.123456, 0.1, 0.15, 0.00005, 0.02),)
    generic_rows = _generic_rows()
    return OrderedDict(
        (
            (
                ReportArtifactName.NUMERICAL_CONSTANTS_AND_SEEDS,
                numerical_constants_and_seeds_table(),
            ),
            (ReportArtifactName.EXPERIMENT_MATRIX, experiment_matrix_table(generic_rows)),
            (
                ReportArtifactName.DATASET_AND_CLIENT_PROTOCOL,
                dataset_and_client_protocol_table(
                    (dataset,),
                    modality_by_dataset={dataset_id: DatasetModality.NETWORK},
                    role_by_dataset={dataset_id: ClientRole.PRIMARY},
                    excluded_class_counts={dataset_id: 0},
                ),
            ),
            (
                ReportArtifactName.INFORMATION_RESOURCE_MATRIX,
                information_resource_matrix_table(),
            ),
            (
                ReportArtifactName.TRANSFER_ONTOLOGY_AND_NULL_PADDING,
                transfer_ontology_and_null_padding_table(generic_rows),
            ),
            (
                ReportArtifactName.MODEL_AND_TRAINING_PROTOCOL,
                model_and_training_protocol_table(generic_rows),
            ),
            (ReportArtifactName.EVIDENCE_STATUS, evidence_status_table(generic_rows)),
            (
                ReportArtifactName.PRIMARY_STRICT_TRANSFER_RESULTS,
                primary_strict_transfer_results_table(metric_records, comparison_records),
            ),
            (
                ReportArtifactName.COUPLING_MECHANISM_RESULTS,
                coupling_mechanism_results_table(generic_rows),
            ),
            (
                ReportArtifactName.EXACT_SOLVER_RESULTS,
                exact_solver_results_table(generic_rows),
            ),
            (ReportArtifactName.ABLATION_RESULTS, ablation_results_table(generic_rows)),
            (
                ReportArtifactName.SPARSITY_AND_DENSE_RESULTS,
                sparsity_and_dense_results_table(generic_rows),
            ),
            (
                ReportArtifactName.CONFIRMATION_RESULTS,
                confirmation_results_table(generic_rows, registered_pairs=(PAIR, OTHER_PAIR)),
            ),
            (
                ReportArtifactName.GENERALIZATION_RESULTS,
                generalization_results_table(generic_rows),
            ),
            (
                ReportArtifactName.FAILURE_BOUNDARY_RESULTS,
                failure_boundary_results_table(generic_rows),
            ),
            (
                ReportArtifactName.SCALABILITY_RESULTS,
                scalability_results_table(generic_rows),
            ),
        )
    )


def _coupling_rows() -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    return (
        OrderedDict(
            (
                (ReportColumnName.CONDITION_OR_PAIR, COUPLING_CONDITION),
                (ReportColumnName.FIXED_ACTION_GAP, 0.25),
                (ReportColumnName.ROBUST_COUPLING_GAP, 0.5),
                (ReportColumnName.SUPPORT, 2),
            )
        ),
        OrderedDict(
            (
                (ReportColumnName.CONDITION_OR_PAIR, "incompatible-1.0-0.5-0.5-2x3"),
                (ReportColumnName.FIXED_ACTION_GAP, 0.75),
                (ReportColumnName.ROBUST_COUPLING_GAP, 0.9),
            )
        ),
        OrderedDict(
            (
                (ReportColumnName.CONDITION_OR_PAIR, OTHER_PAIR),
                (ReportColumnName.FIXED_ACTION_GAP, 0.1),
                (ReportColumnName.ROBUST_COUPLING_GAP, 0.2),
            )
        ),
        OrderedDict(((ReportColumnName.CONDITION_OR_PAIR, COUPLING_CONDITION),)),
    )


def _sparsity_rows() -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    return (
        OrderedDict(
            (
                (ReportColumnName.SUPPORT_OR_DENSE_CONDITION, "exact sparse s=1"),
                (ReportColumnName.PAIR, PAIR),
                (ReportColumnName.REALIZED_GAIN, 0.1),
                (ReportColumnName.RUNTIME, 1.0),
                (ReportColumnName.MEMORY, 32.0),
            )
        ),
        OrderedDict(
            (
                (ReportColumnName.SUPPORT_OR_DENSE_CONDITION, "exact sparse s=2"),
                (ReportColumnName.PAIR, PAIR),
                (ReportColumnName.REALIZED_GAIN, 0.2),
                (ReportColumnName.RUNTIME, 2.0),
                (ReportColumnName.MEMORY, 48.0),
            )
        ),
        OrderedDict(
            (
                (ReportColumnName.SUPPORT_OR_DENSE_CONDITION, "dense CCP"),
                (ReportColumnName.PAIR, PAIR),
                (ReportColumnName.REALIZED_GAIN, None),
                (ReportColumnName.RUNTIME, 4.0),
                (ReportColumnName.MEMORY, 96.0),
            )
        ),
    )


def _scalability_rows() -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    return (
        OrderedDict(
            (
                (ReportColumnName.METHOD, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value),
                (ReportColumnName.PREDICTED_WORK, 100.0),
                (ReportColumnName.RUNTIME_MEDIAN, 1.5),
                (ReportColumnName.TIMEOUT, 0.0),
            )
        ),
        OrderedDict(
            (
                (ReportColumnName.METHOD, TransferMethod.GENERIC_EXACT_QAP.value),
                (ReportColumnName.PREDICTED_WORK, 200.0),
                (ReportColumnName.RUNTIME_MEDIAN, 9.5),
                (ReportColumnName.TIMEOUT, 2.0),
            )
        ),
        OrderedDict(
            (
                (ReportColumnName.METHOD, TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK.value),
                (ReportColumnName.PREDICTED_WORK, None),
                (ReportColumnName.RUNTIME_MEDIAN, None),
                (ReportColumnName.TIMEOUT, 0.0),
            )
        ),
    )


def _boundary_rows() -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    return (
        OrderedDict(
            (
                (ReportColumnName.BOUNDARY_DIMENSION, "response-scale"),
                (ReportColumnName.SETTING, "0.5"),
                (ReportColumnName.PAIR, PAIR),
                (ReportColumnName.REALIZED_GAIN, 0.1),
            )
        ),
        OrderedDict(
            (
                (ReportColumnName.BOUNDARY_DIMENSION, "support-budget"),
                (ReportColumnName.SETTING, "3"),
                (ReportColumnName.PAIR, PAIR),
                (ReportColumnName.REALIZED_GAIN, None),
                (ReportColumnName.ABSTENTION, 1.0),
            )
        ),
        OrderedDict(
            (
                (ReportColumnName.BOUNDARY_DIMENSION, "semantic-sufficiency-partition"),
                (ReportColumnName.SETTING, "partition-unavailable"),
                (ReportColumnName.PAIR, PAIR),
                (ReportColumnName.REALIZED_GAIN, 0.2),
            )
        ),
    )


def _predicted_realized_records() -> tuple[MetricRecord, ...]:
    return (
        _metric_record(MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE, 0.1),
        _metric_record(MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE, 0.2),
        _metric_record(MetricId.RELATIVE_MACRO_CE_GAIN, 0.05),
        _metric_record(MetricId.RELATIVE_MACRO_CE_GAIN, 0.15),
        _metric_record(MetricId.PREDICTED_REALIZED_SPEARMAN, 0.999),
        _metric_record(
            MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE,
            0.3,
            MetricUnit.SCORE,
            OTHER_PAIR,
        ),
    )


def _map_value_records() -> tuple[MetricRecord, ...]:
    return (
        _metric_record(MetricId.ORBIT_RADIUS_MAP_BOUND, 0.4),
        _metric_record(MetricId.EXACT_MAP_ACTION_VALUE, 0.25),
        _metric_record(MetricId.ORBIT_RADIUS_MAP_BOUND, 0.6, MetricUnit.SCORE, OTHER_PAIR),
    )


def _frontier_series() -> tuple[FigureSeries, ...]:
    return (
        FigureSeries(
            name=ReportSeriesName(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value),
            x=(1.0, 2.0),
            y=(0.1, 0.2),
        ),
        FigureSeries(
            name=ReportSeriesName(TransferMethod.EXACT_MAP_ORACLE.value),
            x=(1.0,),
            y=(0.3,),
        ),
        FigureSeries(
            name=ReportSeriesName(TransferMethod.GENERIC_EXACT_QAP.value),
            x=(0.0,),
            y=(0.4,),
        ),
    )


def _registered_figures() -> OrderedDict[ReportArtifactName, EvidenceFigure]:
    series = _series()
    records = _predicted_realized_records()
    return OrderedDict(
        (
            (
                ReportArtifactName.REAL_TRANSFER_GAIN_FOREST_PLOT,
                real_transfer_gain_forest_plot(series, (ReportSeriesName(PAIR),)),
            ),
            (
                ReportArtifactName.BASELINE_PAIRED_DIFFERENCE_PLOT,
                baseline_paired_difference_plot(series),
            ),
            (
                ReportArtifactName.COUPLING_GAP_PHASE_FIGURE,
                coupling_gap_phase_figure(factor_rows=_coupling_rows()),
            ),
            (
                ReportArtifactName.PREDICTED_VS_REALIZED_TRANSFER_FIGURE,
                predicted_vs_realized_transfer_figure(metric_records=records),
            ),
            (
                ReportArtifactName.SPARSITY_UTILITY_EFFICIENCY_FIGURE,
                sparsity_utility_efficiency_figure(sparsity_rows=_sparsity_rows()),
            ),
            (
                ReportArtifactName.CONFIRMATION_SAFETY_COVERAGE_FIGURE,
                confirmation_safety_coverage_figure(
                    confirmation_rows=_generic_rows(),
                    registered_pairs=(PAIR, UNSEEN_PAIR),
                ),
            ),
            (
                ReportArtifactName.SEMANTIC_SUFFICIENCY_FRONTIER_FIGURE,
                semantic_sufficiency_frontier_figure(_frontier_series()),
            ),
            (
                ReportArtifactName.FAILURE_BOUNDARY_FIGURE,
                failure_boundary_figure(boundary_rows=_boundary_rows()),
            ),
            (
                ReportArtifactName.SCALABILITY_FIGURE,
                scalability_figure(scalability_rows=_scalability_rows()),
            ),
            (
                ReportArtifactName.MAP_VALUE_BOUND_FIGURE,
                map_value_bound_figure(metric_records=_map_value_records()),
            ),
        )
    )


def _cell(table: EvidenceTable, column: ReportColumnName) -> object:
    return table.rows[0][table.columns.index(column)]


def _method_cell(table: EvidenceTable, column: ReportColumnName, method: TransferMethod) -> object:
    method_index = table.columns.index(ReportColumnName.METHOD)
    matches = [row for row in table.rows if row[method_index] == method.value]
    assert matches, f"missing rendered method row: {method.value}"
    return matches[0][table.columns.index(column)]


def test_information_resource_matrix_table_covers_registered_methods_in_order() -> None:
    table = information_resource_matrix_table()
    assert table.columns == (
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
    methods = tuple(row[0] for row in table.rows)
    assert methods == (
        TransferMethod.LOCAL_ONLY.value,
        TransferMethod.LOCAL_SIR.value,
        TransferMethod.MATCHED_RESOURCE_RECTANGULAR.value,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT.value,
        TransferMethod.GENERIC_EXACT_QAP.value,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value,
        TransferMethod.EXACT_MAP_ORACLE.value,
    )


def test_information_resource_matrix_table_fedorbit_never_identifies_exact_map() -> None:
    table = information_resource_matrix_table()
    rows_by_method = {row[0]: row for row in table.rows}
    columns = table.columns
    exact_map_index = columns.index("exact_map")
    fedorbit_row = rows_by_method[TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value]
    assert fedorbit_row[exact_map_index] is False
    qap_row = rows_by_method[TransferMethod.POINT_CORRESPONDENCE_COMMITMENT.value]
    assert qap_row[exact_map_index] is True


def test_information_resource_matrix_table_local_only_touches_nothing() -> None:
    table = information_resource_matrix_table()
    rows_by_method = {row[0]: row for row in table.rows}
    local_only_row = rows_by_method[TransferMethod.LOCAL_ONLY.value]
    assert all(value is False for value in local_only_row[1:-1])
    assert local_only_row[-1] is True


def test_information_resource_matrix_table_exact_map_oracle_is_strict_incompatible() -> None:
    table = information_resource_matrix_table()
    rows_by_method = {row[0]: row for row in table.rows}
    oracle_row = rows_by_method[TransferMethod.EXACT_MAP_ORACLE.value]
    assert oracle_row[-1] is False


def test_table_cells_use_registered_reporting_precision() -> None:
    ablation = ablation_results_table(_generic_rows())
    assert _cell(ablation, ReportColumnName.REALIZED_GAIN) == "0.1235"
    assert _cell(ablation, ReportColumnName.DIFFERENCE_VS_FULL) == "-0.0500"
    assert _cell(ablation, ReportColumnName.EQUIVALENCE) == "0.0125"
    assert _cell(ablation, ReportColumnName.RETAINED_GAIN) == "0.5000"
    scalability = scalability_results_table(_generic_rows())
    assert _cell(scalability, ReportColumnName.RUNTIME_MEDIAN) == "1.235"
    assert _cell(scalability, ReportColumnName.RSS) == "64.2"
    assert _cell(scalability, ReportColumnName.LAP_CALLS) == "7"


def test_p_value_cells_use_the_registered_less_than_literal() -> None:
    generalization = generalization_results_table(_generic_rows())
    assert _cell(generalization, ReportColumnName.RAW_P) == "<0.0001"


def _question_row(
    table: EvidenceTable,
    question: EvidenceHypothesis,
    column: ReportColumnName,
) -> object:
    question_index = table.columns.index(ReportColumnName.QUESTION)
    matches = [row for row in table.rows if row[question_index] == question.value]
    assert matches, f"missing claim-support row: {question.value}"
    return matches[0][table.columns.index(column)]


def test_claim_support_table_renders_every_registered_question() -> None:
    table = evidence_status_table(
        (
            OrderedDict(
                (
                    (
                        ReportColumnName.QUESTION,
                        EvidenceHypothesis.STRICT_CROSS_TELEMETRY_TRANSFER_UTILITY.value,
                    ),
                    (ReportColumnName.FINAL_STATE, "Supported"),
                    (ReportColumnName.SCOPE, "registered primary evidence"),
                )
            ),
            OrderedDict(
                (
                    (ReportColumnName.QUESTION, "does transfer help"),
                    (ReportColumnName.FINAL_STATE, "Supported"),
                )
            ),
        )
    )
    recorded = tuple(row[table.columns.index(ReportColumnName.QUESTION)] for row in table.rows)
    assert recorded[: len(EvidenceHypothesis)] == tuple(
        question.value for question in EvidenceHypothesis
    )
    assert recorded[-1] == "does transfer help"
    utility = EvidenceHypothesis.STRICT_CROSS_TELEMETRY_TRANSFER_UTILITY
    assert _question_row(table, utility, ReportColumnName.FINAL_STATE) == "Supported"
    assert _question_row(table, utility, ReportColumnName.SCOPE) == "registered primary evidence"
    assert (
        _question_row(table, utility, ReportColumnName.SUPPORTING_TABLE)
        == ReportArtifactName.PRIMARY_STRICT_TRANSFER_RESULTS
    )
    assert (
        _question_row(table, utility, ReportColumnName.SUPPORTING_FIGURE)
        == ReportArtifactName.REAL_TRANSFER_GAIN_FOREST_PLOT
    )
    sparsity = EvidenceHypothesis.OPERATIONAL_RELEVANCE_OF_SPARSE_SUPPORT
    assert _question_row(table, sparsity, ReportColumnName.FINAL_STATE) == "NA"
    assert _question_row(table, sparsity, ReportColumnName.MATERIALITY_RESULT) is None
    assert (
        _question_row(table, sparsity, ReportColumnName.SUPPORTING_FIGURE)
        == ReportArtifactName.SPARSITY_UTILITY_EFFICIENCY_FIGURE
    )
    assert _question_row(table, sparsity, ReportColumnName.FORBIDDEN_WORDING) is None


def test_primary_strict_transfer_table_consumes_stored_gain_and_statistics() -> None:
    metric_records = (
        _metric_record(MetricId.MACRO_CROSS_ENTROPY, 1.234567),
        _metric_record(MetricId.MACRO_F1, 0.987654, MetricUnit.FRACTION),
        _metric_record(MetricId.BALANCED_ACCURACY, 0.5, MetricUnit.FRACTION),
        _metric_record(MetricId.COVERAGE_CONFIRM, 0.75, MetricUnit.FRACTION),
        _metric_record(
            MetricId.MACRO_CROSS_ENTROPY, 0.25, MetricUnit.NATS, PAIR, TransferMethod.LOCAL_ONLY
        ),
    )
    comparison_records = (_comparison(0.123456, 0.1, 0.15, 0.00005, 0.02),)
    table = primary_strict_transfer_results_table(metric_records, comparison_records)
    assert ReportColumnName.CI in table.columns
    method = TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
    assert _method_cell(table, ReportColumnName.GAIN_VS_LOCAL, method) == "0.1235"
    assert _method_cell(table, ReportColumnName.CI, method) == "0.1235 [0.1000, 0.1500]"
    assert _method_cell(table, ReportColumnName.RAW_P, method) == "<0.0001"
    assert _method_cell(table, ReportColumnName.HOLM_P, method) == "0.0200"
    assert _method_cell(table, ReportColumnName.VALID_SEEDS, method) == "10"
    assert _method_cell(table, ReportColumnName.TEST_MACRO_CE, method) == "1.2346"
    assert _method_cell(table, ReportColumnName.MACRO_F1, method) == "0.9877"
    assert _method_cell(table, ReportColumnName.BALANCED_ACCURACY, method) == "0.5000"
    assert _method_cell(table, ReportColumnName.CONFIRMATION_COVERAGE, method) == "0.7500"


def test_confirmation_table_reports_registered_pairs_without_proposals() -> None:
    other_pair = DirectedPairName("ton_iot_network -> edge_iiotset_network")
    rows: tuple[Mapping[ReportColumnName, TableScalar], ...] = (
        OrderedDict(
            (
                (ReportColumnName.PAIR, PAIR),
                (ReportColumnName.PROPOSALS, 3),
                (ReportColumnName.ACCEPTED, 1),
                (ReportColumnName.COVERAGE, 0.3333),
            )
        ),
    )
    table = confirmation_results_table(rows, registered_pairs=(PAIR, other_pair))
    assert len(table.rows) == 2
    recorded = tuple(row[table.columns.index(ReportColumnName.PAIR)] for row in table.rows)
    assert recorded == (PAIR, other_pair)
    absent = table.rows[1]
    assert absent[table.columns.index(ReportColumnName.PROPOSALS)] == "0"
    assert absent[table.columns.index(ReportColumnName.ACCEPTED)] == "0"
    assert absent[table.columns.index(ReportColumnName.COVERAGE)] is None
    assert absent[table.columns.index(ReportColumnName.HARMFUL_ACCEPTED_RATE)] is None


def test_confirmation_table_preserves_a_legitimate_zero_coverage() -> None:
    rows: tuple[Mapping[ReportColumnName, TableScalar], ...] = (
        OrderedDict(
            (
                (ReportColumnName.PAIR, PAIR),
                (ReportColumnName.PROPOSALS, 3),
                (ReportColumnName.ACCEPTED, 0),
                (ReportColumnName.COVERAGE, 0.0),
            )
        ),
    )
    table = confirmation_results_table(rows, registered_pairs=(PAIR,))
    assert _cell(table, ReportColumnName.COVERAGE) == "0.0000"
    assert _cell(table, ReportColumnName.ACCEPTED) == "0"


COMPLETE_BY_CONSTRUCTION_TABLES = frozenset(
    {
        ReportArtifactName.NUMERICAL_CONSTANTS_AND_SEEDS,
        ReportArtifactName.DATASET_AND_CLIENT_PROTOCOL,
        ReportArtifactName.INFORMATION_RESOURCE_MATRIX,
    }
)

UNAVAILABLE_MARKED_FIGURES = frozenset(
    {
        ReportArtifactName.COUPLING_GAP_PHASE_FIGURE,
        ReportArtifactName.PREDICTED_VS_REALIZED_TRANSFER_FIGURE,
        ReportArtifactName.SPARSITY_UTILITY_EFFICIENCY_FIGURE,
        ReportArtifactName.CONFIRMATION_SAFETY_COVERAGE_FIGURE,
        ReportArtifactName.FAILURE_BOUNDARY_FIGURE,
        ReportArtifactName.SCALABILITY_FIGURE,
        ReportArtifactName.MAP_VALUE_BOUND_FIGURE,
    }
)


def test_every_registered_manuscript_table_and_figure_writes_canonical_exports(
    tmp_path: Path,
) -> None:
    layout = build_layout(tmp_path)
    writer = VerifiedEvidenceWriter(ArtifactStore(layout.execution_root), layout)
    tables = _registered_tables()
    figures = _registered_figures()
    assert set(tables) | set(figures) == set(ReportArtifactName)
    for name, table in tables.items():
        destination = writer.write_project_evidence_table(table, name)
        assert destination.name == f"{name}.csv"
        initial = destination.read_bytes()
        assert writer.write_project_evidence_table(table, name).read_bytes() == initial
        assert initial.endswith(b"\n")
        if name not in COMPLETE_BY_CONSTRUCTION_TABLES:
            assert b"NA" in initial, f"{name} lacks an explicit unavailable cell"
    for name, figure in figures.items():
        destination = writer.write_project_evidence_figure(figure, name)
        assert destination.name == f"{name}.svg"
        initial = destination.read_bytes()
        assert writer.write_project_evidence_figure(figure, name).read_bytes() == initial
        assert b"dc:date" not in initial
        if name in UNAVAILABLE_MARKED_FIGURES:
            assert b"unavailable" in initial, f"{name} lacks an explicit unavailable marker"
    frontier = figures[ReportArtifactName.SEMANTIC_SUFFICIENCY_FRONTIER_FIGURE]
    assert {str(item.name).endswith("| oracle/diagnostic-only") for item in frontier.series} == {
        True,
        False,
    }


def test_unavailable_cells_render_as_explicit_na_text(tmp_path: Path) -> None:
    layout = build_layout(tmp_path)
    writer = VerifiedEvidenceWriter(ArtifactStore(layout.execution_root), layout)
    table = EvidenceTable(
        columns=(ReportColumnName.PAIR, ReportColumnName.REALIZED_GAIN),
        rows=((PAIR, None),),
    )
    destination = writer.write_project_evidence_table(table, ReportArtifactName.ABLATION_RESULTS)
    rendered = destination.read_text(encoding="utf-8").splitlines()
    assert rendered[0] == "pair,realized_gain"
    assert rendered[1] == f"{PAIR},{UNAVAILABLE_CELL_TEXT}"


def test_export_fingerprints_are_recorded_per_registered_export(tmp_path: Path) -> None:
    layout = build_layout(tmp_path)
    writer = VerifiedEvidenceWriter(ArtifactStore(layout.execution_root), layout)
    writer.write_project_summary((), ())
    table_name = ReportArtifactName.ABLATION_RESULTS
    figure_name = ReportArtifactName.SCALABILITY_FIGURE
    writer.write_project_evidence_table(
        _registered_tables()[table_name],
        table_name,
        dependency_artifact_ids=(ArtifactIdentifier("metric-artifact"),),
    )
    writer.write_project_evidence_figure(
        _registered_figures()[figure_name],
        figure_name,
        dependency_artifact_ids=(ArtifactIdentifier("metric-artifact"),),
    )
    ledger = layout.project_summary / "reproducibility" / "execution" / "execution.json"
    recorded = json.loads(ledger.read_text(encoding="utf-8"))[EXPORT_LEDGER_KEY]
    assert set(recorded) == {table_name.value, figure_name.value}
    assert recorded[table_name.value] != recorded[figure_name.value]


def test_changing_a_declared_dependency_changes_only_that_export(tmp_path: Path) -> None:
    layout = build_layout(tmp_path)
    writer = VerifiedEvidenceWriter(ArtifactStore(layout.execution_root), layout)
    tables = _registered_tables()
    figures = _registered_figures()
    name = ReportArtifactName.SCALABILITY_FIGURE
    other = ReportArtifactName.ABLATION_RESULTS
    first = writer.write_project_evidence_figure(figures[name], name, (ArtifactIdentifier("a"),))
    initial = first.read_bytes()
    other_path = writer.write_project_evidence_table(tables[other], other)
    other_initial = other_path.read_bytes()
    changed = writer.write_project_evidence_figure(figures[name], name, (ArtifactIdentifier("b"),))
    assert changed.read_bytes() != initial
    assert other_path.read_bytes() == other_initial


def test_renderers_do_not_recompute_registered_statistics() -> None:
    for module in (reporting_module, evidence_module):
        assert module.__file__ is not None
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "scipy" not in source
        assert "spearmanr" not in source
    assert active_config().reporting.precision.scientific_metric_decimals == 4


def _ablation_comparison(
    pair: DirectedPairName,
    method: TransferMethod,
    decision: ComparisonDecision,
    equivalence_margin_low: float | None,
) -> PairedComparisonRecord:
    return PairedComparisonRecord(
        contrast_name=ContrastName(f"FedORBIT vs ablation: {pair}"),
        family=MultiplicityFamily.MECHANISM_ABLATIONS,
        pair=pair,
        method_a=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        method_b=method,
        metric=MetricId.MACRO_CROSS_ENTROPY,
        paired_seed_count=8,
        mean_difference=0.5,
        median_difference=0.5,
        bca_ci_low=0.1,
        bca_ci_high=0.9,
        raw_p=0.02,
        holm_p=0.04,
        materiality_threshold=None,
        equivalence_margin_low=equivalence_margin_low,
        equivalence_margin_high=(
            None if equivalence_margin_low is None else equivalence_margin_low + 0.01
        ),
        input_metric_artifact_ids=(ArtifactIdentifier("ablation-metric"),),
        dependency_fingerprint_sha256=FINGERPRINT,
        decision=decision,
    )


def test_ablation_table_consumes_stored_gain_and_equivalence_disposition() -> None:
    ablation = TransferMethod.COUPLING_DESTROYED_FEDORBIT
    metric_records = (
        _metric_record(
            MetricId.RELATIVE_MACRO_CE_GAIN, 0.42, MetricUnit.FRACTION, OTHER_PAIR, ablation
        ),
        _metric_record(MetricId.MACRO_CROSS_ENTROPY, 1.1, MetricUnit.NATS, OTHER_PAIR, ablation),
    )
    comparison_records = (
        _ablation_comparison(OTHER_PAIR, ablation, ComparisonDecision.EQUIVALENT, 0.05),
        _ablation_comparison(OTHER_PAIR, ablation, ComparisonDecision.SUPERIOR, None),
    )
    table = ablation_results_table(_generic_rows(), metric_records, comparison_records)
    method_index = table.columns.index(ReportColumnName.ABLATION)
    matches = [row for row in table.rows if row[method_index] == ablation.value]
    assert len(matches) == 1
    row = matches[0]
    assert row[table.columns.index(ReportColumnName.REALIZED_GAIN)] == "0.4200"
    assert row[table.columns.index(ReportColumnName.DIFFERENCE_VS_FULL)] is None
    assert not isinstance(row[table.columns.index(ReportColumnName.DIFFERENCE_VS_FULL)], float)
    assert row[table.columns.index(ReportColumnName.EQUIVALENCE)] == "Equivalent"


def test_ablation_table_leaves_unstored_columns_unavailable() -> None:
    table = ablation_results_table(_generic_rows())
    method_index = table.columns.index(ReportColumnName.ABLATION)
    matches = [row for row in table.rows if row[method_index] == "Coupling-Destroyed FedORBIT"]
    row = matches[0]
    assert row[table.columns.index(ReportColumnName.REALIZED_GAIN)] is None
    assert row[table.columns.index(ReportColumnName.RETAINED_GAIN)] is None
    assert row[table.columns.index(ReportColumnName.CONFIRMATION_SAFETY)] is None
    assert row[table.columns.index(ReportColumnName.EQUIVALENCE)] is None


def test_sparsity_table_consumes_two_stored_gains_for_the_dense_difference() -> None:
    sparse_condition = EvaluationCondition.exact_sparse(
        SupportSize(active_config().scientific.action.principal_sparse_support)
    ).name
    dense_condition = EvaluationCondition(EvaluationConditionKind.DENSE_CCP).name
    metric_records = (
        _metric_record(
            MetricId.RELATIVE_MACRO_CE_GAIN,
            0.2,
            MetricUnit.FRACTION,
            PAIR,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            sparse_condition,
        ),
        _metric_record(
            MetricId.RELATIVE_MACRO_CE_GAIN,
            0.5,
            MetricUnit.FRACTION,
            PAIR,
            TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
            dense_condition,
        ),
    )
    rows: tuple[Mapping[ReportColumnName, TableScalar], ...] = (
        OrderedDict(
            (
                (ReportColumnName.SUPPORT_OR_DENSE_CONDITION, dense_condition),
                (ReportColumnName.PAIR, PAIR),
            )
        ),
        OrderedDict(
            (
                (ReportColumnName.SUPPORT_OR_DENSE_CONDITION, sparse_condition),
                (ReportColumnName.PAIR, PAIR),
            )
        ),
    )
    table = sparsity_and_dense_results_table(rows, metric_records)
    difference = table.rows[0][table.columns.index(ReportColumnName.DENSE_MINUS_SPARSE_DIFFERENCE)]
    assert difference == "0.3000"
    unavailable = sparsity_and_dense_results_table(rows)
    assert (
        unavailable.rows[0][
            unavailable.columns.index(ReportColumnName.DENSE_MINUS_SPARSE_DIFFERENCE)
        ]
        is None
    )


def test_failure_boundary_table_derives_states_from_stored_evidence() -> None:
    table = failure_boundary_results_table(_boundary_rows())
    dimension_index = table.columns.index(ReportColumnName.BOUNDARY_DIMENSION)
    state_index = table.columns.index(ReportColumnName.STATE)
    by_dimension = {row[dimension_index]: row[state_index] for row in table.rows}
    assert by_dimension["response-scale"] == "Completed"
    assert by_dimension["support-budget"] == "Abstained"
    assert by_dimension["semantic-sufficiency-partition"] == "Completed"
    abstained = failure_boundary_results_table(
        (
            OrderedDict(
                (
                    (ReportColumnName.BOUNDARY_DIMENSION, "response-scale"),
                    (ReportColumnName.SETTING, "5"),
                    (ReportColumnName.REALIZED_GAIN, None),
                    (ReportColumnName.ABSTENTION, None),
                    (ReportColumnName.STATE, "Completed"),
                )
            ),
        )
    )
    state = abstained.rows[0][abstained.columns.index(ReportColumnName.STATE)]
    assert isinstance(state, str)
    assert state.startswith("Unavailable")


def test_coupling_gap_factor_series_represents_registered_factors_and_states() -> None:
    series = coupling_gap_factor_series(_coupling_rows())
    names = tuple(str(item.name) for item in series)
    assert "compatibility | compatibility=jointly_realizable" in names
    assert "response heterogeneity | compatibility=jointly_realizable" in names
    assert "directed asymmetry | compatibility=incompatible" in names
    assert "support budget" in names
    labelled = {str(item.name): item for item in series}
    heterogeneity = labelled["response heterogeneity | compatibility=jointly_realizable"]
    assert heterogeneity.x == (0.5,)
    assert heterogeneity.y == (0.25,)
    assert any(str(item.name).startswith("unparsed coupling conditions") for item in series)


def test_coupling_gap_figure_skips_unparsed_conditions_instead_of_numbering_rows() -> None:
    figure = coupling_gap_phase_figure(factor_rows=_coupling_rows())
    assert figure.horizontal_reference_lines == (
        0.0,
        active_config().generators.coupling_structure.incompatible_fixed_action_gap_strictly_greater_than,
    )
    for item in figure.series:
        if str(item.name).startswith("unparsed"):
            assert item.x == ()
            assert item.y == ()
            continue
        assert all(isinstance(value, int | float) for value in item.x)
        assert all(0.0 <= value <= 3.0 for value in item.x)


def test_sparsity_figure_separates_support_and_dense_conditions() -> None:
    figure = sparsity_utility_efficiency_figure(sparsity_rows=_sparsity_rows())
    names = tuple(str(item.name) for item in figure.series)
    assert names == (
        "exact sparse s=1",
        "exact sparse s=2",
        "unavailable sparsity conditions | n=1",
    )
    dense_series = next(item for item in figure.series if str(item.name) == "exact sparse s=2")
    assert dense_series.x == (2.0,)
    assert dense_series.y == (0.2,)
    assert dense_series.marker_sizes == (48.0,)


def test_scalability_figure_separates_methods_and_marks_timeouts() -> None:
    figure = scalability_figure(scalability_rows=_scalability_rows())
    names = tuple(str(item.name) for item in figure.series)
    assert names[0] == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value
    assert names[1] == TransferMethod.GENERIC_EXACT_QAP.value
    assert "timeout limit reached" in names
    assert "unavailable scalability cells | n=1" in names
    timeouts = next(item for item in figure.series if str(item.name) == "timeout limit reached")
    assert timeouts.x == (200.0,)
    assert timeouts.y == (9.5,)


def test_confirmation_safety_series_keeps_zero_coverage_and_no_proposal_pairs() -> None:
    figure = confirmation_safety_coverage_figure(
        confirmation_rows=_generic_rows(),
        registered_pairs=(PAIR, UNSEEN_PAIR),
    )
    by_name = {str(item.name): item for item in figure.series}
    assert by_name[PAIR].x == (0.0,)
    assert by_name[PAIR].y == (0.25,)
    assert by_name[PAIR].arrow_y == (0.1,)
    absent = [name for name in by_name if name.startswith(f"{UNSEEN_PAIR} | no proposal state")]
    assert len(absent) == 1
    assert by_name[absent[0]].x == ()
    assert any(
        name.startswith(f"{OTHER_PAIR} | confirmation rates unavailable") for name in by_name
    )


def test_failure_boundary_figure_requires_a_stored_numeric_setting() -> None:
    figure = failure_boundary_figure(boundary_rows=_boundary_rows())
    by_name = {str(item.name): item for item in figure.series}
    assert by_name["response-scale"].x == (0.5,)
    assert by_name["response-scale"].y == (0.1,)
    assert "support-budget" not in by_name
    assert "semantic-sufficiency-partition" not in by_name
    unavailable = [name for name in by_name if name.startswith("abstained or ineligible")]
    assert len(unavailable) == 1


def test_frontier_figure_marks_oracle_and_singleton_partitions() -> None:
    figure = semantic_sufficiency_frontier_figure(_frontier_series())
    marked = tuple(str(item.name) for item in figure.series)
    assert marked[0] == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value
    assert marked[1].endswith("| oracle/diagnostic-only")
    assert marked[2].endswith("| oracle/diagnostic-only")


def test_predicted_vs_realized_figure_consumes_the_stored_spearman_correlation() -> None:
    figure = predicted_vs_realized_transfer_figure(metric_records=_predicted_realized_records())
    names = tuple(str(item.name) for item in figure.series)
    assert names[0] == f"{PAIR} | Spearman rho=0.999; n=2"
    assert names[1].startswith("predicted-to-realized pairs | unavailable")
    assert all("0.881" not in name for name in names)


def test_map_value_bound_figure_marks_ineligible_conditions() -> None:
    figure = map_value_bound_figure(metric_records=_map_value_records())
    names = tuple(str(item.name) for item in figure.series)
    assert names[0] == MetricId.EXACT_MAP_ACTION_VALUE.value
    assert names[1].startswith("ineligible map-value bound cells | unavailable n=1")
    assert figure.series[0].x == (0.4,)
    assert figure.series[0].y == (0.25,)
