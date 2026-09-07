from __future__ import annotations

from fedorbit.analysis.records import (
    ComparisonDecision,
    MetricDirection,
    MetricRecord,
    PairedComparisonRecord,
)
from fedorbit.infrastructure.manifests import DatasetManifest
from fedorbit.reporting import (
    FigureSeries,
    ablation_results_table,
    baseline_paired_difference_plot,
    confirmation_results_table,
    confirmation_safety_coverage_figure,
    coupling_gap_phase_figure,
    coupling_mechanism_results_table,
    dataset_and_client_protocol_table,
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
    ClientRole,
    DatasetId,
    ExperimentName,
    MetricId,
    MultiplicityFamily,
    TransferMethod,
)


def _metric(
    pair: str,
    method: TransferMethod,
    metric_name: MetricId,
    value: float,
) -> MetricRecord:
    return MetricRecord(
        experiment=ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        pair=pair,
        method=method,
        condition="primary",
        seed=1103,
        metric_name=metric_name,
        metric_value=value,
        metric_unit="unitless",
        direction=MetricDirection.DESCRIPTIVE,
        evaluation_class_set_sha256="a" * 64,
        input_artifact_ids=("artifact-1",),
        dependency_fingerprint_sha256="b" * 64,
        valid=True,
        invalid_reason=None,
    )


def _comparison(pair: str, method: TransferMethod) -> PairedComparisonRecord:
    return PairedComparisonRecord(
        contrast_name="primary-transfer-vs-local",
        family=MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
        pair=pair,
        method_a=method,
        method_b=TransferMethod.LOCAL_ONLY,
        metric=MetricId.RELATIVE_MACRO_CE_GAIN,
        paired_seed_count=8,
        mean_difference=0.02,
        median_difference=0.02,
        bca_ci_low=0.01,
        bca_ci_high=0.03,
        raw_p=0.01,
        holm_p=0.02,
        materiality_threshold=0.01,
        equivalence_margin_low=-0.01,
        equivalence_margin_high=0.01,
        input_metric_artifact_ids=("artifact-1",),
        dependency_fingerprint_sha256="c" * 64,
        decision=ComparisonDecision.SUPERIOR,
    )


def test_numerical_constants_and_seeds_table_is_generated_from_config() -> None:
    table = numerical_constants_and_seeds_table()
    assert table.columns == ("configuration_path", "value")
    assert any(str(row[0]).startswith("scientific.randomness") for row in table.rows)


def test_experiment_matrix_table_renders_supplied_rows() -> None:
    table = experiment_matrix_table(
        (
            {
                "experiment": ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION.value,
                "classification": "Validation",
                "datasets_or_pairs": "",
                "methods": "",
                "registered_seeds": "0",
                "conditions": "()",
                "derived_planned_cells": 1,
                "prerequisites": "",
                "evidence_relationship": "",
            },
        )
    )
    assert table.columns[0] == "experiment"
    assert table.rows[0][0] == ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION.value


def test_dataset_and_client_protocol_table_reflects_manifest_fields() -> None:
    manifest = DatasetManifest.model_validate(
        {
            "dataset": DatasetId.TON_IOT_WINDOWS10_HOST,
            "component": "windows10",
            "raw_files": ("windows10.csv",),
            "raw_sha256": "a" * 64,
            "raw_counts": {"windows10.csv": 100},
            "schema": "1.0",
            "adapter_feature_order": ("f1", "f2"),
            "adapter_feature_roles": {"f1": "behavioral_numeric", "f2": "behavioral_categorical"},
            "accepted_schema_aliases": ("ts",),
            "adapter_adaptations": (),
            "timestamp_field": "ts",
            "timestamp_range": ("0.0", "100.0"),
            "duplicate_counts": {"total": 2},
            "conflicting_duplicate_counts": {"total": 0},
            "local_class_counts": {"normal": 80, "backdoor": 20},
            "transfer_candidate_counts": {"Backdoor": 20},
            "feature_quality": {"dropped_feature_count": 0},
            "preprocessing_state": "materialized",
            "dependency_fingerprint_sha256": "b" * 64,
            "producer_code_sha256": "c" * 64,
        }
    )
    table = dataset_and_client_protocol_table(
        (manifest,),
        modality_by_dataset={DatasetId.TON_IOT_WINDOWS10_HOST.value: "host"},
        role_by_dataset={DatasetId.TON_IOT_WINDOWS10_HOST.value: ClientRole.PRIMARY},
        excluded_class_counts={DatasetId.TON_IOT_WINDOWS10_HOST.value: 1},
    )
    assert table.rows == (
        ("windows10", "host", 100, 100, "0.0..100.0", 2, 2, 1, 1, "primary", "a" * 64),
    )


def test_primary_strict_transfer_results_table_joins_metric_and_comparison_records() -> None:
    metrics = (
        _metric("Edge->Windows", TransferMethod.LOCAL_ONLY, MetricId.MACRO_CROSS_ENTROPY, 1.2),
        _metric(
            "Edge->Windows",
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            MetricId.MACRO_CROSS_ENTROPY,
            1.0,
        ),
    )
    comparisons = (_comparison("Edge->Windows", TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER),)
    table = primary_strict_transfer_results_table(metrics, comparisons)
    assert table.columns[0] == "pair"
    solver_row = next(
        row for row in table.rows if row[1] == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value
    )
    assert solver_row[3] == 1.0
    assert solver_row[6] == 0.02


def test_generic_row_tables_enforce_their_column_sets() -> None:
    for builder, extra_columns in (
        (transfer_ontology_and_null_padding_table, ("coarse_group",)),
        (model_and_training_protocol_table, ("model",)),
        (information_resource_matrix_table, ("method",)),
        (coupling_mechanism_results_table, ("condition_or_pair",)),
        (exact_solver_results_table, ("k",)),
        (ablation_results_table, ("ablation",)),
        (sparsity_and_dense_results_table, ("support_or_dense_condition",)),
        (confirmation_results_table, ("pair",)),
        (generalization_results_table, ("pair",)),
        (failure_boundary_results_table, ("boundary_dimension",)),
        (scalability_results_table, ("k",)),
    ):
        table = builder(())
        assert table.columns[0] == extra_columns[0]
        assert table.rows == ()


def test_figure_builders_apply_the_roadmap_axis_labels() -> None:
    series = (FigureSeries("Edge->Windows", (0.0,), (0.02,)),)
    for builder, x_label, y_label in (
        (
            real_transfer_gain_forest_plot,
            "paired mean relative macro-CE gain vs local",
            "primary directed pair",
        ),
        (
            baseline_paired_difference_plot,
            "primary directed pair",
            "seed-level paired difference",
        ),
        (
            coupling_gap_phase_figure,
            "coupling factor combination",
            "predicted structural zero/strict state",
        ),
        (
            predicted_vs_realized_transfer_figure,
            "certified robust predicted value",
            "TEST relative macro-CE gain",
        ),
        (sparsity_utility_efficiency_figure, "runtime", "realized gain"),
        (
            confirmation_safety_coverage_figure,
            "confirmation coverage",
            "harmful accepted rate",
        ),
        (semantic_sufficiency_frontier_figure, "log|orbit|", "realized gain"),
        (
            failure_boundary_figure,
            "boundary setting",
            "certified value / realized gain",
        ),
        (scalability_figure, "N_S * sum(n_g^3) (log scale)", "runtime (log scale)"),
        (map_value_bound_figure, "orbit-radius bound", "exact map action value"),
    ):
        figure = builder(series)
        assert figure.x_label == x_label
        assert figure.y_label == y_label
        assert figure.series == series
