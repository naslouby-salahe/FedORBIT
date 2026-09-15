from __future__ import annotations

import pytest

from fedorbit.config.models import FedorbitConfig
from fedorbit.types import ClientRole, DatasetId, MetricId


def test_action_contract_is_locked(fedorbit_config: FedorbitConfig) -> None:
    action = fedorbit_config.scientific.action
    assert action.principal_sparse_support == 2
    assert action.sparse_support_sensitivity == (1, 3)
    assert action.total_curriculum_budget == 0.50
    assert action.coordinate_cap == 0.25
    assert action.linear_cost_per_actionable_node == 0.01
    assert action.positive_source_value_threshold == 0.0
    assert action.maximum_source_proposals_per_target == 3


def test_client_registry_and_primary_pairs_are_locked(fedorbit_config: FedorbitConfig) -> None:
    datasets = fedorbit_config.scientific.datasets
    assert tuple(datasets.clients) == (
        DatasetId.EDGE_IIOTSET_NETWORK,
        DatasetId.TON_IOT_WINDOWS10_HOST,
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        DatasetId.TON_IOT_NETWORK,
    )
    assert datasets.clients[DatasetId.EDGE_IIOTSET_NETWORK].role == ClientRole.EXTERNAL
    assert all(
        datasets.clients[dataset].role == ClientRole.PRIMARY
        for dataset in (
            DatasetId.TON_IOT_WINDOWS10_HOST,
            DatasetId.TON_IOT_LINUX_PROCESS_HOST,
            DatasetId.TON_IOT_NETWORK,
        )
    )
    assert tuple((pair.source, pair.target) for pair in datasets.primary_directed_pairs) == (
        (DatasetId.TON_IOT_WINDOWS10_HOST, DatasetId.TON_IOT_LINUX_PROCESS_HOST),
        (DatasetId.TON_IOT_LINUX_PROCESS_HOST, DatasetId.TON_IOT_WINDOWS10_HOST),
        (DatasetId.TON_IOT_WINDOWS10_HOST, DatasetId.TON_IOT_NETWORK),
        (DatasetId.TON_IOT_NETWORK, DatasetId.TON_IOT_WINDOWS10_HOST),
        (DatasetId.TON_IOT_LINUX_PROCESS_HOST, DatasetId.TON_IOT_NETWORK),
        (DatasetId.TON_IOT_NETWORK, DatasetId.TON_IOT_LINUX_PROCESS_HOST),
    )
    assert datasets.secondary_directed_pairs == ()


def test_materiality_contract_is_locked(fedorbit_config: FedorbitConfig) -> None:
    materiality = fedorbit_config.scientific.materiality
    assert materiality.coupling_objective_units == 0.005
    assert materiality.realized_relative_macro_ce == 0.01
    assert materiality.macro_f1_absolute == 0.005
    assert materiality.equivalence_relative_macro_ce.lower == -0.01
    assert materiality.equivalence_relative_macro_ce.upper == 0.01
    assert materiality.harmful_transfer_relative_macro_ce_gain == -0.01
    assert materiality.useful_transfer_relative_macro_ce_gain == 0.01


def test_transfer_support_contract_is_locked(fedorbit_config: FedorbitConfig) -> None:
    support = fedorbit_config.scientific.transfer_support
    assert support.source_train_minimum == 200
    assert support.source_meta_minimum == 40
    assert support.target_meta_minimum == 40
    assert support.target_confirm_minimum == 40
    assert support.target_test_minimum == 40
    assert support.local_prediction_attack_class_total_rows_minimum == 200
    assert support.minimum_actionable_target_concepts == 4
    assert support.minimum_nontrivial_block_size == 2


def test_chronological_split_contract_is_locked(fedorbit_config: FedorbitConfig) -> None:
    intervals = fedorbit_config.scientific.split.duplicate_safe_chronological_intervals
    assert intervals.train == (0.0, 0.55)
    assert intervals.meta == (0.55, 0.70)
    assert intervals.valid == (0.70, 0.80)
    assert intervals.confirm == (0.80, 0.90)
    assert intervals.test == (0.90, 1.0)


def test_preprocessing_contract_is_locked(fedorbit_config: FedorbitConfig) -> None:
    preprocessing = fedorbit_config.scientific.preprocessing
    assert preprocessing.missing_indicator_train_rate_threshold == 0.001
    assert preprocessing.rare_category_train_frequency_threshold == 0.001
    assert preprocessing.feature_missing_or_nonfinite_drop_threshold == 0.05
    assert preprocessing.client_invalidity_dropped_feature_fraction_threshold == 0.20
    assert preprocessing.numeric_clip.lower == -10.0
    assert preprocessing.numeric_clip.upper == 10.0
    assert preprocessing.zero_iqr_replacement_scale == 1.0


def test_training_and_pilot_contract_is_locked(fedorbit_config: FedorbitConfig) -> None:
    training = fedorbit_config.scientific.training
    pilot = fedorbit_config.scientific.base_model_pilot
    assert training.maximum_epochs == 50
    assert training.batch_size == 512
    assert training.early_stopping.patience_completed_epochs == 7
    assert training.early_stopping.minimum_improvement == 1e-4
    assert training.checkpoint.tie_tolerance == 1e-6
    assert pilot.learning_rates == (3e-4, 1e-3, 3e-3)
    assert pilot.weight_decays == (0.0, 1e-4)
    assert pilot.dropouts == (0.0, 0.1)


def test_randomness_contract_is_locked(fedorbit_config: FedorbitConfig) -> None:
    randomness = fedorbit_config.scientific.randomness
    assert randomness.pilot_seeds == (101, 202, 303)
    assert randomness.confirmatory_seeds == (
        1103,
        2207,
        3319,
        4421,
        5531,
        6653,
        7753,
        8861,
        9973,
        11027,
    )
    assert randomness.statistical_seed == 300


def test_solver_contract_is_locked(fedorbit_config: FedorbitConfig) -> None:
    exact_sparse = fedorbit_config.solvers.exact_sparse
    exact_qap = fedorbit_config.solvers.generic_exact_qap
    dense_ccp = fedorbit_config.solvers.dense_ccp
    assert exact_sparse.lp_primal_feasibility_tolerance == 1e-9
    assert exact_sparse.lp_dual_feasibility_tolerance == 1e-9
    assert exact_sparse.lp_optimality_tolerance == 1e-9
    assert exact_sparse.separator_cut_stopping_tolerance == 1e-8
    assert exact_sparse.exact_validation_absolute_tolerance == 1e-9
    assert exact_sparse.permutation_certificate_residual_tolerance == 1e-10
    assert exact_sparse.action_tie_tolerance == 1e-10
    assert exact_sparse.action_tie_comparison_rounding_precision == 1e-12
    assert exact_sparse.lap_objective_tie_tolerance == 1e-12
    assert exact_sparse.maximum_cuts_per_support == 500
    assert exact_sparse.lp_threads_per_solve == 1
    assert exact_sparse.maximum_concurrent_supports == 4
    assert exact_sparse.deterministic_random_seed == 0
    assert exact_qap.relative_mip_gap == 1e-9
    assert exact_qap.feasibility_tolerance == 1e-9
    assert exact_qap.wall_time_seconds_per_solve == 3600
    assert exact_qap.threads == 1
    assert exact_qap.random_seed == 0
    assert dense_ccp.penalty_multipliers_relative_to_scale == (0.1, 1.0, 10.0, 100.0, 1000.0)
    assert dense_ccp.maximum_iterations_per_penalty_level == 50
    assert dense_ccp.assignment_integrality_residual == 1e-8
    assert dense_ccp.relative_objective_convergence_tolerance == 1e-8
    assert dense_ccp.deterministic_starts == 5
    assert dense_ccp.outer_action_cuts == 1000
    assert dense_ccp.wall_time_seconds == 3600
    assert dense_ccp.lp_threads == 1


def test_multi_source_ranking_coefficients_locked(fedorbit_config: FedorbitConfig) -> None:
    selection = fedorbit_config.scientific.multi_source_selection
    assert selection.communication_cost_coefficient_in_principal_ranking == 0.0
    assert selection.confirmation_cost_coefficient_in_principal_ranking == 0.0


def test_metric_floors_locked(fedorbit_config: FedorbitConfig) -> None:
    metrics = fedorbit_config.scientific.metrics
    assert metrics.probability_log_floor == 1e-12
    assert metrics.relative_macro_ce_denominator_floor == 1e-12
    assert metrics.relative_solver_error_denominator_floor == 1e-12


def test_generator_distributions_locked(fedorbit_config: FedorbitConfig) -> None:
    theorem = fedorbit_config.generators.exact_separator_theorem
    assert theorem.response_uniform == (-0.2, 0.2)
    assert theorem.serialization_upper_band_increment_uniform == (0.0, 0.05)
    assert theorem.target_importance_gamma.shape == 2.0
    assert theorem.target_importance_gamma.scale == 1.0
    assert theorem.active_action_uniform == (0.05, 0.25)
    assert theorem.block_patterns == ((2,), (3,), (4,), (2, 2), (2, 3), (3, 3))
    assert theorem.supports == (1, 2, 3)
    assert theorem.generated_instances_per_block_pattern_support_seed_cell == 100


def test_coupling_generator_factors_locked(fedorbit_config: FedorbitConfig) -> None:
    coupling = fedorbit_config.generators.coupling_structure
    assert coupling.unconstrained_response_uniform == (-0.1, 0.1)
    assert coupling.compatibility == ("jointly_realizable", "incompatible")
    assert coupling.response_heterogeneity == (0.5, 1.0, 2.0)
    assert coupling.directed_asymmetry == (0.0, 0.5, 1.0)
    assert coupling.response_sparsity == (0.25, 0.5, 1.0)
    assert coupling.block_patterns == ((2, 2), (2, 3), (3, 3))
    assert coupling.incompatible_fixed_action_gap_strictly_greater_than == 1e-6
    assert coupling.maximum_attempts_per_instance == 10000


def test_unresolved_map_generators_locked(fedorbit_config: FedorbitConfig) -> None:
    common = fedorbit_config.generators.common_action_unresolved_map
    assert common.block_pattern == (2, 2)
    assert common.block_pair_response_uniform == (0.04, 0.12)
    assert common.maximum_attempts == 1000
    robust = fedorbit_config.generators.robust_compromise_unresolved_map
    assert robust.response_uniform == (-0.1, 0.2)
    assert robust.robust_pre_map_value_strictly_greater_than == 0.005
    assert robust.maximum_attempts_per_fixture == 100000
    dependent = fedorbit_config.generators.map_dependent
    assert dependent.response_uniform == (-0.15, 0.25)
    assert dependent.map_value_minimum == 0.01
    assert dependent.maximum_attempts == 100000


def test_scalability_generator_locked(fedorbit_config: FedorbitConfig) -> None:
    scalability = fedorbit_config.generators.scalability
    assert scalability.response_uniform == (-0.1, 0.1)
    assert scalability.block_patterns == ("balanced", "maximally_skewed_two_block")


def test_experiment_grids_locked(fedorbit_config: FedorbitConfig) -> None:
    experiments = fedorbit_config.experiments
    assert experiments.mathematical_primitive_validation.hand_fixture_seed == 0
    assert experiments.mathematical_primitive_validation.fixture_error_tolerance == 1e-10
    assert experiments.exact_sparse_solver_benchmark.synthetic_k.minimum == 4
    assert experiments.exact_sparse_solver_benchmark.synthetic_k.maximum == 18
    assert (
        experiments.exact_sparse_solver_benchmark.exhaustive_truth_correspondence_count_maximum
        == 100000
    )
    assert experiments.common_action_under_unidentified_map.fixtures_per_seed == 50
    assert experiments.robust_compromise_under_unidentified_map.fixtures_per_seed == 50
    assert experiments.map_dependent_action_boundary.fixtures_per_seed == 50
    assert experiments.exact_map_value_bound_validation.zero_map_value_fixtures_per_seed == 25
    assert experiments.exact_map_value_bound_validation.high_map_value_fixtures_per_seed == 25
    assert experiments.map_availability_applicability_audit.independent_researchers == 2
    assert experiments.map_availability_applicability_audit.minutes_per_researcher_per_pair == 60
    assert experiments.scalability_and_efficiency.k_values == (6, 8, 10, 12, 16, 20, 24, 32)
    assert experiments.scalability_and_efficiency.exact_qap_supports == (1, 2, 3)


def test_weak_signal_grids_locked(fedorbit_config: FedorbitConfig) -> None:
    weak = fedorbit_config.experiments.weak_signal_support_and_heterogeneity_boundaries
    assert weak.response_scales == (1.0, 0.75, 0.5, 0.25, 0.0)
    assert weak.baseline_response_scale == 1.0
    assert weak.ci_half_width_multipliers == (1.0, 1.5, 2.0, 4.0)
    assert weak.baseline_ci_half_width_multiplier == 1.0
    assert weak.target_usable_support_fractions == (1.0, 0.5, 0.25, 0.1)
    assert weak.baseline_target_usable_support_fraction == 1.0
    assert weak.response_heterogeneity_multipliers == (0.5, 1.0, 2.0)
    assert weak.baseline_response_heterogeneity_multiplier == 1.0
    assert weak.support_budgets == (1, 2, 3)
    assert weak.baseline_support_budget == 2
    assert weak.distinct_condition_count() == 15


def test_weak_signal_baselines_must_be_registered_grid_values(
    fedorbit_config: FedorbitConfig,
) -> None:
    weak = fedorbit_config.experiments.weak_signal_support_and_heterogeneity_boundaries
    payload = weak.model_dump(mode="json")
    payload["baseline_support_budget"] = 99
    with pytest.raises(ValueError, match="baseline values"):
        type(weak).model_validate(payload)


def test_metric_catalogue_identities() -> None:
    assert MetricId.MACRO_CROSS_ENTROPY == "Macro Cross-Entropy"
    assert MetricId.RELATIVE_MACRO_CE_GAIN == "Relative Macro-CE Gain"
    assert MetricId.ACTIVE_IMAGE_CANDIDATES == "Active-Image Candidates"
    assert MetricId.LAP_CALLS == "LAP Calls"
    assert MetricId.DENSE_BOUND_GAP == "Dense Bound Gap"
    assert MetricId.ABSOLUTE_RISK_REDUCTION == "Absolute Risk Reduction"
    assert MetricId.RELATIVE_RISK_REDUCTION == "Relative Risk Reduction"
    assert MetricId.ORBIT_SIZE == "Orbit Size"
    assert MetricId.WORK_STRUCTURE_SPEARMAN == "Work-Structure Spearman"
    assert len(tuple(MetricId)) >= 40
