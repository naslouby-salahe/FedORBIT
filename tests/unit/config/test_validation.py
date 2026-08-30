from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.typed_access import ConfigDocument

from fedorbit.config.models import FedorbitConfig, nominal_alpha
from fedorbit.config.validation import ConfigurationContractError, validate_cross_field_contract
from fedorbit.domain.enums import TransferMethod


def _validate_raw(config: ConfigDocument) -> FedorbitConfig:
    model = FedorbitConfig.model_validate(config.as_dict())
    validate_cross_field_contract(model)
    return model


def test_principal_sparse_support_locked(fedorbit_config: FedorbitConfig) -> None:
    assert fedorbit_config.scientific.action.principal_sparse_support == 2


def test_sparse_support_sensitivity_locked(fedorbit_config: FedorbitConfig) -> None:
    assert fedorbit_config.scientific.action.sparse_support_sensitivity == (1, 3)


def test_action_constants_locked(fedorbit_config: FedorbitConfig) -> None:
    action = fedorbit_config.scientific.action
    assert action.total_curriculum_budget == 0.5
    assert action.coordinate_cap == 0.25
    assert action.linear_cost_per_actionable_node == 0.01
    assert action.positive_source_value_threshold == 0.0
    assert action.maximum_source_proposals_per_target == 3


def test_materiality_constants_locked(fedorbit_config: FedorbitConfig) -> None:
    materiality = fedorbit_config.scientific.materiality
    assert materiality.coupling_objective_units == 0.005
    assert materiality.realized_relative_macro_ce == 0.01
    assert materiality.macro_f1_absolute == 0.005
    assert materiality.equivalence_relative_macro_ce.lower == -0.01
    assert materiality.equivalence_relative_macro_ce.upper == 0.01
    assert materiality.harmful_transfer_relative_macro_ce_gain == -0.01
    assert materiality.useful_transfer_relative_macro_ce_gain == 0.01


def test_transfer_support_thresholds_locked(fedorbit_config: FedorbitConfig) -> None:
    support = fedorbit_config.scientific.transfer_support
    assert support.source_train_minimum == 200
    assert support.source_meta_minimum == 40
    assert support.target_meta_minimum == 40
    assert support.target_confirm_minimum == 40
    assert support.target_test_minimum == 40
    assert support.local_prediction_attack_class_total_rows_minimum == 200
    assert support.minimum_actionable_target_concepts == 4
    assert support.minimum_nontrivial_block_size == 2


def test_split_intervals_locked(fedorbit_config: FedorbitConfig) -> None:
    intervals = fedorbit_config.scientific.split.duplicate_safe_chronological_intervals
    assert intervals.train == (0.0, 0.55)
    assert intervals.meta == (0.55, 0.7)
    assert intervals.valid == (0.7, 0.8)
    assert intervals.confirm == (0.8, 0.9)
    assert intervals.test == (0.9, 1.0)


def test_preprocessing_thresholds_locked(fedorbit_config: FedorbitConfig) -> None:
    preprocessing = fedorbit_config.scientific.preprocessing
    assert preprocessing.missing_indicator_train_rate_threshold == 0.001
    assert preprocessing.rare_category_train_frequency_threshold == 0.001
    assert preprocessing.feature_missing_or_nonfinite_drop_threshold == 0.05
    assert preprocessing.client_invalidity_dropped_feature_fraction_threshold == 0.2
    assert preprocessing.numeric_clip.lower == -10.0
    assert preprocessing.numeric_clip.upper == 10.0
    assert preprocessing.zero_iqr_replacement_scale == 1.0


def test_training_parameters_locked(fedorbit_config: FedorbitConfig) -> None:
    training = fedorbit_config.scientific.training
    assert training.adamw.beta1 == 0.9
    assert training.adamw.beta2 == 0.999
    assert training.adamw.epsilon == 1e-8
    assert training.maximum_epochs == 50
    assert training.batch_size == 512
    assert training.gradient_clip_global_l2_norm == 1.0
    assert training.early_stopping.patience_completed_epochs == 7
    assert training.early_stopping.minimum_improvement == 1e-4
    assert training.checkpoint.tie_tolerance == 1e-6
    assert training.label_smoothing == 0.0
    assert training.dataloader_workers == 0


def test_pilot_grids_locked(fedorbit_config: FedorbitConfig) -> None:
    pilot = fedorbit_config.scientific.base_model_pilot
    assert pilot.learning_rates == (0.0003, 0.001, 0.003)
    assert pilot.weight_decays == (0.0, 0.0001)
    assert pilot.dropouts == (0.0, 0.1)


def test_seed_lists_locked(fedorbit_config: FedorbitConfig) -> None:
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


def test_statistics_parameters_locked(fedorbit_config: FedorbitConfig) -> None:
    statistics = fedorbit_config.scientific.statistics
    assert statistics.confidence_level == 0.95
    assert statistics.ci_bootstrap_repetitions == 10000
    assert statistics.minimum_valid_paired_seeds == 8
    assert statistics.tost_alpha_per_one_sided_test == 0.05
    assert statistics.spearman_minimum_valid_points == 5
    assert statistics.mcnemar_exact_to_asymptotic_discordant_pair_switch == 25


def test_nominal_alpha_is_derived_not_configured(fedorbit_config: FedorbitConfig) -> None:
    assert nominal_alpha(fedorbit_config) == 0.05


def test_evaluation_criteria_locked(fedorbit_config: FedorbitConfig) -> None:
    criteria = fedorbit_config.scientific.evaluation_criteria
    assert criteria.strict_cross_telemetry_utility.successful_primary_pairs_required == 3
    assert criteria.strict_cross_telemetry_utility.holm_adjusted_p_maximum == 0.05
    assert criteria.coupling_mechanism.theorem_zero_strict_classification_accuracy_required == 1.0
    assert criteria.coupling_mechanism.real_packet_fraction_with_material_gap_minimum == 0.25
    assert criteria.coupling_mechanism.destruction_positive_gain_retention_minimum == 0.9
    assert criteria.sparse_operational_relevance.compared_sparse_support == 3
    assert criteria.sparse_operational_relevance.valid_unit_fraction_required == 0.75
    assert criteria.confirmation_safety.absolute_risk_reduction_minimum == 0.02
    assert criteria.confirmation_safety.relative_risk_reduction_minimum == 0.30
    assert criteria.confirmation_safety.pair_coverage_loss_maximum == 0.20


def test_solver_parameters_locked(fedorbit_config: FedorbitConfig) -> None:
    exact_sparse = fedorbit_config.solvers.exact_sparse
    assert exact_sparse.lp_primal_feasibility_tolerance == 1e-9
    assert exact_sparse.lp_dual_feasibility_tolerance == 1e-9
    assert exact_sparse.lp_optimality_tolerance == 1e-9
    assert exact_sparse.separator_cut_stopping_tolerance == 1e-8
    assert exact_sparse.exact_validation_absolute_tolerance == 1e-9
    assert exact_sparse.permutation_certificate_residual_tolerance == 1e-10
    assert exact_sparse.maximum_cuts_per_support == 500
    assert exact_sparse.lp_threads_per_solve == 1
    assert exact_sparse.maximum_concurrent_supports == 4
    assert exact_sparse.deterministic_random_seed == 0
    assert fedorbit_config.solvers.dense_ccp.deterministic_starts == 5
    assert fedorbit_config.solvers.dense_ccp.outer_action_cuts == 1000


def test_target_budget_locked(fedorbit_config: FedorbitConfig) -> None:
    budget = fedorbit_config.scientific.target_optimizer_budget
    assert budget.maximum_steps_per_method_pair_seed_before_test == 10000
    assert budget.reserved.target_response_diagnostic == 3200
    assert budget.reserved.confirmation_candidates == 6000
    assert budget.reserved.live_assimilation == 500
    assert budget.reserved.nontransferable_safety_reserve == 300


def test_environment_versions_locked(fedorbit_config: FedorbitConfig) -> None:
    environment = fedorbit_config.environment
    assert environment.python == "3.13.12"
    assert environment.pytorch == "2.13.0"
    assert environment.numpy == "2.5.2"
    assert environment.scipy == "1.18.0"
    assert environment.scikit_learn == "1.9.0"
    assert environment.pandas == "3.0.5"
    assert environment.pyarrow == "25.0.1"
    assert environment.highspy_highs == "1.15.1"
    assert environment.pyscipopt == "6.2.1"
    assert environment.pydantic == "2.13.4"
    assert environment.typer == "0.27.1"
    assert environment.psutil == "7.2.2"
    assert environment.pytest == "9.1.1"
    assert environment.pytest_cov == "7.1.0"


def test_reporting_precision_locked(fedorbit_config: FedorbitConfig) -> None:
    precision = fedorbit_config.reporting.precision
    assert precision.scientific_metric_decimals == 4
    assert precision.macro_f1_decimals == 4
    assert precision.balanced_accuracy_decimals == 4
    assert precision.p_value_decimals == 4
    assert precision.p_value_less_than_threshold == 0.0001
    assert precision.runtime_seconds_decimals == 3
    assert precision.memory_decimals == 1


def test_reference_hardware_locked(fedorbit_config: FedorbitConfig) -> None:
    assert fedorbit_config.runtime.reference_model_gpu == "NVIDIA GeForce RTX 5060 Ti 16 GB"
    assert fedorbit_config.runtime.solver_cpu_worker_ceiling == 4
    assert fedorbit_config.runtime.host_ram_ceiling_gib_for_registered_efficiency_runs == 16


def test_all_experiment_methods_are_registered(fedorbit_config: FedorbitConfig) -> None:
    registered = {method.value for method in TransferMethod}
    for experiment in (
        fedorbit_config.experiments.primary_strict_cross_telemetry_transfer,
        fedorbit_config.experiments.mechanism_ablations,
        fedorbit_config.experiments.target_confirmation_and_portability,
        fedorbit_config.experiments.secondary_cross_modality_generalization,
        fedorbit_config.experiments.semantic_sufficiency_frontier,
        fedorbit_config.experiments.weak_signal_support_and_heterogeneity_boundaries,
        fedorbit_config.experiments.exact_sparse_solver_benchmark,
        fedorbit_config.experiments.synthetic_coupling_mechanism_validation,
    ):
        for method in experiment.methods:
            assert method in registered or method == "exact_orbit"
    audit = fedorbit_config.experiments.map_availability_applicability_audit
    for method in audit.packet_only_recovery_methods:
        assert method in registered


def _expect_contract_error(config: ConfigDocument) -> None:
    with pytest.raises(ConfigurationContractError):
        _validate_raw(config)


def test_rejects_nine_confirmatory_seeds(mutable_config: ConfigDocument) -> None:
    seeds = mutable_config.list("scientific", "randomness", "confirmatory_seeds")
    mutable_config.set_value("scientific", "randomness", "confirmatory_seeds", value=seeds[:-1])
    _expect_contract_error(mutable_config)


def test_rejects_duplicate_pilot_seed(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value("scientific", "randomness", "pilot_seeds", value=[101, 101, 303])
    _expect_contract_error(mutable_config)


def test_rejects_gap_in_split_intervals(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value(
        "scientific",
        "split",
        "duplicate_safe_chronological_intervals",
        "train",
        value=[0.0, 0.54],
    )
    _expect_contract_error(mutable_config)


def test_rejects_unknown_pair_client(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value(
        "scientific",
        "datasets",
        "primary_directed_pairs",
        0,
        "target",
        value="invented_client",
    )
    with pytest.raises(ValidationError):
        _validate_raw(mutable_config)


def test_rejects_self_pair(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value(
        "scientific",
        "datasets",
        "primary_directed_pairs",
        0,
        "target",
        value="edge_iiotset_network",
    )
    _expect_contract_error(mutable_config)


def test_rejects_secondary_client_in_primary_pair(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value(
        "scientific",
        "datasets",
        "primary_directed_pairs",
        0,
        "target",
        value="ton_iot_network",
    )
    _expect_contract_error(mutable_config)


def test_rejects_nonzero_dataloader_workers(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value("scientific", "training", "dataloader_workers", value=2)
    _expect_contract_error(mutable_config)


def test_rejects_unknown_experiment_method(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value(
        "experiments",
        "primary_strict_cross_telemetry_transfer",
        "methods",
        value=["FedORBIT Exact-Sparse Solver", "Local-Only", "Invented Method"],
    )
    _expect_contract_error(mutable_config)


def test_rejects_benchmark_without_principal_solver(mutable_config: ConfigDocument) -> None:
    methods = mutable_config.list("experiments", "exact_sparse_solver_benchmark", "methods")
    filtered = [method for method in methods if method != "FedORBIT Exact-Sparse Solver"]
    mutable_config.set_value(
        "experiments", "exact_sparse_solver_benchmark", "methods", value=filtered
    )
    _expect_contract_error(mutable_config)


def test_rejects_budget_overflow(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value(
        "scientific",
        "target_optimizer_budget",
        "reserved",
        "nontransferable_safety_reserve",
        value=1000,
    )
    _expect_contract_error(mutable_config)


def test_rejects_invalid_split_width(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value(
        "scientific",
        "split",
        "duplicate_safe_chronological_intervals",
        "test",
        value=[0.9, 0.9],
    )
    _expect_contract_error(mutable_config)


def test_rejects_unsupported_sensitivity_value(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value("scientific", "action", "sparse_support_sensitivity", value=[1, 4])
    _expect_contract_error(mutable_config)


def test_rejects_sensitivity_containing_principal_support(
    mutable_config: ConfigDocument,
) -> None:
    mutable_config.set_value("scientific", "action", "sparse_support_sensitivity", value=[1, 2])
    _expect_contract_error(mutable_config)


def test_rejects_unknown_generator_support(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value("generators", "exact_separator_theorem", "supports", value=[1, 2, 4])
    _expect_contract_error(mutable_config)


def test_rejects_wrong_artifact_layout(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value("runtime", "artifact_layout", "execution_root", value="artifacts")
    _expect_contract_error(mutable_config)


def test_rejects_wrong_preprocessing_subdirectories(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value(
        "runtime",
        "artifact_layout",
        "preprocessing_subdirectories",
        value=["inventories", "prepared"],
    )
    _expect_contract_error(mutable_config)


def test_rejects_missing_primary_pair(mutable_config: ConfigDocument) -> None:
    pairs = mutable_config.list("scientific", "datasets", "primary_directed_pairs")
    mutable_config.set_value("scientific", "datasets", "primary_directed_pairs", value=pairs[:-1])
    _expect_contract_error(mutable_config)


def test_rejects_claim_requiring_five_pairs(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value(
        "scientific",
        "evaluation_criteria",
        "strict_cross_telemetry_utility",
        "successful_primary_pairs_required",
        value=5,
    )
    _expect_contract_error(mutable_config)


def test_rejects_wrong_nominal_confidence(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value("scientific", "statistics", "confidence_level", value=1.5)
    _expect_contract_error(mutable_config)


def test_rejects_unknown_multi_source_target(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value(
        "experiments", "multi_source_selection_validation", "targets", value=["invented_target"]
    )
    with pytest.raises(ValidationError):
        _validate_raw(mutable_config)
