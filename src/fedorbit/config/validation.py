from __future__ import annotations

from collections import OrderedDict

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import ClientRole, DatasetId, TransferMethod


class ConfigurationContractError(ValueError):
    pass


_registered_method_names = {candidate.value for candidate in TransferMethod}

_experiment_local_method_names = {"exact_orbit"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationContractError(message)


def _validate_split_intervals(config: FedorbitConfig) -> None:
    intervals = config.scientific.split.duplicate_safe_chronological_intervals
    numerical_tolerance = config.scientific.source_response_pilot.numerical_floor
    named: OrderedDict[str, tuple[float, float]] = OrderedDict(
        train=intervals.train,
        meta=intervals.meta,
        valid=intervals.valid,
        confirm=intervals.confirm,
        test=intervals.test,
    )
    previous_upper = 0.0
    for name, (lower, upper) in named.items():
        _require(
            abs(lower - previous_upper) <= numerical_tolerance,
            f"split interval {name} must be contiguous with the previous interval",
        )
        _require(
            lower < upper,
            f"split interval {name} must have positive width",
        )
        _require(
            upper <= 1.0 + numerical_tolerance,
            f"split interval {name} must end at or before 1.0",
        )
        previous_upper = upper
    _require(
        abs(previous_upper - 1.0) <= numerical_tolerance,
        "split intervals must cover the full [0, 1] range",
    )


def _validate_seeds(config: FedorbitConfig) -> None:
    randomness = config.scientific.randomness
    _require(len(randomness.pilot_seeds) == 3, "pilot seed list must contain exactly 3 seeds")
    _require(
        len(randomness.confirmatory_seeds) == 10,
        "confirmatory seed list must contain exactly 10 seeds",
    )
    _require(
        len(set(randomness.pilot_seeds)) == 3,
        "pilot seeds must be distinct",
    )
    _require(
        len(set(randomness.confirmatory_seeds)) == 10,
        "confirmatory seeds must be distinct",
    )
    _require(
        all(seed > 0 for seed in randomness.pilot_seeds),
        "pilot seeds must be positive",
    )
    _require(
        all(seed > 0 for seed in randomness.confirmatory_seeds),
        "confirmatory seeds must be positive",
    )
    _require(randomness.statistical_seed > 0, "statistical seed must be positive")


def _validate_action(config: FedorbitConfig) -> None:
    action = config.scientific.action
    _require(
        action.principal_sparse_support >= 1,
        "principal sparse support must be at least 1",
    )
    _require(
        action.principal_sparse_support <= 3,
        "principal sparse support must be at most 3",
    )
    _require(
        all(support in (1, 2, 3) for support in action.sparse_support_sensitivity),
        "sparse support sensitivity values must belong to {1, 2, 3}",
    )
    _require(
        action.principal_sparse_support not in action.sparse_support_sensitivity,
        "principal sparse support must not be listed as a sensitivity alternative",
    )
    _require(action.total_curriculum_budget > 0.0, "total curriculum budget must be positive")
    _require(action.coordinate_cap > 0.0, "coordinate cap must be positive")
    _require(
        action.coordinate_cap <= action.total_curriculum_budget,
        "coordinate cap must not exceed the total curriculum budget",
    )
    _require(
        action.linear_cost_per_actionable_node >= 0.0,
        "linear cost per actionable node must be nonnegative",
    )
    _require(
        action.maximum_source_proposals_per_target >= 1,
        "maximum source proposals per target must be at least 1",
    )


def _validate_datasets(config: FedorbitConfig) -> None:
    datasets = config.scientific.datasets
    _require(
        tuple(datasets.clients.keys())
        == (
            DatasetId.EDGE_IIOTSET_NETWORK,
            DatasetId.TON_IOT_WINDOWS10_HOST,
            DatasetId.TON_IOT_LINUX_PROCESS_HOST,
            DatasetId.TON_IOT_NETWORK,
        ),
        "dataset client registry must contain exactly the four registered clients",
    )
    registered = set(datasets.clients.keys())
    for pair in (*datasets.primary_directed_pairs, *datasets.secondary_directed_pairs):
        _require(pair.source in registered, f"unknown source client in pair {pair}")
        _require(pair.target in registered, f"unknown target client in pair {pair}")
        _require(pair.source != pair.target, f"self-pair is not a directed pair: {pair}")
    _require(
        len(datasets.primary_directed_pairs) == 4,
        "exactly four primary directed pairs are registered",
    )
    _require(
        len(datasets.secondary_directed_pairs) == 4,
        "exactly four secondary directed pairs are registered",
    )
    primary_sources = {pair.source for pair in datasets.primary_directed_pairs}
    primary_targets = {pair.target for pair in datasets.primary_directed_pairs}
    _require(
        primary_sources
        == {
            DatasetId.EDGE_IIOTSET_NETWORK,
            DatasetId.TON_IOT_WINDOWS10_HOST,
            DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        },
        "primary directed pairs must involve exactly the three primary clients",
    )
    _require(
        primary_targets
        == {
            DatasetId.EDGE_IIOTSET_NETWORK,
            DatasetId.TON_IOT_WINDOWS10_HOST,
            DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        },
        "primary directed pairs must target exactly the three primary clients",
    )
    _require(
        DatasetId.TON_IOT_NETWORK not in primary_sources
        and DatasetId.TON_IOT_NETWORK not in primary_targets,
        "the secondary network client must not appear in primary directed pairs",
    )
    roles = {client_id: client.role for client_id, client in datasets.clients.items()}
    _require(
        roles[DatasetId.EDGE_IIOTSET_NETWORK].value == ClientRole.PRIMARY.value
        and roles[DatasetId.TON_IOT_WINDOWS10_HOST].value == ClientRole.PRIMARY.value
        and roles[DatasetId.TON_IOT_LINUX_PROCESS_HOST].value == ClientRole.PRIMARY.value,
        "the three primary benchmark clients must have role primary",
    )
    _require(
        roles[DatasetId.TON_IOT_NETWORK].value == ClientRole.SECONDARY.value,
        "the network client must have role secondary",
    )


def _validate_training(config: FedorbitConfig) -> None:
    training = config.scientific.training
    _require(training.maximum_epochs >= 1, "maximum epochs must be at least 1")
    _require(training.batch_size >= 1, "batch size must be at least 1")
    _require(training.dataloader_workers == 0, "dataloader workers must be 0")
    _require(
        training.early_stopping.patience_completed_epochs >= 0,
        "early-stopping patience must be nonnegative",
    )
    pilot = config.scientific.base_model_pilot
    _require(
        len(pilot.learning_rates) == 3
        and len(pilot.weight_decays) == 2
        and len(pilot.dropouts) == 2,
        "base-model pilot grid must contain 3 learning rates, 2 weight decays, and 2 dropouts",
    )
    _require(
        all(rate > 0.0 for rate in pilot.learning_rates),
        "pilot learning rates must be positive",
    )


def _validate_response(config: FedorbitConfig) -> None:
    pilot = config.scientific.source_response_pilot
    _require(
        len(pilot.intervention_magnitudes) == 3 and len(pilot.optimizer_step_horizons) == 3,
        "source-response pilot grid must contain 3 magnitudes and 3 horizons",
    )
    _require(
        pilot.paired_schedules_per_candidate >= 1,
        "paired schedules per candidate must be at least 1",
    )
    final = config.scientific.source_response_final
    _require(
        final.paired_replicates_per_intervention >= 1,
        "paired replicates per intervention must be at least 1",
    )
    _require(
        0.0 < final.simultaneous_confidence_level < 1.0,
        "simultaneous confidence level must lie in (0, 1)",
    )
    diagnostic = config.scientific.target_response_diagnostic
    _require(
        diagnostic.paired_replicates >= 1 and diagnostic.simultaneous_bootstrap_resamples >= 1,
        "target-local response diagnostic counts must be positive",
    )


def _validate_confirmation(config: FedorbitConfig) -> None:
    confirmation = config.scientific.confirmation
    _require(
        confirmation.optimizer_steps_per_shadow >= 1
        and confirmation.paired_replicates >= 1
        and confirmation.hierarchical_bootstrap_resamples >= 1,
        "confirmation counts must be positive",
    )
    _require(
        0.0 < confirmation.one_sided_confidence_level < 1.0,
        "one-sided confidence level must lie in (0, 1)",
    )
    budget = config.scientific.target_optimizer_budget
    reserved = budget.reserved
    total_reserved = (
        reserved.target_response_diagnostic
        + reserved.confirmation_candidates
        + reserved.live_assimilation
        + reserved.nontransferable_safety_reserve
    )
    _require(
        total_reserved <= budget.maximum_steps_per_method_pair_seed_before_test,
        "reserved target-local budget categories must not exceed the maximum step budget",
    )


def _validate_statistics(config: FedorbitConfig) -> None:
    statistics = config.scientific.statistics
    _require(
        0.0 < statistics.confidence_level < 1.0,
        "nominal confidence level must lie in (0, 1)",
    )
    _require(
        statistics.ci_bootstrap_repetitions >= 1 and statistics.minimum_valid_paired_seeds >= 1,
        "statistical counts must be positive",
    )
    _require(
        0.0 < statistics.tost_alpha_per_one_sided_test < 1.0,
        "TOST alpha must lie in (0, 1)",
    )
    _require(
        statistics.spearman_minimum_valid_points >= 2,
        "Spearman minimum valid points must be at least 2",
    )
    _require(
        statistics.exact_sign_flip_max_nonzero_differences_for_enumeration >= 1,
        "exact sign-flip enumeration limit must be at least 1",
    )


def _validate_evaluation_criteria(config: FedorbitConfig) -> None:
    criteria = config.scientific.evaluation_criteria
    primary_pair_count = 4
    _require(
        criteria.strict_cross_telemetry_utility.successful_primary_pairs_required
        <= primary_pair_count,
        "strict cross-telemetry required pairs must not exceed the primary pair count",
    )
    _require(
        criteria.external_source_value_vs_local_sir.successful_primary_pairs_required
        <= primary_pair_count,
        "external-source required pairs must not exceed the primary pair count",
    )
    _require(
        criteria.coupling_mechanism.primary_pairs_with_material_mean_gap_required
        <= primary_pair_count,
        "coupling mechanism required pairs must not exceed the primary pair count",
    )
    _require(
        criteria.confirmation_safety.qualifying_primary_pairs_required <= primary_pair_count,
        "confirmation safety required pairs must not exceed the primary pair count",
    )
    _require(
        0.0 < criteria.sparse_operational_relevance.valid_unit_fraction_required <= 1.0,
        "sparse operational relevance unit fraction must lie in (0, 1]",
    )
    _require(
        criteria.sparse_operational_relevance.compared_sparse_support in (1, 2, 3),
        "compared sparse support must belong to {1, 2, 3}",
    )


def _validate_simplification_rules(config: FedorbitConfig) -> None:
    rules = config.scientific.simplification_rules
    rectangular_rule = rules.rectangularization_is_sufficient
    rectangular_fraction = (
        rectangular_rule.valid_real_packet_fraction_below_coupling_materiality_minimum
    )
    _require(
        0.0 < rectangular_fraction <= 1.0,
        "rectangularization trigger fraction must lie in (0, 1]",
    )
    instability_rule = rules.source_response_is_too_unstable
    instability_fraction = (
        instability_rule.principal_source_packet_failure_fraction_strictly_greater_than
    )
    _require(
        0.0 < instability_fraction <= 1.0,
        "source-response instability trigger fraction must lie in (0, 1]",
    )
    _require(
        all(
            support in (1, 2, 3)
            for support in (
                rules.sparse_support_is_operationally_irrelevant.sparse_supports_that_must_fail_useful_materiality
            )
        ),
        "sparse supports in the irrelevance rule must belong to {1, 2, 3}",
    )
    _require(
        rules.generic_qap_dominates.intended_sparse_support_maximum in (1, 2, 3),
        "generic QAP dominates intended support maximum must belong to {1, 2, 3}",
    )


def _validate_generators(config: FedorbitConfig) -> None:
    generators = config.generators
    _require(
        all(support in (1, 2, 3) for support in generators.exact_separator_theorem.supports),
        "exact-separator theorem generator supports must belong to {1, 2, 3}",
    )
    _require(
        all(support in (1, 2, 3) for support in generators.coupling_structure.supports),
        "coupling-structure generator supports must belong to {1, 2, 3}",
    )
    _require(
        generators.coupling_structure.maximum_attempts_per_instance >= 1
        and generators.common_action_unresolved_map.maximum_attempts >= 1
        and generators.robust_compromise_unresolved_map.maximum_attempts_per_fixture >= 1
        and generators.map_dependent.maximum_attempts >= 1,
        "generator rejection-sampling attempt limits must be positive",
    )


def _validate_solvers(config: FedorbitConfig) -> None:
    exact_sparse = config.solvers.exact_sparse
    _require(
        exact_sparse.maximum_concurrent_supports >= 1,
        "maximum concurrently executed supports must be at least 1",
    )
    _require(
        exact_sparse.lp_threads_per_solve == 1,
        "exact-sparse LP threads per solve must be 1",
    )
    _require(
        exact_sparse.maximum_cuts_per_support >= 1,
        "maximum cuts per support must be at least 1",
    )
    _require(
        config.solvers.dense_ccp.deterministic_starts >= 1,
        "dense-CCP deterministic starts must be at least 1",
    )
    _require(
        config.solvers.dense_ccp.outer_action_cuts >= 1,
        "dense-CCP outer action cuts must be at least 1",
    )


def _validate_experiments(config: FedorbitConfig) -> None:
    experiments = config.experiments
    _require(
        experiments.exact_sparse_solver_benchmark.synthetic_k.minimum >= 1
        and experiments.exact_sparse_solver_benchmark.synthetic_k.maximum
        >= experiments.exact_sparse_solver_benchmark.synthetic_k.minimum,
        "exact-sparse benchmark K range must be valid",
    )
    _require(
        all(support in (1, 2, 3) for support in experiments.exact_sparse_solver_benchmark.supports),
        "exact-sparse benchmark supports must belong to {1, 2, 3}",
    )
    _require(
        all(
            support in (1, 2, 3)
            for support in experiments.scalability_and_efficiency.exact_qap_supports
        ),
        "scalability exact/QAP supports must belong to {1, 2, 3}",
    )
    _require(
        all(
            support in (1, 2, 3)
            for support in (
                experiments.weak_signal_support_and_heterogeneity_boundaries.support_budgets
            )
        ),
        "weak-signal support budgets must belong to {1, 2, 3}",
    )
    for experiment in (
        experiments.primary_strict_cross_telemetry_transfer,
        experiments.mechanism_ablations,
        experiments.target_confirmation_and_portability,
        experiments.secondary_cross_modality_generalization,
        experiments.semantic_sufficiency_frontier,
        experiments.weak_signal_support_and_heterogeneity_boundaries,
        experiments.exact_sparse_solver_benchmark,
        experiments.synthetic_coupling_mechanism_validation,
    ):
        for method in experiment.methods:
            _require(
                method in _registered_method_names or method in _experiment_local_method_names,
                f"unregistered method name in experiment configuration: {method}",
            )
    registered = set(config.scientific.datasets.clients.keys())
    for target in experiments.multi_source_selection_validation.targets:
        _require(target in registered, f"unknown multi-source validation target: {target}")
    _require(
        experiments.map_availability_applicability_audit.independent_researchers >= 1,
        "map-availability audit requires at least one independent researcher",
    )
    _require(
        experiments.map_availability_applicability_audit.minutes_per_researcher_per_pair >= 1,
        "map-availability audit session duration must be positive",
    )
    benchmark_methods = set(experiments.exact_sparse_solver_benchmark.methods)
    _require(
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value in benchmark_methods,
        "exact-sparse solver benchmark must include FedORBIT Exact-Sparse Solver",
    )
    _require(
        TransferMethod.GENERIC_EXACT_QAP.value in benchmark_methods,
        "exact-sparse solver benchmark must include Generic Exact QAP",
    )
    _require(
        TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK.value in benchmark_methods,
        "exact-sparse solver benchmark must include FedORBIT Dense-CCP Fallback",
    )
    transfer_methods = experiments.primary_strict_cross_telemetry_transfer.methods
    _require(
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value in transfer_methods,
        "primary strict transfer must include FedORBIT Exact-Sparse Solver",
    )
    _require(
        TransferMethod.LOCAL_ONLY.value in transfer_methods,
        "primary strict transfer must include Local-Only",
    )


def _validate_runtime(config: FedorbitConfig) -> None:
    runtime = config.runtime
    _require(
        runtime.failure_handling.retries_after_initial_infrastructure_failure >= 0,
        "infrastructure retry count must be nonnegative",
    )
    _require(runtime.solver_cpu_worker_ceiling >= 1, "solver CPU worker ceiling must be at least 1")
    _require(
        runtime.deterministic_kernel_warmups >= 0
        and runtime.deterministic_kernel_timed_repetitions >= 1,
        "deterministic kernel timing counts must be valid",
    )
    layout = runtime.artifact_layout
    _require(layout.execution_root == "outputs", "execution root must be outputs")
    _require(layout.manuscript_root == "results", "manuscript root must be results")
    _require(
        layout.preprocessing_subdirectories
        == ("inventories", "validation", "prepared", "splits", "features", "metadata"),
        "preprocessing subdirectories must be exactly the registered set",
    )
    _require(
        layout.reusable_artifact_subdirectories
        == ("models", "scores", "fitted", "baselines", "derived"),
        "reusable artifact subdirectories must be exactly the registered set",
    )
    _require(
        layout.cache_subdirectories
        == ("preprocessing", "models", "evaluation", "analysis", "staging"),
        "cache subdirectories must be exactly the registered set",
    )


def _validate_reporting(config: FedorbitConfig) -> None:
    precision = config.reporting.precision
    _require(
        all(
            value >= 0
            for value in (
                precision.scientific_metric_decimals,
                precision.macro_f1_decimals,
                precision.balanced_accuracy_decimals,
                precision.p_value_decimals,
                precision.runtime_seconds_decimals,
                precision.memory_decimals,
            )
        ),
        "reporting decimal precisions must be nonnegative",
    )
    _require(
        0.0 < precision.p_value_less_than_threshold < 1.0,
        "p-value display threshold must lie in (0, 1)",
    )


def validate_cross_field_contract(config: FedorbitConfig) -> None:
    _validate_split_intervals(config)
    _validate_seeds(config)
    _validate_action(config)
    _validate_datasets(config)
    _validate_training(config)
    _validate_response(config)
    _validate_confirmation(config)
    _validate_statistics(config)
    _validate_evaluation_criteria(config)
    _validate_simplification_rules(config)
    _validate_generators(config)
    _validate_solvers(config)
    _validate_experiments(config)
    _validate_runtime(config)
    _validate_reporting(config)
