from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from fedorbit.domain.enums import ClientRole, DatasetId


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ActionConfig(FrozenModel):
    principal_sparse_support: int
    sparse_support_sensitivity: tuple[int, ...]
    total_curriculum_budget: float
    coordinate_cap: float
    linear_cost_per_actionable_node: float
    positive_source_value_threshold: float
    maximum_source_proposals_per_target: int


class EquivalenceRelativeMacroCe(FrozenModel):
    lower: float
    upper: float


class MaterialityConfig(FrozenModel):
    coupling_objective_units: float
    realized_relative_macro_ce: float
    macro_f1_absolute: float
    equivalence_relative_macro_ce: EquivalenceRelativeMacroCe
    harmful_transfer_relative_macro_ce_gain: float
    useful_transfer_relative_macro_ce_gain: float


class TransferSupportConfig(FrozenModel):
    source_train_minimum: int
    source_meta_minimum: int
    target_meta_minimum: int
    target_confirm_minimum: int
    target_test_minimum: int
    local_prediction_attack_class_total_rows_minimum: int
    minimum_actionable_target_concepts: int
    minimum_nontrivial_block_size: int


class ClientConfig(FrozenModel):
    role: ClientRole
    source: str
    component: str
    expected_timestamp_field: str


class TimestampAliasAcceptance(FrozenModel):
    retained_row_parse_success_minimum: float


class DirectedPairSpec(FrozenModel):
    source: DatasetId
    target: DatasetId


class DatasetsConfig(FrozenModel):
    clients: dict[DatasetId, ClientConfig]
    timestamp_alias_acceptance: TimestampAliasAcceptance
    primary_directed_pairs: tuple[DirectedPairSpec, ...]
    secondary_directed_pairs: tuple[DirectedPairSpec, ...]
    local_prediction_normal_label: str


class SplitInterval(FrozenModel):
    train: tuple[float, float]
    meta: tuple[float, float]
    valid: tuple[float, float]
    confirm: tuple[float, float]
    test: tuple[float, float]


class SplitConfig(FrozenModel):
    duplicate_safe_chronological_intervals: SplitInterval


class NumericClip(FrozenModel):
    lower: float
    upper: float


class PreprocessingConfig(FrozenModel):
    missing_indicator_train_rate_threshold: float
    rare_category_train_frequency_threshold: float
    feature_missing_or_nonfinite_drop_threshold: float
    client_invalidity_dropped_feature_fraction_threshold: float
    numeric_clip: NumericClip
    zero_iqr_replacement_scale: float


class AdamWConfig(FrozenModel):
    beta1: float
    beta2: float
    epsilon: float


class EarlyStoppingConfig(FrozenModel):
    patience_completed_epochs: int
    minimum_improvement: float


class CheckpointConfig(FrozenModel):
    tie_tolerance: float


class TrainingConfig(FrozenModel):
    adamw: AdamWConfig
    maximum_epochs: int
    batch_size: int
    gradient_clip_global_l2_norm: float
    early_stopping: EarlyStoppingConfig
    checkpoint: CheckpointConfig
    label_smoothing: float
    dataloader_workers: int


class BaseModelPilotConfig(FrozenModel):
    learning_rates: tuple[float, ...]
    weight_decays: tuple[float, ...]
    dropouts: tuple[float, ...]


class SourceResponsePilotConfig(FrozenModel):
    intervention_magnitudes: tuple[float, ...]
    optimizer_step_horizons: tuple[int, ...]
    paired_schedules_per_candidate: int
    relative_derivative_discrepancy_ceiling: float
    sign_agreement_minimum: float
    useful_response_magnitude_threshold: float
    minimum_useful_intervention_columns: int
    curvature_penalty_coefficient: float
    numerical_floor: float


class SourceResponseFinalConfig(FrozenModel):
    paired_replicates_per_intervention: int
    simultaneous_confidence_level: float
    max_t_bootstrap_resamples: int
    response_risk_denominator_floor: float
    response_standard_error_floor: float
    useful_response_magnitude_threshold: float
    minimum_useful_intervention_columns: int
    median_band_width_to_median_absolute_mean_response_maximum: float


class TargetResponseDiagnosticConfig(FrozenModel):
    intervention_magnitude: float
    shadow_optimizer_steps: int
    paired_replicates: int
    simultaneous_bootstrap_resamples: int
    confidence_level: float


class ConfirmationConfig(FrozenModel):
    optimizer_steps_per_shadow: int
    paired_replicates: int
    hierarchical_bootstrap_resamples: int
    one_sided_confidence_level: float
    lower_bound_acceptance_threshold_relative_macro_ce: float
    accepted_live_assimilation_steps: int


class ReservedBudgetConfig(FrozenModel):
    target_response_diagnostic: int
    confirmation_candidates: int
    live_assimilation: int
    nontransferable_safety_reserve: int


class TargetOptimizerBudgetConfig(FrozenModel):
    maximum_steps_per_method_pair_seed_before_test: int
    reserved: ReservedBudgetConfig


class PointCorrespondenceBaselineConfig(FrozenModel):
    qap_tie_tolerance: float


class BaselinesConfig(FrozenModel):
    point_correspondence_commitment: PointCorrespondenceBaselineConfig


class TargetImportanceConfig(FrozenModel):
    class_risk_floor: float


class RandomnessConfig(FrozenModel):
    pilot_seeds: tuple[int, ...]
    confirmatory_seeds: tuple[int, ...]
    statistical_seed: int


class StatisticsConfig(FrozenModel):
    confidence_level: float
    exact_sign_flip_max_nonzero_differences_for_enumeration: int
    exact_sign_flip_comparison_tolerance: float
    ci_bootstrap_repetitions: int
    identical_difference_tolerance: float
    minimum_valid_paired_seeds: int
    tost_alpha_per_one_sided_test: float
    spearman_minimum_valid_points: int
    mcnemar_exact_to_asymptotic_discordant_pair_switch: int


class StrictCrossTelemetryUtilityCriteria(FrozenModel):
    successful_primary_pairs_required: int
    holm_adjusted_p_maximum: float
    bca_lower_bound_strictly_greater_than: float


class ExternalSourceValueVsLocalSirCriteria(FrozenModel):
    successful_primary_pairs_required: int
    holm_adjusted_p_maximum: float
    bca_lower_bound_strictly_greater_than: float


class CouplingMechanismCriteria(FrozenModel):
    theorem_zero_strict_classification_accuracy_required: float
    real_packet_fraction_with_material_gap_minimum: float
    primary_pairs_with_material_mean_gap_required: int
    holm_adjusted_p_maximum: float
    destruction_positive_gain_retention_minimum: float


class SparseOperationalRelevanceCriteria(FrozenModel):
    compared_sparse_support: int
    dense_minus_sparse_gain_maximum: float
    valid_unit_fraction_required: float
    primary_pairs_with_useful_gain_required: int


class ConfirmationSafetyCriteria(FrozenModel):
    qualifying_primary_pairs_required: int
    absolute_risk_reduction_minimum: float
    relative_risk_reduction_minimum: float
    qualifying_pair_coverage_loss_maximum: float
    pair_harmful_rate_worsening_maximum: float
    pair_coverage_loss_maximum: float
    equal_pair_absolute_risk_reduction_minimum: float
    equal_pair_relative_risk_reduction_minimum: float


class ClaimCriteriaConfig(FrozenModel):
    strict_cross_telemetry_utility: StrictCrossTelemetryUtilityCriteria
    external_source_value_vs_local_sir: ExternalSourceValueVsLocalSirCriteria
    coupling_mechanism: CouplingMechanismCriteria
    sparse_operational_relevance: SparseOperationalRelevanceCriteria
    confirmation_safety: ConfirmationSafetyCriteria


class MetricsConfig(FrozenModel):
    probability_log_floor: float
    relative_macro_ce_denominator_floor: float
    relative_solver_error_denominator_floor: float


class MultiSourceSelectionConfig(FrozenModel):
    communication_cost_coefficient_in_principal_ranking: float
    confirmation_cost_coefficient_in_principal_ranking: float


class RectangularizationIsSufficientRule(FrozenModel):
    valid_real_packet_fraction_below_coupling_materiality_minimum: float


class GenericQapDominatesRule(FrozenModel):
    intended_sparse_support_maximum: int
    median_runtime_ratio_to_exact_sparse_maximum: float
    p95_runtime_ratio_to_exact_sparse_maximum: float
    peak_memory_ratio_to_exact_sparse_maximum: float


class SparseSupportIsOperationallyIrrelevantRule(FrozenModel):
    dense_gain_advantage_over_support_3_minimum: float
    valid_primary_unit_fraction_minimum: float
    sparse_supports_that_must_fail_useful_materiality: tuple[int, ...]


class PointMatchingIsSufficientRule(FrozenModel):
    harmful_rate_worsening_maximum: float
    utility_advantage_over_fedorbit_minimum: float


class StrictInterfaceRemovesGainRule(FrozenModel):
    primary_pair_majority_required: int
    point_gain_maximum: float
    bca_upper_bound_maximum: float


class SourceResponseIsTooUnstableRule(FrozenModel):
    principal_source_packet_failure_fraction_strictly_greater_than: float


class SimplificationRulesConfig(FrozenModel):
    rectangularization_is_sufficient: RectangularizationIsSufficientRule
    generic_qap_dominates: GenericQapDominatesRule
    sparse_support_is_operationally_irrelevant: SparseSupportIsOperationallyIrrelevantRule
    point_matching_is_sufficient: PointMatchingIsSufficientRule
    strict_interface_removes_gain: StrictInterfaceRemovesGainRule
    source_response_is_too_unstable: SourceResponseIsTooUnstableRule


class ScientificConfig(FrozenModel):
    action: ActionConfig
    materiality: MaterialityConfig
    transfer_support: TransferSupportConfig
    datasets: DatasetsConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    base_model_pilot: BaseModelPilotConfig
    source_response_pilot: SourceResponsePilotConfig
    source_response_final: SourceResponseFinalConfig
    target_response_diagnostic: TargetResponseDiagnosticConfig
    confirmation: ConfirmationConfig
    target_optimizer_budget: TargetOptimizerBudgetConfig
    baselines: BaselinesConfig
    target_importance: TargetImportanceConfig
    randomness: RandomnessConfig
    statistics: StatisticsConfig
    claim_criteria: ClaimCriteriaConfig
    metrics: MetricsConfig
    multi_source_selection: MultiSourceSelectionConfig
    simplification_rules: SimplificationRulesConfig


class ExactSparseSolverConfig(FrozenModel):
    lp_primal_feasibility_tolerance: float
    lp_dual_feasibility_tolerance: float
    lp_optimality_tolerance: float
    separator_cut_stopping_tolerance: float
    exact_validation_absolute_tolerance: float
    permutation_certificate_residual_tolerance: float
    action_tie_tolerance: float
    action_tie_comparison_rounding_precision: float
    lap_objective_tie_tolerance: float
    maximum_cuts_per_support: int
    lp_threads_per_solve: int
    maximum_concurrent_supports: int
    deterministic_random_seed: int


class GenericExactQapSolverConfig(FrozenModel):
    relative_mip_gap: float
    feasibility_tolerance: float
    wall_time_seconds_per_solve: int
    threads: int
    random_seed: int


class DenseCcpSolverConfig(FrozenModel):
    penalty_multipliers_relative_to_scale: tuple[float, ...]
    maximum_iterations_per_penalty_level: int
    assignment_integrality_residual: float
    relative_objective_convergence_tolerance: float
    deterministic_starts: int
    outer_action_cuts: int
    wall_time_seconds: int
    lp_threads: int


class SolversConfig(FrozenModel):
    exact_sparse: ExactSparseSolverConfig
    generic_exact_qap: GenericExactQapSolverConfig
    dense_ccp: DenseCcpSolverConfig


class TargetImportanceGamma(FrozenModel):
    shape: float
    scale: float


class ExactSeparatorTheoremGeneratorConfig(FrozenModel):
    response_uniform: tuple[float, float]
    serialization_upper_band_increment_uniform: tuple[float, float]
    target_importance_gamma: TargetImportanceGamma
    active_action_uniform: tuple[float, float]
    block_patterns: tuple[tuple[int, ...], ...]
    supports: tuple[int, ...]
    generated_instances_per_block_pattern_support_seed_cell: int


class CouplingStructureGeneratorConfig(FrozenModel):
    unconstrained_response_uniform: tuple[float, float]
    compatibility: tuple[str, ...]
    response_heterogeneity: tuple[float, ...]
    directed_asymmetry: tuple[float, ...]
    response_sparsity: tuple[float, ...]
    block_patterns: tuple[tuple[int, ...], ...]
    supports: tuple[int, ...]
    incompatible_fixed_action_gap_strictly_greater_than: float
    maximum_attempts_per_instance: int


class CommonActionUnresolvedMapGeneratorConfig(FrozenModel):
    block_pattern: tuple[int, ...]
    block_pair_response_uniform: tuple[float, float]
    maximum_attempts: int


class RobustCompromiseUnresolvedMapGeneratorConfig(FrozenModel):
    block_pattern: tuple[int, ...]
    response_uniform: tuple[float, float]
    robust_pre_map_value_strictly_greater_than: float
    maximum_attempts_per_fixture: int


class MapDependentGeneratorConfig(FrozenModel):
    block_pattern: tuple[int, ...]
    response_uniform: tuple[float, float]
    map_value_minimum: float
    maximum_attempts: int


class ScalabilityGeneratorConfig(FrozenModel):
    response_uniform: tuple[float, float]
    block_patterns: tuple[str, ...]


class GeneratorsConfig(FrozenModel):
    exact_separator_theorem: ExactSeparatorTheoremGeneratorConfig
    coupling_structure: CouplingStructureGeneratorConfig
    common_action_unresolved_map: CommonActionUnresolvedMapGeneratorConfig
    robust_compromise_unresolved_map: RobustCompromiseUnresolvedMapGeneratorConfig
    map_dependent: MapDependentGeneratorConfig
    scalability: ScalabilityGeneratorConfig


class MathematicalPrimitiveValidationConfig(FrozenModel):
    hand_fixture_seed: int
    fixture_error_tolerance: float
    invalid_permutations_allowed: int


class SyntheticKRange(FrozenModel):
    minimum: int
    maximum: int


class ExactSparseSolverBenchmarkConfig(FrozenModel):
    synthetic_k: SyntheticKRange
    block_patterns: tuple[str, ...]
    supports: tuple[int, ...]
    exhaustive_truth_correspondence_count_maximum: int
    methods: tuple[str, ...]


class SyntheticCouplingMechanismValidationConfig(FrozenModel):
    methods: tuple[str, ...]


class CommonActionUnderUnidentifiedMapConfig(FrozenModel):
    fixtures_per_seed: int


class RobustCompromiseUnderUnidentifiedMapConfig(FrozenModel):
    fixtures_per_seed: int


class MapDependentActionBoundaryConfig(FrozenModel):
    fixtures_per_seed: int


class ExactMapValueBoundValidationConfig(FrozenModel):
    zero_map_value_fixtures_per_seed: int
    high_map_value_fixtures_per_seed: int


class PrimaryStrictCrossTelemetryTransferConfig(FrozenModel):
    methods: tuple[str, ...]


class MultiSourceSelectionValidationConfig(FrozenModel):
    targets: tuple[DatasetId, ...]


class MechanismAblationsConfig(FrozenModel):
    methods: tuple[str, ...]


class TargetConfirmationAndPortabilityConfig(FrozenModel):
    methods: tuple[str, ...]


class SecondaryCrossModalityGeneralizationConfig(FrozenModel):
    methods: tuple[str, ...]


class SemanticSufficiencyFrontierConfig(FrozenModel):
    partitions: tuple[str | tuple[str, ...], ...]
    methods: tuple[str, ...]


class WeakSignalSupportAndHeterogeneityBoundariesConfig(FrozenModel):
    response_scales: tuple[float, ...]
    ci_half_width_multipliers: tuple[float, ...]
    target_usable_support_fractions: tuple[float, ...]
    response_heterogeneity_multipliers: tuple[float, ...]
    support_budgets: tuple[int, ...]
    methods: tuple[str, ...]


class MapAvailabilityApplicabilityAuditConfig(FrozenModel):
    packet_only_recovery_methods: tuple[str, ...]
    independent_researchers: int
    minutes_per_researcher_per_pair: int


class ScalabilityAndEfficiencyConfig(FrozenModel):
    k_values: tuple[int, ...]
    block_patterns: tuple[str, ...]
    exact_qap_supports: tuple[int, ...]


class ExperimentsConfig(FrozenModel):
    mathematical_primitive_validation: MathematicalPrimitiveValidationConfig
    exact_sparse_solver_benchmark: ExactSparseSolverBenchmarkConfig
    synthetic_coupling_mechanism_validation: SyntheticCouplingMechanismValidationConfig
    common_action_under_unidentified_map: CommonActionUnderUnidentifiedMapConfig
    robust_compromise_under_unidentified_map: RobustCompromiseUnderUnidentifiedMapConfig
    map_dependent_action_boundary: MapDependentActionBoundaryConfig
    exact_map_value_bound_validation: ExactMapValueBoundValidationConfig
    primary_strict_cross_telemetry_transfer: PrimaryStrictCrossTelemetryTransferConfig
    multi_source_selection_validation: MultiSourceSelectionValidationConfig
    mechanism_ablations: MechanismAblationsConfig
    target_confirmation_and_portability: TargetConfirmationAndPortabilityConfig
    secondary_cross_modality_generalization: SecondaryCrossModalityGeneralizationConfig
    semantic_sufficiency_frontier: SemanticSufficiencyFrontierConfig
    weak_signal_support_and_heterogeneity_boundaries: (
        WeakSignalSupportAndHeterogeneityBoundariesConfig
    )
    map_availability_applicability_audit: MapAvailabilityApplicabilityAuditConfig
    scalability_and_efficiency: ScalabilityAndEfficiencyConfig


class FailureHandlingConfig(FrozenModel):
    retries_after_initial_infrastructure_failure: int


class ExperimentSubdirectories(FrozenModel):
    artifacts: tuple[str, ...]
    evaluations: tuple[str, ...]
    metrics: tuple[str, ...]
    statistics: tuple[str, ...]
    checkpoints: tuple[str, ...]
    diagnostics: tuple[str, ...]
    logs: tuple[str, ...]
    provenance: tuple[str, ...]


class ManuscriptExperimentSubdirectories(FrozenModel):
    figures: tuple[str, ...]
    tables: tuple[str, ...]
    metrics: tuple[str, ...]
    statistics: tuple[str, ...]


class ProjectSummarySubdirectories(FrozenModel):
    figures: tuple[str, ...]
    tables: tuple[str, ...]
    metrics: tuple[str, ...]
    statistics: tuple[str, ...]
    reproducibility: tuple[str, ...]


class ArtifactLayoutConfig(FrozenModel):
    execution_root: str
    manuscript_root: str
    preprocessing_subdirectories: tuple[str, ...]
    reusable_artifact_subdirectories: tuple[str, ...]
    experiment_subdirectories: ExperimentSubdirectories
    cache_subdirectories: tuple[str, ...]
    manuscript_experiment_subdirectories: ManuscriptExperimentSubdirectories
    project_summary_subdirectories: ProjectSummarySubdirectories


class RuntimeConfig(FrozenModel):
    failure_handling: FailureHandlingConfig
    reference_model_gpu: str
    solver_cpu_worker_ceiling: int
    host_ram_ceiling_gib_for_registered_efficiency_runs: int
    deterministic_kernel_warmups: int
    deterministic_kernel_timed_repetitions: int
    full_training_timing_repetitions_per_scientific_cell: int
    artifact_layout: ArtifactLayoutConfig


class EnvironmentConfig(FrozenModel):
    python: str
    pytorch: str
    numpy: str
    scipy: str
    scikit_learn: str
    pandas: str
    pyarrow: str
    highspy_highs: str
    pyscipopt: str
    pydantic: str
    typer: str
    psutil: str
    pytest: str
    pytest_cov: str


class ReportingPrecisionConfig(FrozenModel):
    scientific_metric_decimals: int
    macro_f1_decimals: int
    balanced_accuracy_decimals: int
    p_value_decimals: int
    p_value_less_than_threshold: float
    runtime_seconds_decimals: int
    memory_decimals: int


class ReportingConfig(FrozenModel):
    precision: ReportingPrecisionConfig


class FedorbitConfig(FrozenModel):
    scientific: ScientificConfig
    solvers: SolversConfig
    generators: GeneratorsConfig
    experiments: ExperimentsConfig
    runtime: RuntimeConfig
    environment: EnvironmentConfig
    reporting: ReportingConfig


def directed_pair_specs(pairs: tuple[DirectedPairSpec, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((pair.source.value, pair.target.value) for pair in pairs)


def nominal_alpha(config: FedorbitConfig) -> float:
    return round(1.0 - config.scientific.statistics.confidence_level, 10)


def all_registered_methods(config: FedorbitConfig) -> tuple[str, ...]:
    experiment_configs: tuple[
        PrimaryStrictCrossTelemetryTransferConfig
        | MechanismAblationsConfig
        | TargetConfirmationAndPortabilityConfig
        | SecondaryCrossModalityGeneralizationConfig
        | SemanticSufficiencyFrontierConfig
        | WeakSignalSupportAndHeterogeneityBoundariesConfig
        | ExactSparseSolverBenchmarkConfig
        | SyntheticCouplingMechanismValidationConfig,
        ...,
    ] = (
        config.experiments.primary_strict_cross_telemetry_transfer,
        config.experiments.mechanism_ablations,
        config.experiments.target_confirmation_and_portability,
        config.experiments.secondary_cross_modality_generalization,
        config.experiments.semantic_sufficiency_frontier,
        config.experiments.weak_signal_support_and_heterogeneity_boundaries,
        config.experiments.exact_sparse_solver_benchmark,
        config.experiments.synthetic_coupling_mechanism_validation,
    )
    methods: list[str] = []
    for experiment in experiment_configs:
        for method in experiment.methods:
            if method not in methods:
                methods.append(method)
    for (
        method
    ) in config.experiments.map_availability_applicability_audit.packet_only_recovery_methods:
        if method not in methods:
            methods.append(method)
    return tuple(methods)


def registered_client_ids(config: FedorbitConfig) -> tuple[DatasetId, ...]:
    return tuple(config.scientific.datasets.clients.keys())
