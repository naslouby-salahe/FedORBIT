from __future__ import annotations

from collections.abc import Mapping

from fedorbit.types import (
    AbsoluteMetric,
    AttemptCount,
    BatchSize,
    Budget,
    ClientComponentName,
    ClientRole,
    Coefficient,
    ConceptCount,
    ConcurrencyCount,
    ConfidenceLevel,
    CouplingCompatibility,
    CutCount,
    DatasetId,
    DatasetLabel,
    DecimalPrecision,
    DomainModel,
    DurationMinutes,
    EpochCount,
    Floor,
    Fraction,
    GiBMemory,
    InterventionMagnitude,
    InvalidPermutationCount,
    LearningRate,
    MethodName,
    PatienceCount,
    ProposalCount,
    RandomSeed,
    RelativeGain,
    RepetitionCount,
    ReplicateCount,
    ResampleCount,
    ResearcherCount,
    RetryCount,
    SampleCount,
    ScalabilityBlockPattern,
    ScaleFactor,
    SignificanceLevel,
    SourceLabel,
    StepCount,
    SupportCount,
    ThreadCount,
    Threshold,
    TimeBudgetSeconds,
    TimestampFieldName,
    Tolerance,
    TransferMethod,
    WeightDecay,
    WorkerCount,
)

FrozenModel = DomainModel


class ActionConfig(FrozenModel):
    principal_sparse_support: SupportCount
    sparse_support_sensitivity: tuple[SupportCount, ...]
    total_curriculum_budget: Budget
    coordinate_cap: Budget
    linear_cost_per_actionable_node: Coefficient
    positive_source_value_threshold: Threshold
    maximum_source_proposals_per_target: ProposalCount


class EquivalenceRelativeMacroCe(FrozenModel):
    lower: RelativeGain
    upper: RelativeGain


class MaterialityConfig(FrozenModel):
    coupling_objective_units: Coefficient
    realized_relative_macro_ce: RelativeGain
    macro_f1_absolute: AbsoluteMetric
    equivalence_relative_macro_ce: EquivalenceRelativeMacroCe
    harmful_transfer_relative_macro_ce_gain: RelativeGain
    useful_transfer_relative_macro_ce_gain: RelativeGain


class TransferSupportConfig(FrozenModel):
    source_train_minimum: SampleCount
    source_meta_minimum: SampleCount
    target_meta_minimum: SampleCount
    target_confirm_minimum: SampleCount
    target_test_minimum: SampleCount
    local_prediction_attack_class_total_rows_minimum: SampleCount
    minimum_actionable_target_concepts: ConceptCount
    minimum_nontrivial_block_size: ConceptCount


class ClientConfig(FrozenModel):
    role: ClientRole
    source: SourceLabel
    component: ClientComponentName
    expected_timestamp_field: TimestampFieldName


class TimestampAliasAcceptance(FrozenModel):
    retained_row_parse_success_minimum: Fraction


class DirectedPairSpec(FrozenModel):
    source: DatasetId
    target: DatasetId


class DatasetsConfig(FrozenModel):
    clients: Mapping[DatasetId, ClientConfig]
    timestamp_alias_acceptance: TimestampAliasAcceptance
    primary_directed_pairs: tuple[DirectedPairSpec, ...]
    secondary_directed_pairs: tuple[DirectedPairSpec, ...]
    local_prediction_normal_label: DatasetLabel


class SplitInterval(FrozenModel):
    train: tuple[Fraction, Fraction]
    meta: tuple[Fraction, Fraction]
    valid: tuple[Fraction, Fraction]
    confirm: tuple[Fraction, Fraction]
    test: tuple[Fraction, Fraction]


class SplitConfig(FrozenModel):
    duplicate_safe_chronological_intervals: SplitInterval


class NumericClip(FrozenModel):
    lower: Coefficient
    upper: Coefficient


class PreprocessingConfig(FrozenModel):
    missing_indicator_train_rate_threshold: Threshold
    rare_category_train_frequency_threshold: Threshold
    feature_missing_or_nonfinite_drop_threshold: Threshold
    client_invalidity_dropped_feature_fraction_threshold: Fraction
    numeric_clip: NumericClip
    zero_iqr_replacement_scale: ScaleFactor


class AdamWConfig(FrozenModel):
    beta1: ConfidenceLevel
    beta2: ConfidenceLevel
    epsilon: Tolerance


class EarlyStoppingConfig(FrozenModel):
    patience_completed_epochs: PatienceCount
    minimum_improvement: Tolerance


class CheckpointConfig(FrozenModel):
    tie_tolerance: Tolerance


class TrainingConfig(FrozenModel):
    adamw: AdamWConfig
    maximum_epochs: EpochCount
    batch_size: BatchSize
    gradient_clip_global_l2_norm: ScaleFactor
    early_stopping: EarlyStoppingConfig
    checkpoint: CheckpointConfig
    label_smoothing: Fraction
    dataloader_workers: WorkerCount


class BaseModelPilotConfig(FrozenModel):
    learning_rates: tuple[LearningRate, ...]
    weight_decays: tuple[WeightDecay, ...]
    dropouts: tuple[Fraction, ...]


class SourceResponsePilotConfig(FrozenModel):
    intervention_magnitudes: tuple[InterventionMagnitude, ...]
    optimizer_step_horizons: tuple[StepCount, ...]
    paired_schedules_per_candidate: ReplicateCount
    relative_derivative_discrepancy_ceiling: Fraction
    sign_agreement_minimum: Fraction
    useful_response_magnitude_threshold: Threshold
    minimum_useful_intervention_columns: ConceptCount
    curvature_penalty_coefficient: Coefficient
    numerical_floor: Floor


class SourceResponseFinalConfig(FrozenModel):
    paired_replicates_per_intervention: ReplicateCount
    simultaneous_confidence_level: ConfidenceLevel
    max_t_bootstrap_resamples: ResampleCount
    response_risk_denominator_floor: Floor
    response_standard_error_floor: Floor
    useful_response_magnitude_threshold: Threshold
    minimum_useful_intervention_columns: ConceptCount
    median_band_width_to_median_absolute_mean_response_maximum: ScaleFactor


class TargetResponseDiagnosticConfig(FrozenModel):
    intervention_magnitude: InterventionMagnitude
    shadow_optimizer_steps: StepCount
    paired_replicates: ReplicateCount
    simultaneous_bootstrap_resamples: ResampleCount
    confidence_level: ConfidenceLevel


class ConfirmationConfig(FrozenModel):
    optimizer_steps_per_shadow: StepCount
    paired_replicates: ReplicateCount
    hierarchical_bootstrap_resamples: ResampleCount
    one_sided_confidence_level: ConfidenceLevel
    lower_bound_acceptance_threshold_relative_macro_ce: RelativeGain
    accepted_live_assimilation_steps: StepCount


class ReservedBudgetConfig(FrozenModel):
    target_response_diagnostic: StepCount
    confirmation_candidates: StepCount
    live_assimilation: StepCount
    nontransferable_safety_reserve: StepCount


class TargetOptimizerBudgetConfig(FrozenModel):
    maximum_steps_per_method_pair_seed_before_test: StepCount
    reserved: ReservedBudgetConfig


class PointCorrespondenceBaselineConfig(FrozenModel):
    qap_tie_tolerance: Tolerance


class BaselinesConfig(FrozenModel):
    point_correspondence_commitment: PointCorrespondenceBaselineConfig


class TargetImportanceConfig(FrozenModel):
    class_risk_floor: Floor


class RandomnessConfig(FrozenModel):
    pilot_seeds: tuple[RandomSeed, ...]
    confirmatory_seeds: tuple[RandomSeed, ...]
    statistical_seed: RandomSeed


class StatisticsConfig(FrozenModel):
    confidence_level: ConfidenceLevel
    exact_sign_flip_max_nonzero_differences_for_enumeration: StepCount
    exact_sign_flip_comparison_tolerance: Tolerance
    ci_bootstrap_repetitions: ResampleCount
    identical_difference_tolerance: Tolerance
    minimum_valid_paired_seeds: SampleCount
    tost_alpha_per_one_sided_test: SignificanceLevel
    spearman_minimum_valid_points: SampleCount
    mcnemar_exact_to_asymptotic_discordant_pair_switch: StepCount


class StrictCrossTelemetryUtilityCriteria(FrozenModel):
    successful_primary_pairs_required: SampleCount
    holm_adjusted_p_maximum: SignificanceLevel
    bca_lower_bound_strictly_greater_than: Fraction


class ExternalSourceValueVsLocalSirCriteria(FrozenModel):
    successful_primary_pairs_required: SampleCount
    holm_adjusted_p_maximum: SignificanceLevel
    bca_lower_bound_strictly_greater_than: Fraction


class CouplingMechanismCriteria(FrozenModel):
    theorem_zero_strict_classification_accuracy_required: Fraction
    real_packet_fraction_with_material_gap_minimum: Fraction
    primary_pairs_with_material_mean_gap_required: SampleCount
    holm_adjusted_p_maximum: SignificanceLevel
    destruction_positive_gain_retention_minimum: Fraction


class SparseOperationalRelevanceCriteria(FrozenModel):
    compared_sparse_support: SupportCount
    dense_minus_sparse_gain_maximum: RelativeGain
    valid_unit_fraction_required: Fraction
    primary_pairs_with_useful_gain_required: SampleCount


class ConfirmationSafetyCriteria(FrozenModel):
    qualifying_primary_pairs_required: SampleCount
    absolute_risk_reduction_minimum: Fraction
    relative_risk_reduction_minimum: Fraction
    qualifying_pair_coverage_loss_maximum: Fraction
    pair_harmful_rate_worsening_maximum: Fraction
    pair_coverage_loss_maximum: Fraction
    equal_pair_absolute_risk_reduction_minimum: Fraction
    equal_pair_relative_risk_reduction_minimum: Fraction


class EvaluationCriteriaConfig(FrozenModel):
    strict_cross_telemetry_utility: StrictCrossTelemetryUtilityCriteria
    external_source_value_vs_local_sir: ExternalSourceValueVsLocalSirCriteria
    coupling_mechanism: CouplingMechanismCriteria
    sparse_operational_relevance: SparseOperationalRelevanceCriteria
    confirmation_safety: ConfirmationSafetyCriteria


class MetricsConfig(FrozenModel):
    probability_log_floor: Floor
    relative_macro_ce_denominator_floor: Floor
    relative_solver_error_denominator_floor: Floor


class MultiSourceSelectionConfig(FrozenModel):
    communication_cost_coefficient_in_principal_ranking: Coefficient
    confirmation_cost_coefficient_in_principal_ranking: Coefficient


class RectangularizationIsSufficientRule(FrozenModel):
    valid_real_packet_fraction_below_coupling_materiality_minimum: Fraction


class GenericQapDominatesRule(FrozenModel):
    intended_sparse_support_maximum: SupportCount
    median_runtime_ratio_to_exact_sparse_maximum: RelativeGain
    p95_runtime_ratio_to_exact_sparse_maximum: RelativeGain
    peak_memory_ratio_to_exact_sparse_maximum: RelativeGain


class SparseSupportIsOperationallyIrrelevantRule(FrozenModel):
    dense_gain_advantage_over_support_3_minimum: RelativeGain
    valid_primary_unit_fraction_minimum: Fraction
    sparse_supports_that_must_fail_useful_materiality: tuple[SupportCount, ...]


class PointMatchingIsSufficientRule(FrozenModel):
    harmful_rate_worsening_maximum: Fraction
    utility_advantage_over_fedorbit_minimum: RelativeGain


class StrictInterfaceRemovesGainRule(FrozenModel):
    primary_pair_majority_required: SampleCount
    point_gain_maximum: AbsoluteMetric
    bca_upper_bound_maximum: RelativeGain


class SourceResponseIsTooUnstableRule(FrozenModel):
    principal_source_packet_failure_fraction_strictly_greater_than: Fraction


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
    evaluation_criteria: EvaluationCriteriaConfig
    metrics: MetricsConfig
    multi_source_selection: MultiSourceSelectionConfig
    simplification_rules: SimplificationRulesConfig


class ExactSparseSolverConfig(FrozenModel):
    lp_primal_feasibility_tolerance: Tolerance
    lp_dual_feasibility_tolerance: Tolerance
    lp_optimality_tolerance: Tolerance
    separator_cut_stopping_tolerance: Tolerance
    exact_validation_absolute_tolerance: Tolerance
    permutation_certificate_residual_tolerance: Tolerance
    action_tie_tolerance: Tolerance
    action_tie_comparison_rounding_precision: Tolerance
    lap_objective_tie_tolerance: Tolerance
    maximum_cuts_per_support: CutCount
    lp_threads_per_solve: ThreadCount
    maximum_concurrent_supports: ConcurrencyCount
    deterministic_random_seed: RandomSeed


class GenericExactQapSolverConfig(FrozenModel):
    relative_mip_gap: Tolerance
    feasibility_tolerance: Tolerance
    wall_time_seconds_per_solve: TimeBudgetSeconds
    threads: ThreadCount
    random_seed: RandomSeed


class DenseCcpSolverConfig(FrozenModel):
    penalty_multipliers_relative_to_scale: tuple[Coefficient, ...]
    maximum_iterations_per_penalty_level: StepCount
    assignment_integrality_residual: Tolerance
    relative_objective_convergence_tolerance: Tolerance
    deterministic_starts: StepCount
    outer_action_cuts: CutCount
    wall_time_seconds: TimeBudgetSeconds
    lp_threads: ThreadCount


class SolversConfig(FrozenModel):
    exact_sparse: ExactSparseSolverConfig
    generic_exact_qap: GenericExactQapSolverConfig
    dense_ccp: DenseCcpSolverConfig


class TargetImportanceGamma(FrozenModel):
    shape: Coefficient
    scale: ScaleFactor


class ExactSeparatorTheoremGeneratorConfig(FrozenModel):
    response_uniform: tuple[Coefficient, Coefficient]
    serialization_upper_band_increment_uniform: tuple[Coefficient, Coefficient]
    target_importance_gamma: TargetImportanceGamma
    active_action_uniform: tuple[Coefficient, Coefficient]
    block_patterns: tuple[tuple[ConceptCount, ...], ...]
    supports: tuple[SupportCount, ...]
    generated_instances_per_block_pattern_support_seed_cell: ReplicateCount


class CouplingStructureGeneratorConfig(FrozenModel):
    unconstrained_response_uniform: tuple[Coefficient, Coefficient]
    compatibility: tuple[CouplingCompatibility, ...]
    response_heterogeneity: tuple[Coefficient, ...]
    directed_asymmetry: tuple[Coefficient, ...]
    response_sparsity: tuple[Coefficient, ...]
    block_patterns: tuple[tuple[ConceptCount, ...], ...]
    supports: tuple[SupportCount, ...]
    incompatible_fixed_action_gap_strictly_greater_than: Threshold
    maximum_attempts_per_instance: AttemptCount


class CommonActionUnresolvedMapGeneratorConfig(FrozenModel):
    block_pattern: tuple[ConceptCount, ...]
    block_pair_response_uniform: tuple[Coefficient, Coefficient]
    maximum_attempts: AttemptCount


class RobustCompromiseUnresolvedMapGeneratorConfig(FrozenModel):
    block_pattern: tuple[ConceptCount, ...]
    response_uniform: tuple[Coefficient, Coefficient]
    robust_pre_map_value_strictly_greater_than: Threshold
    maximum_attempts_per_fixture: AttemptCount


class MapDependentGeneratorConfig(FrozenModel):
    block_pattern: tuple[ConceptCount, ...]
    response_uniform: tuple[Coefficient, Coefficient]
    map_value_minimum: Threshold
    maximum_attempts: AttemptCount


class ScalabilityGeneratorConfig(FrozenModel):
    response_uniform: tuple[Coefficient, Coefficient]
    block_patterns: tuple[ScalabilityBlockPattern, ...]


class GeneratorsConfig(FrozenModel):
    exact_separator_theorem: ExactSeparatorTheoremGeneratorConfig
    coupling_structure: CouplingStructureGeneratorConfig
    common_action_unresolved_map: CommonActionUnresolvedMapGeneratorConfig
    robust_compromise_unresolved_map: RobustCompromiseUnresolvedMapGeneratorConfig
    map_dependent: MapDependentGeneratorConfig
    scalability: ScalabilityGeneratorConfig


class MathematicalPrimitiveValidationConfig(FrozenModel):
    hand_fixture_seed: RandomSeed
    fixture_error_tolerance: Tolerance
    invalid_permutations_allowed: InvalidPermutationCount


class SyntheticKRange(FrozenModel):
    minimum: ConceptCount
    maximum: ConceptCount


class ExactSparseSolverBenchmarkConfig(FrozenModel):
    synthetic_k: SyntheticKRange
    block_patterns: tuple[ScalabilityBlockPattern, ...]
    supports: tuple[SupportCount, ...]
    exhaustive_truth_correspondence_count_maximum: StepCount
    methods: tuple[MethodName, ...]


class SyntheticCouplingMechanismValidationConfig(FrozenModel):
    methods: tuple[MethodName, ...]


class CommonActionUnderUnidentifiedMapConfig(FrozenModel):
    fixtures_per_seed: ReplicateCount


class RobustCompromiseUnderUnidentifiedMapConfig(FrozenModel):
    fixtures_per_seed: ReplicateCount


class MapDependentActionBoundaryConfig(FrozenModel):
    fixtures_per_seed: ReplicateCount


class ExactMapValueBoundValidationConfig(FrozenModel):
    zero_map_value_fixtures_per_seed: ReplicateCount
    high_map_value_fixtures_per_seed: ReplicateCount


class PrimaryStrictCrossTelemetryTransferConfig(FrozenModel):
    methods: tuple[MethodName, ...]


class MultiSourceSelectionValidationConfig(FrozenModel):
    targets: tuple[DatasetId, ...]


class MechanismAblationsConfig(FrozenModel):
    methods: tuple[MethodName, ...]


class TargetConfirmationAndPortabilityConfig(FrozenModel):
    methods: tuple[MethodName, ...]


class SecondaryCrossModalityGeneralizationConfig(FrozenModel):
    methods: tuple[MethodName, ...]


class SemanticSufficiencyFrontierConfig(FrozenModel):
    partitions: tuple[str | tuple[str, ...], ...]
    methods: tuple[MethodName, ...]


class WeakSignalSupportAndHeterogeneityBoundariesConfig(FrozenModel):
    response_scales: tuple[ScaleFactor, ...]
    ci_half_width_multipliers: tuple[ScaleFactor, ...]
    target_usable_support_fractions: tuple[Fraction, ...]
    response_heterogeneity_multipliers: tuple[ScaleFactor, ...]
    support_budgets: tuple[SupportCount, ...]
    methods: tuple[MethodName, ...]


class MapAvailabilityApplicabilityAuditConfig(FrozenModel):
    packet_only_recovery_methods: tuple[MethodName, ...]
    independent_researchers: ResearcherCount
    minutes_per_researcher_per_pair: DurationMinutes


class ScalabilityAndEfficiencyConfig(FrozenModel):
    k_values: tuple[ConceptCount, ...]
    block_patterns: tuple[ScalabilityBlockPattern, ...]
    exact_qap_supports: tuple[SupportCount, ...]


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
    retries_after_initial_infrastructure_failure: RetryCount


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
    solver_cpu_worker_ceiling: ConcurrencyCount
    host_ram_ceiling_gib_for_registered_efficiency_runs: GiBMemory
    deterministic_kernel_warmups: RepetitionCount
    deterministic_kernel_timed_repetitions: RepetitionCount
    full_training_timing_repetitions_per_scientific_cell: RepetitionCount
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
    scientific_metric_decimals: DecimalPrecision
    macro_f1_decimals: DecimalPrecision
    balanced_accuracy_decimals: DecimalPrecision
    p_value_decimals: DecimalPrecision
    p_value_less_than_threshold: SignificanceLevel
    runtime_seconds_decimals: DecimalPrecision
    memory_decimals: DecimalPrecision


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


def nominal_alpha() -> float:
    from fedorbit.config.loading import active_config

    return round(1.0 - active_config().scientific.statistics.confidence_level, 10)


def _registered_method_values() -> set[str]:
    return {method.value for method in TransferMethod}


def _append_registered_method(
    methods: list[TransferMethod], candidate_name: str, registered_values: set[str]
) -> None:
    if candidate_name not in registered_values:
        return
    candidate = TransferMethod(candidate_name)
    if candidate not in methods:
        methods.append(candidate)


def all_registered_methods() -> tuple[TransferMethod, ...]:
    from fedorbit.config.loading import active_config

    config = active_config()
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
    registered_values = _registered_method_values()
    methods: list[TransferMethod] = []
    for experiment in experiment_configs:
        for method in experiment.methods:
            _append_registered_method(methods, method, registered_values)
    for (
        method
    ) in config.experiments.map_availability_applicability_audit.packet_only_recovery_methods:
        _append_registered_method(methods, method, registered_values)
    return tuple(methods)


def registered_client_ids() -> tuple[DatasetId, ...]:
    from fedorbit.config.loading import active_config

    return tuple(active_config().scientific.datasets.clients.keys())
