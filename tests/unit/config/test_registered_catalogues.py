from __future__ import annotations

from fedorbit.config.models import FedorbitConfig
from fedorbit.types import MetricId


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
    assert weak.ci_half_width_multipliers == (1.0, 1.5, 2.0, 4.0)
    assert weak.target_usable_support_fractions == (1.0, 0.5, 0.25, 0.1)
    assert weak.response_heterogeneity_multipliers == (0.5, 1.0, 2.0)
    assert weak.support_budgets == (1, 2, 3)


def test_metric_catalogue_identities() -> None:
    assert MetricId.MACRO_CROSS_ENTROPY == "Macro Cross-Entropy"
    assert MetricId.RELATIVE_MACRO_CE_GAIN == "Relative Macro-CE Gain"
    assert MetricId.ACTIVE_IMAGE_CANDIDATES == "Active-Image Candidates"
    assert MetricId.LAP_CALLS == "LAP Calls"
    assert MetricId.DENSE_BOUND_GAP == "Dense Bound Gap"
    assert MetricId.ABSOLUTE_RISK_REDUCTION == "Absolute Risk Reduction"
    assert MetricId.RELATIVE_RISK_REDUCTION == "Relative Risk Reduction"
    assert len(tuple(MetricId)) >= 40
