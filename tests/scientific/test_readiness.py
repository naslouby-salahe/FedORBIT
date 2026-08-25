from __future__ import annotations

from fedorbit.config.loading import snapshot_matches_contract
from fedorbit.config.models import FedorbitConfig
from fedorbit.experiments.catalogue import build_catalogue

GOVERNED_VALUE_PROBES = (
    ("scientific", "action", "principal_sparse_support"),
    ("scientific", "action", "total_curriculum_budget"),
    ("scientific", "materiality", "realized_relative_macro_ce"),
    ("scientific", "transfer_support", "source_train_minimum"),
    ("scientific", "split", "duplicate_safe_chronological_intervals", "train"),
    ("scientific", "preprocessing", "feature_missing_or_nonfinite_drop_threshold"),
    ("scientific", "training", "maximum_epochs"),
    ("scientific", "base_model_pilot", "learning_rates"),
    ("scientific", "source_response_final", "paired_replicates_per_intervention"),
    ("scientific", "confirmation", "lower_bound_acceptance_threshold_relative_macro_ce"),
    ("scientific", "target_optimizer_budget", "maximum_steps_per_method_pair_seed_before_test"),
    ("scientific", "target_importance", "class_risk_floor"),
    ("scientific", "randomness", "confirmatory_seeds"),
    ("scientific", "statistics", "ci_bootstrap_repetitions"),
    (
        "scientific",
        "evaluation_criteria",
        "strict_cross_telemetry_utility",
        "successful_primary_pairs_required",
    ),
    ("scientific", "metrics", "probability_log_floor"),
    (
        "scientific",
        "simplification_rules",
        "source_response_is_too_unstable",
        "principal_source_packet_failure_fraction_strictly_greater_than",
    ),
    ("solvers", "exact_sparse", "maximum_cuts_per_support"),
    ("solvers", "dense_ccp", "deterministic_starts"),
    (
        "generators",
        "exact_separator_theorem",
        "generated_instances_per_block_pattern_support_seed_cell",
    ),
    ("experiments", "scalability_and_efficiency", "k_values"),
    ("runtime", "failure_handling", "retries_after_initial_infrastructure_failure"),
    ("runtime", "artifact_layout", "execution_root"),
    ("environment", "python"),
    ("reporting", "precision", "p_value_less_than_threshold"),
)


def test_readiness_gate_typed_contract_matches_roadmap(fedorbit_config: FedorbitConfig) -> None:
    assert snapshot_matches_contract(fedorbit_config)


def test_readiness_gate_all_governed_values_bound(fedorbit_config: FedorbitConfig) -> None:
    for probe in GOVERNED_VALUE_PROBES:
        current: object = fedorbit_config
        for key in probe:
            assert hasattr(current, key), f"missing governed value: {'.'.join(probe)}"
            current = getattr(current, key)
        assert current is not None, f"unbound governed value: {'.'.join(probe)}"


def test_readiness_gate_catalogue_expansion_is_deterministic(
    fedorbit_config: FedorbitConfig,
) -> None:
    first = build_catalogue(fedorbit_config)
    second = build_catalogue(fedorbit_config)
    assert first == second
    assert len(first) == len(second)


def test_readiness_gate_no_placeholder_values(fedorbit_config: FedorbitConfig) -> None:
    rendered = fedorbit_config.model_dump(mode="json")
    assert "TODO" not in str(rendered)
    assert "FIXME" not in str(rendered)
    assert "PLACEHOLDER" not in str(rendered).upper()


def test_readiness_gate_seed_sets_complete(fedorbit_config: FedorbitConfig) -> None:
    seeds = fedorbit_config.scientific.randomness
    assert len(seeds.confirmatory_seeds) == 10
    assert len(seeds.pilot_seeds) == 3
    assert seeds.statistical_seed == 300
