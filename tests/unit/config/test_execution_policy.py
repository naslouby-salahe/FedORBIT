from __future__ import annotations

import pytest
from tests.typed_access import ConfigDocument

from fedorbit.config.models import FedorbitConfig
from fedorbit.config.validation import ConfigurationContractError, validate_cross_field_contract


def _validate_raw(config: ConfigDocument) -> FedorbitConfig:
    model = FedorbitConfig.model_validate(config.as_dict())
    validate_cross_field_contract(model)
    return model


def test_rectangularization_is_sufficient_rule_locked(fedorbit_config: FedorbitConfig) -> None:
    rule = fedorbit_config.scientific.simplification_rules.rectangularization_is_sufficient
    assert rule.valid_real_packet_fraction_below_coupling_materiality_minimum == 0.9


def test_generic_qap_dominates_rule_locked(fedorbit_config: FedorbitConfig) -> None:
    rule = fedorbit_config.scientific.simplification_rules.generic_qap_dominates
    assert rule.intended_sparse_support_maximum == 2
    assert rule.median_runtime_ratio_to_exact_sparse_maximum == 1.0
    assert rule.p95_runtime_ratio_to_exact_sparse_maximum == 1.2
    assert rule.peak_memory_ratio_to_exact_sparse_maximum == 1.0


def test_sparse_irrelevance_rule_locked(fedorbit_config: FedorbitConfig) -> None:
    rule = (
        fedorbit_config.scientific.simplification_rules.sparse_support_is_operationally_irrelevant
    )
    assert rule.dense_gain_advantage_over_support_3_minimum == 0.02
    assert rule.valid_primary_unit_fraction_minimum == 0.75
    assert rule.sparse_supports_that_must_fail_useful_materiality == (1, 2, 3)


def test_point_matching_rule_locked(fedorbit_config: FedorbitConfig) -> None:
    rule = fedorbit_config.scientific.simplification_rules.point_matching_is_sufficient
    assert rule.harmful_rate_worsening_maximum == 0.02
    assert rule.utility_advantage_over_fedorbit_minimum == 0.01


def test_strict_interface_removes_gain_rule_locked(fedorbit_config: FedorbitConfig) -> None:
    rule = fedorbit_config.scientific.simplification_rules.strict_interface_removes_gain
    assert rule.primary_pair_majority_required == 3
    assert rule.point_gain_maximum == 0.0
    assert rule.bca_upper_bound_maximum == 0.01


def test_source_response_instability_rule_locked(fedorbit_config: FedorbitConfig) -> None:
    rule = fedorbit_config.scientific.simplification_rules.source_response_is_too_unstable
    assert rule.principal_source_packet_failure_fraction_strictly_greater_than == 0.5


def test_infrastructure_retry_count_locked(fedorbit_config: FedorbitConfig) -> None:
    assert (
        fedorbit_config.runtime.failure_handling.retries_after_initial_infrastructure_failure == 2
    )


def test_artifact_roots_locked(fedorbit_config: FedorbitConfig) -> None:
    layout = fedorbit_config.runtime.artifact_layout
    assert layout.execution_root == "outputs"
    assert layout.manuscript_root == "results"


def test_preprocessing_subdirectories_locked(fedorbit_config: FedorbitConfig) -> None:
    layout = fedorbit_config.runtime.artifact_layout
    assert layout.preprocessing_subdirectories == (
        "inventories",
        "validation",
        "prepared",
        "splits",
        "features",
        "metadata",
    )


def test_reusable_artifact_subdirectories_locked(fedorbit_config: FedorbitConfig) -> None:
    layout = fedorbit_config.runtime.artifact_layout
    assert layout.reusable_artifact_subdirectories == (
        "models",
        "scores",
        "fitted",
        "baselines",
        "derived",
    )


def test_experiment_subdirectories_locked(fedorbit_config: FedorbitConfig) -> None:
    subdirs = fedorbit_config.runtime.artifact_layout.experiment_subdirectories
    assert subdirs.artifacts == ("fitted", "predictions", "derived")
    assert subdirs.evaluations == ("records", "comparisons", "aggregates")
    assert subdirs.metrics == ("per_seed", "per_condition", "aggregate")
    assert subdirs.statistics == ("tests", "confidence_intervals", "effects", "multiplicity")
    assert subdirs.checkpoints == ("training", "execution")
    assert subdirs.diagnostics == ("scientific", "numerical", "runtime")
    assert subdirs.logs == ("execution", "failures")
    assert subdirs.provenance == (
        "configuration",
        "data",
        "seeds",
        "code",
        "environment",
        "dependencies",
    )


def test_cache_subdirectories_locked(fedorbit_config: FedorbitConfig) -> None:
    layout = fedorbit_config.runtime.artifact_layout
    assert layout.cache_subdirectories == (
        "preprocessing",
        "models",
        "evaluation",
        "analysis",
        "staging",
    )


def test_manuscript_experiment_subdirectories_locked(fedorbit_config: FedorbitConfig) -> None:
    subdirs = fedorbit_config.runtime.artifact_layout.manuscript_experiment_subdirectories
    assert subdirs.figures == ("main", "supplementary")
    assert subdirs.tables == ("main", "supplementary")
    assert subdirs.metrics == ("primary", "secondary", "summary")
    assert subdirs.statistics == ("tests", "confidence_intervals", "effects", "multiplicity")


def test_project_summary_subdirectories_locked(fedorbit_config: FedorbitConfig) -> None:
    subdirs = fedorbit_config.runtime.artifact_layout.project_summary_subdirectories
    assert subdirs.figures == ("main", "supplementary")
    assert subdirs.tables == ("main", "supplementary")
    assert subdirs.metrics == ("primary", "summary")
    assert subdirs.statistics == ("comparisons", "confidence_intervals", "effects", "multiplicity")
    assert subdirs.reproducibility == (
        "configuration",
        "datasets",
        "seeds",
        "software",
        "execution",
    )


def test_runtime_layout_values_locked(fedorbit_config: FedorbitConfig) -> None:
    runtime = fedorbit_config.runtime
    assert runtime.reference_model_gpu == "NVIDIA GeForce RTX 5060 Ti 16 GB"
    assert runtime.solver_cpu_worker_ceiling == 4
    assert runtime.host_ram_ceiling_gib_for_registered_efficiency_runs == 16
    assert runtime.deterministic_kernel_warmups == 3
    assert runtime.deterministic_kernel_timed_repetitions == 10
    assert runtime.full_training_timing_repetitions_per_scientific_cell == 1


def test_falsification_fraction_must_be_valid(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value(
        "scientific",
        "simplification_rules",
        "rectangularization_is_sufficient",
        "valid_real_packet_fraction_below_coupling_materiality_minimum",
        value=1.5,
    )
    with pytest.raises(ConfigurationContractError):
        _validate_raw(mutable_config)


def test_retry_count_must_be_nonnegative(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value(
        "runtime", "failure_handling", "retries_after_initial_infrastructure_failure", value=-1
    )
    with pytest.raises(ConfigurationContractError):
        _validate_raw(mutable_config)


def test_artifact_root_cannot_drift(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value("runtime", "artifact_layout", "manuscript_root", value="artifacts")
    with pytest.raises(ConfigurationContractError):
        _validate_raw(mutable_config)
