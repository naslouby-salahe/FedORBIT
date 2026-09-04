from __future__ import annotations

import pytest

from fedorbit.config.models import FedorbitConfig
from fedorbit.experiments.catalogue import ExperimentCatalogue, build_catalogue
from fedorbit.types import ExperimentName, TransferMethod


@pytest.fixture(scope="module")
def catalogue() -> ExperimentCatalogue:
    return build_catalogue()


def test_every_registered_experiment_has_a_catalogue_entry(
    catalogue: ExperimentCatalogue,
) -> None:
    expected = set(ExperimentName)
    actual = set(catalogue.registered_names())
    assert actual == expected


def test_every_entry_has_classification(
    catalogue: ExperimentCatalogue,
) -> None:
    for name in catalogue.registered_names():
        assert catalogue.definition(name).classification.value, name


def test_confirmatory_experiments_use_confirmatory_seeds(
    catalogue: ExperimentCatalogue,
    fedorbit_config: FedorbitConfig,
) -> None:
    seeds = fedorbit_config.scientific.randomness.confirmatory_seeds
    for name in catalogue.registered_names():
        definition = catalogue.definition(name)
        if (
            definition.classification.value.startswith("Confirmatory")
            and definition.derived_planned_cells > 0
        ):
            assert definition.seeds == seeds, name


def test_primary_transfer_derived_cells(
    catalogue: ExperimentCatalogue,
    fedorbit_config: FedorbitConfig,
) -> None:
    definition = catalogue.definition(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    expected = (
        len(fedorbit_config.scientific.datasets.primary_directed_pairs)
        * 10
        * len(fedorbit_config.experiments.primary_strict_cross_telemetry_transfer.methods)
    )
    assert definition.derived_planned_cells == expected
    assert definition.derived_planned_cells == 420


def test_theorem_validation_derived_cells(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION)
    assert definition.derived_planned_cells == 17000


def test_coupling_validation_derived_cells(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION)
    assert definition.derived_planned_cells == 4860


def test_mechanism_ablations_derived_cells(
    catalogue: ExperimentCatalogue,
    fedorbit_config: FedorbitConfig,
) -> None:
    definition = catalogue.definition(ExperimentName.MECHANISM_ABLATIONS)
    assert definition.derived_planned_cells == 6 * 10 * len(
        fedorbit_config.experiments.mechanism_ablations.methods
    )
    assert definition.derived_planned_cells == 480


def test_sparsity_and_dense_derived_cells(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(ExperimentName.SPARSITY_AND_DENSE_FALLBACK)
    assert definition.derived_planned_cells == 240


def test_confirmation_portability_derived_cells(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY)
    assert definition.derived_planned_cells == 120


def test_secondary_generalization_derived_cells(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(ExperimentName.SECONDARY_CROSS_MODALITY_GENERALIZATION)
    assert definition.derived_planned_cells == 0


def test_semantic_frontier_derived_cells(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(ExperimentName.SEMANTIC_SUFFICIENCY_FRONTIER)
    assert definition.derived_planned_cells == 720


def test_weak_signal_boundaries_derived_cells(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(
        ExperimentName.WEAK_SIGNAL_SUPPORT_AND_HETEROGENEITY_BOUNDARIES
    )
    assert definition.derived_planned_cells == 2700


def test_scalability_derived_cells(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(ExperimentName.SCALABILITY_AND_EFFICIENCY)
    assert definition.derived_planned_cells == 1120 + 120


def test_map_audit_recovery_attempts(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT)
    assert definition.derived_planned_cells == 120


def test_multi_source_target_decisions(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(ExperimentName.MULTI_SOURCE_SELECTION_VALIDATION)
    assert definition.derived_planned_cells == 30


def test_diagnostic_fixture_counts(
    catalogue: ExperimentCatalogue,
) -> None:
    for name in (
        ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP,
        ExperimentName.ROBUST_COMPROMISE_UNDER_UNIDENTIFIED_MAP,
        ExperimentName.MAP_DEPENDENT_ACTION_BOUNDARY,
    ):
        assert catalogue.definition(name).derived_planned_cells == 500, name
    assert (
        catalogue.definition(ExperimentName.EXACT_MAP_VALUE_BOUND_VALIDATION).derived_planned_cells
        == 500
    )


def test_benchmark_methods_include_all_solvers(
    catalogue: ExperimentCatalogue,
) -> None:
    methods = set(catalogue.definition(ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK).methods)
    assert TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value in methods
    assert TransferMethod.GENERIC_EXACT_QAP.value in methods
    assert TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK.value in methods


def test_primary_transfer_method_membership(
    catalogue: ExperimentCatalogue,
    fedorbit_config: FedorbitConfig,
) -> None:
    methods = catalogue.definition(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER).methods
    assert (
        tuple(methods)
        == fedorbit_config.experiments.primary_strict_cross_telemetry_transfer.methods
    )
    assert TransferMethod.LOCAL_ONLY.value in methods
    assert TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value in methods
    assert TransferMethod.EXACT_MAP_ORACLE.value in methods


def test_pilot_uses_pilot_seeds(
    catalogue: ExperimentCatalogue,
    fedorbit_config: FedorbitConfig,
) -> None:
    pilot = catalogue.definition(ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT)
    assert fedorbit_config.scientific.randomness.pilot_seeds == (101, 202, 303)
    assert pilot.derived_planned_cells == 144 + 40


def test_source_response_pilot_candidate_cells(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(ExperimentName.SOURCE_RESPONSE_ESTIMATOR_PILOT)
    assert definition.derived_planned_cells == 108


def test_final_packets_planned(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION)
    assert definition.derived_planned_cells == 40


def test_baseline_validation_seeds(
    catalogue: ExperimentCatalogue,
) -> None:
    definition = catalogue.definition(ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION)
    assert definition.seeds == (1103, 5531)
    assert definition.derived_planned_cells == 12


def test_semantic_frontier_partitions_registered(
    catalogue: ExperimentCatalogue,
    fedorbit_config: FedorbitConfig,
) -> None:
    definition = catalogue.definition(ExperimentName.SEMANTIC_SUFFICIENCY_FRONTIER)
    partitions = fedorbit_config.experiments.semantic_sufficiency_frontier.partitions
    assert len(definition.conditions) == len(partitions)
    assert len(partitions) == 4


def test_classifications_match_roadmap(
    catalogue: ExperimentCatalogue,
) -> None:
    assert (
        catalogue.definition(ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION).classification.value
        == "Validation"
    )
    assert (
        catalogue.definition(ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT).classification.value
        == "Exploratory"
    )
    assert (
        catalogue.definition(ExperimentName.MAP_DEPENDENT_ACTION_BOUNDARY).classification.value
        == "Failure Boundary"
    )
    assert (
        catalogue.definition(ExperimentName.STATISTICAL_SYNTHESIS).classification.value
        == "Confirmatory ANALYSIS"
    )
