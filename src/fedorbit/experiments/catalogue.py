from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

from fedorbit.config.loading import active_config
from fedorbit.experiments.cells import (
    ConditionLabel,
    RegisteredCondition,
    RegisteredConditions,
)
from fedorbit.types import ExperimentClassification, ExperimentName, TransferMethod

_PRIMARY_PAIRS_LABEL = "four primary directed pairs"
_SECONDARY_PAIRS_LABEL = "four secondary directed pairs"


def _experiment_name(name: ExperimentName) -> str:
    return name.value


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    name: ExperimentName
    classification: ExperimentClassification
    methods: tuple[str, ...]
    datasets_or_pairs: tuple[str, ...]
    conditions: RegisteredConditions
    seeds: tuple[int, ...]
    derived_planned_cells: int
    prerequisites: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExperimentCatalogue:
    definitions_by_name: Mapping[ExperimentName, ExperimentDefinition]

    def definition(self, name: ExperimentName) -> ExperimentDefinition:
        definition = self.definitions_by_name.get(name)
        if definition is None:
            raise CatalogueError(f"unregistered experiment: {name.value}")
        return definition

    def registered_names(self) -> tuple[ExperimentName, ...]:
        return tuple(self.definitions_by_name.keys())

    def __len__(self) -> int:
        return len(self.definitions_by_name)


class CatalogueError(KeyError):
    pass


def build_catalogue() -> ExperimentCatalogue:
    config = active_config()
    confirmatory_seeds = config.scientific.randomness.confirmatory_seeds
    pilot_seeds = config.scientific.randomness.pilot_seeds
    primary_pair_count = len(config.scientific.datasets.primary_directed_pairs)
    secondary_pair_count = len(config.scientific.datasets.secondary_directed_pairs)
    experiments = config.experiments
    methods = experiments.primary_strict_cross_telemetry_transfer.methods

    def definition(
        name: ExperimentName,
        classification: ExperimentClassification,
        method_names: tuple[str, ...],
        pairs: tuple[str, ...],
        conditions: tuple[str | tuple[str, ...], ...],
        seeds: tuple[int, ...],
        derived_cells: int,
        prerequisites: tuple[str, ...],
    ) -> ExperimentDefinition:
        return ExperimentDefinition(
            name=name,
            classification=classification,
            methods=method_names,
            datasets_or_pairs=pairs,
            conditions=RegisteredConditions(
                tuple(
                    RegisteredCondition(
                        tuple(ConditionLabel(label) for label in entry)
                        if isinstance(entry, tuple)
                        else (ConditionLabel(entry),)
                    )
                    for entry in conditions
                )
            ),
            seeds=seeds,
            derived_planned_cells=derived_cells,
            prerequisites=prerequisites,
        )

    catalogue: OrderedDict[ExperimentName, ExperimentDefinition] = OrderedDict()

    catalogue[ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION] = definition(
        ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION,
        ExperimentClassification.VALIDATION,
        (),
        (),
        ("hand fixtures",),
        (0,),
        0,
        ("primitive implementation",),
    )

    block_patterns = config.generators.exact_separator_theorem.block_patterns
    theorem_generator = config.generators.exact_separator_theorem
    instances_per_cell = theorem_generator.generated_instances_per_block_pattern_support_seed_cell
    feasible_support_cells = sum(min(sum(pattern), 3) for pattern in block_patterns)
    theorem_cells = feasible_support_cells * len(confirmatory_seeds) * instances_per_cell
    catalogue[ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION] = definition(
        ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION,
        ExperimentClassification.VALIDATION,
        ("exhaustive orbit", TransferMethod.GENERIC_EXACT_QAP.value),
        (),
        (),
        confirmatory_seeds,
        theorem_cells,
        (_experiment_name(ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION),),
    )

    coupling_factorial = (
        len(config.generators.coupling_structure.compatibility)
        * len(config.generators.coupling_structure.response_heterogeneity)
        * len(config.generators.coupling_structure.directed_asymmetry)
        * len(config.generators.coupling_structure.response_sparsity)
        * len(config.generators.coupling_structure.block_patterns)
        * len(config.generators.coupling_structure.supports)
        * len(confirmatory_seeds)
    )
    catalogue[ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION] = definition(
        ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION,
        ExperimentClassification.VALIDATION,
        (),
        (),
        (),
        confirmatory_seeds,
        coupling_factorial,
        (_experiment_name(ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION),),
    )

    client_count = len(config.scientific.datasets.clients)
    catalogue[ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION] = definition(
        ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION,
        ExperimentClassification.VALIDATION,
        (),
        ("all primary and secondary pairs",),
        (),
        confirmatory_seeds,
        client_count * (primary_pair_count + secondary_pair_count) * len(confirmatory_seeds),
        ("raw manifests",),
    )

    pilot_configs = (
        len(config.scientific.base_model_pilot.learning_rates)
        * len(config.scientific.base_model_pilot.weight_decays)
        * len(config.scientific.base_model_pilot.dropouts)
    )
    pilot_fits = client_count * pilot_configs * len(pilot_seeds)
    confirmatory_checkpoints = client_count * len(confirmatory_seeds)
    catalogue[ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT] = definition(
        ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT,
        ExperimentClassification.EXPLORATORY,
        (),
        (),
        (),
        (*pilot_seeds, *confirmatory_seeds),
        pilot_fits + confirmatory_checkpoints,
        (_experiment_name(ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION),),
    )

    response_candidates = len(
        config.scientific.source_response_pilot.intervention_magnitudes
    ) * len(config.scientific.source_response_pilot.optimizer_step_horizons)
    catalogue[ExperimentName.SOURCE_RESPONSE_ESTIMATOR_PILOT] = definition(
        ExperimentName.SOURCE_RESPONSE_ESTIMATOR_PILOT,
        ExperimentClassification.EXPLORATORY,
        (),
        (),
        (),
        pilot_seeds,
        client_count * len(pilot_seeds) * response_candidates,
        (
            _experiment_name(ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT),
            _experiment_name(ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION),
        ),
    )

    planned_packets = client_count * len(confirmatory_seeds)
    catalogue[ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION] = definition(
        ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION,
        ExperimentClassification.VALIDATION,
        (),
        (),
        (),
        confirmatory_seeds,
        planned_packets,
        (_experiment_name(ExperimentName.SOURCE_RESPONSE_ESTIMATOR_PILOT),),
    )

    validation_seeds = (confirmatory_seeds[0], confirmatory_seeds[4])
    catalogue[ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION] = definition(
        ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION,
        ExperimentClassification.VALIDATION,
        (
            TransferMethod.LOCAL_ONLY.value,
            TransferMethod.LOCAL_SIR.value,
            TransferMethod.MATCHED_RESOURCE_RECTANGULAR.value,
            TransferMethod.POINT_CORRESPONDENCE_COMMITMENT.value,
            TransferMethod.GENERIC_EXACT_QAP.value,
            TransferMethod.EXACT_MAP_ORACLE.value,
        ),
        (),
        (),
        validation_seeds,
        primary_pair_count * len(validation_seeds),
        (_experiment_name(ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION),),
    )

    benchmark_k = (
        experiments.exact_sparse_solver_benchmark.synthetic_k.maximum
        - experiments.exact_sparse_solver_benchmark.synthetic_k.minimum
        + 1
    )
    benchmark_cells = (
        benchmark_k
        * len(experiments.exact_sparse_solver_benchmark.block_patterns)
        * len(experiments.exact_sparse_solver_benchmark.supports)
        * len(confirmatory_seeds)
    )
    catalogue[ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK] = definition(
        ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK,
        ExperimentClassification.CONFIRMATORY,
        experiments.exact_sparse_solver_benchmark.methods,
        (),
        (),
        confirmatory_seeds,
        benchmark_cells,
        (_experiment_name(ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION),),
    )

    catalogue[ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION] = definition(
        ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION,
        ExperimentClassification.CONFIRMATORY_MECHANISM,
        experiments.synthetic_coupling_mechanism_validation.methods,
        (),
        (),
        confirmatory_seeds,
        coupling_factorial * len(experiments.synthetic_coupling_mechanism_validation.methods),
        (_experiment_name(ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION),),
    )

    catalogue[ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION] = definition(
        ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION,
        ExperimentClassification.CONFIRMATORY_MECHANISM,
        ("exact orbit", TransferMethod.MATCHED_RESOURCE_RECTANGULAR.value),
        ("primary pairs",),
        (),
        confirmatory_seeds,
        primary_pair_count * len(confirmatory_seeds),
        (
            _experiment_name(ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION),
            _experiment_name(ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION),
        ),
    )

    diagnostic_fixtures = experiments.common_action_under_unidentified_map.fixtures_per_seed * len(
        confirmatory_seeds
    )
    catalogue[ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP] = definition(
        ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP,
        ExperimentClassification.DIAGNOSTIC,
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value,),
        (),
        (),
        confirmatory_seeds,
        diagnostic_fixtures,
        (_experiment_name(ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION),),
    )

    catalogue[ExperimentName.ROBUST_COMPROMISE_UNDER_UNIDENTIFIED_MAP] = definition(
        ExperimentName.ROBUST_COMPROMISE_UNDER_UNIDENTIFIED_MAP,
        ExperimentClassification.DIAGNOSTIC,
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value,),
        (),
        (),
        confirmatory_seeds,
        diagnostic_fixtures,
        (_experiment_name(ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION),),
    )

    catalogue[ExperimentName.MAP_DEPENDENT_ACTION_BOUNDARY] = definition(
        ExperimentName.MAP_DEPENDENT_ACTION_BOUNDARY,
        ExperimentClassification.FAILURE_BOUNDARY,
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value,),
        (),
        (),
        confirmatory_seeds,
        diagnostic_fixtures,
        (_experiment_name(ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION),),
    )

    bound_fixtures = (
        experiments.exact_map_value_bound_validation.zero_map_value_fixtures_per_seed
        + experiments.exact_map_value_bound_validation.high_map_value_fixtures_per_seed
    ) * len(confirmatory_seeds)
    catalogue[ExperimentName.EXACT_MAP_VALUE_BOUND_VALIDATION] = definition(
        ExperimentName.EXACT_MAP_VALUE_BOUND_VALIDATION,
        ExperimentClassification.VALIDATION,
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value,),
        (),
        (),
        confirmatory_seeds,
        bound_fixtures,
        (_experiment_name(ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION),),
    )

    catalogue[ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER] = definition(
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        ExperimentClassification.CONFIRMATORY,
        methods,
        (_PRIMARY_PAIRS_LABEL,),
        (),
        confirmatory_seeds,
        primary_pair_count * len(confirmatory_seeds) * len(methods),
        (
            _experiment_name(ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION),
            _experiment_name(ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION),
        ),
    )

    multi_source_targets = len(experiments.multi_source_selection_validation.targets)
    catalogue[ExperimentName.MULTI_SOURCE_SELECTION_VALIDATION] = definition(
        ExperimentName.MULTI_SOURCE_SELECTION_VALIDATION,
        ExperimentClassification.DIAGNOSTIC,
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value,),
        ("target clients",),
        (),
        confirmatory_seeds,
        multi_source_targets * len(confirmatory_seeds),
        (_experiment_name(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER),),
    )

    ablation_methods = experiments.mechanism_ablations.methods
    catalogue[ExperimentName.MECHANISM_ABLATIONS] = definition(
        ExperimentName.MECHANISM_ABLATIONS,
        ExperimentClassification.ABLATION,
        ablation_methods,
        (_PRIMARY_PAIRS_LABEL,),
        (),
        confirmatory_seeds,
        primary_pair_count * len(confirmatory_seeds) * len(ablation_methods),
        (_experiment_name(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER),),
    )

    sparse_conditions = 4
    catalogue[ExperimentName.SPARSITY_AND_DENSE_FALLBACK] = definition(
        ExperimentName.SPARSITY_AND_DENSE_FALLBACK,
        ExperimentClassification.ROBUSTNESS,
        (
            "exact sparse s=1",
            "exact sparse s=2",
            "exact sparse s=3",
            TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK.value,
        ),
        (_PRIMARY_PAIRS_LABEL,),
        (),
        confirmatory_seeds,
        primary_pair_count * len(confirmatory_seeds) * sparse_conditions,
        (_experiment_name(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER),),
    )

    confirmation_methods = experiments.target_confirmation_and_portability.methods
    catalogue[ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY] = definition(
        ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY,
        ExperimentClassification.CONFIRMATORY_SAFETY,
        confirmation_methods,
        ("primary and secondary pairs",),
        (),
        confirmatory_seeds,
        (primary_pair_count + secondary_pair_count)
        * len(confirmatory_seeds)
        * len(confirmation_methods),
        (_experiment_name(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER),),
    )

    secondary_methods = experiments.secondary_cross_modality_generalization.methods
    catalogue[ExperimentName.SECONDARY_CROSS_MODALITY_GENERALIZATION] = definition(
        ExperimentName.SECONDARY_CROSS_MODALITY_GENERALIZATION,
        ExperimentClassification.GENERALIZATION,
        secondary_methods,
        (_SECONDARY_PAIRS_LABEL,),
        (),
        confirmatory_seeds,
        secondary_pair_count * len(confirmatory_seeds) * len(secondary_methods),
        (_experiment_name(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER),),
    )

    frontier_partitions = len(experiments.semantic_sufficiency_frontier.partitions)
    frontier_methods = experiments.semantic_sufficiency_frontier.methods
    catalogue[ExperimentName.SEMANTIC_SUFFICIENCY_FRONTIER] = definition(
        ExperimentName.SEMANTIC_SUFFICIENCY_FRONTIER,
        ExperimentClassification.FAILURE_BOUNDARY,
        frontier_methods,
        (_PRIMARY_PAIRS_LABEL,),
        experiments.semantic_sufficiency_frontier.partitions,
        confirmatory_seeds,
        primary_pair_count * frontier_partitions * len(frontier_methods) * len(confirmatory_seeds),
        (_experiment_name(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER),),
    )

    weak_signal = experiments.weak_signal_support_and_heterogeneity_boundaries
    weak_conditions = (
        len(weak_signal.response_scales)
        + len(weak_signal.ci_half_width_multipliers)
        + len(weak_signal.target_usable_support_fractions)
        + len(weak_signal.response_heterogeneity_multipliers)
        + len(weak_signal.support_budgets)
        - 4
    )
    catalogue[ExperimentName.WEAK_SIGNAL_SUPPORT_AND_HETEROGENEITY_BOUNDARIES] = definition(
        ExperimentName.WEAK_SIGNAL_SUPPORT_AND_HETEROGENEITY_BOUNDARIES,
        ExperimentClassification.FAILURE_BOUNDARY,
        weak_signal.methods,
        (_PRIMARY_PAIRS_LABEL,),
        (),
        confirmatory_seeds,
        weak_conditions * len(weak_signal.methods) * primary_pair_count * len(confirmatory_seeds),
        (_experiment_name(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER),),
    )

    recovery_methods = experiments.map_availability_applicability_audit.packet_only_recovery_methods
    recovery_attempts = len(recovery_methods) * primary_pair_count * len(confirmatory_seeds)
    catalogue[ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT] = definition(
        ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT,
        ExperimentClassification.DIAGNOSTIC,
        recovery_methods,
        (_PRIMARY_PAIRS_LABEL,),
        (),
        confirmatory_seeds,
        recovery_attempts,
        (_experiment_name(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER),),
    )

    scalability = experiments.scalability_and_efficiency
    scalability_exact_cells = (
        len(scalability.k_values)
        * len(scalability.block_patterns)
        * len(scalability.exact_qap_supports)
        * len(confirmatory_seeds)
        * 2
    )
    scalability_dense_cells = (
        len(scalability.k_values) * len(scalability.block_patterns) * len(confirmatory_seeds)
    )
    real_timing_cells = planned_packets * 3
    catalogue[ExperimentName.SCALABILITY_AND_EFFICIENCY] = definition(
        ExperimentName.SCALABILITY_AND_EFFICIENCY,
        ExperimentClassification.ROBUSTNESS,
        (
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value,
            TransferMethod.GENERIC_EXACT_QAP.value,
            TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK.value,
        ),
        (),
        (),
        confirmatory_seeds,
        scalability_exact_cells + scalability_dense_cells + real_timing_cells,
        (_experiment_name(ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK),),
    )

    catalogue[ExperimentName.STATISTICAL_SYNTHESIS] = definition(
        ExperimentName.STATISTICAL_SYNTHESIS,
        ExperimentClassification.CONFIRMATORY_ANALYSIS,
        (),
        (),
        (),
        (),
        0,
        ("completed registered artifacts",),
    )

    completed = ExperimentCatalogue(catalogue)
    from fedorbit.experiments.catalogue import validate_catalogue

    validate_catalogue(completed)
    return completed


class ExperimentValidationError(ValueError):
    pass


def validate_catalogue(catalogue: ExperimentCatalogue) -> None:
    registered = catalogue.registered_names()
    if set(registered) != set(ExperimentName):
        raise ExperimentValidationError("catalogue must define every registered experiment")
    if len(registered) != len(set(registered)):
        raise ExperimentValidationError("catalogue registers an experiment more than once")
    for definition in (catalogue.definition(name) for name in registered):
        if definition.derived_planned_cells < 0:
            raise ExperimentValidationError(
                f"experiment has negative planned cell count: {definition.name.value}"
            )
        if definition.derived_planned_cells > 0 and not definition.seeds:
            raise ExperimentValidationError(
                f"executable experiment has no registered seeds: {definition.name.value}"
            )
        if any(not prerequisite for prerequisite in definition.prerequisites):
            raise ExperimentValidationError(
                f"experiment has an empty prerequisite: {definition.name.value}"
            )
