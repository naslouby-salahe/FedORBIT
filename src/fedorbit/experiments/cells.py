from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

from fedorbit.types import EvaluationConditionName, ExperimentName, Index, SemanticCoordinate

CELL_RELEVANCE_BY_EXPERIMENT: Mapping[ExperimentName, frozenset[SemanticCoordinate]] = OrderedDict(
    (
        (
            ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.SEED,
                    SemanticCoordinate.CONDITION,
                    SemanticCoordinate.SUPPORT,
                }
            ),
        ),
        (
            ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.SEED,
                    SemanticCoordinate.CONDITION,
                }
            ),
        ),
        (
            ExperimentName.ROBUST_COMPROMISE_UNDER_UNIDENTIFIED_MAP,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.SEED,
                    SemanticCoordinate.CONDITION,
                }
            ),
        ),
        (
            ExperimentName.MAP_DEPENDENT_ACTION_BOUNDARY,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.SEED,
                    SemanticCoordinate.CONDITION,
                }
            ),
        ),
        (
            ExperimentName.EXACT_MAP_VALUE_BOUND_VALIDATION,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.SEED,
                    SemanticCoordinate.CONDITION,
                }
            ),
        ),
        (
            ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.SEED,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                }
            ),
        ),
        (
            ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.SOURCE_CLIENT,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.SOURCE_RESPONSE_ESTIMATOR_PILOT,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SUPPORT,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.MECHANISM_ABLATIONS,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SUPPORT,
                    SemanticCoordinate.CONDITION,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.SPARSITY_AND_DENSE_FALLBACK,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.CONDITION,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SUPPORT,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.SECONDARY_CROSS_MODALITY_GENERALIZATION,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.SEMANTIC_SUFFICIENCY_FRONTIER,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SUPPORT,
                    SemanticCoordinate.CONDITION,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.WEAK_SIGNAL_SUPPORT_AND_HETEROGENEITY_BOUNDARIES,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SUPPORT,
                    SemanticCoordinate.CONDITION,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.CONDITION,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SUPPORT,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.CONDITION,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SUPPORT,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.MULTI_SOURCE_SELECTION_VALIDATION,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.CONDITION,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.STATISTICAL_SYNTHESIS,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.DATASET,
                    SemanticCoordinate.DIRECTED_PAIR,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.CONDITION,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.EVIDENCE_CLASSIFICATION,
            frozenset({SemanticCoordinate.EXPERIMENT, SemanticCoordinate.SEED}),
        ),
        (
            ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.CONDITION,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
        (
            ExperimentName.SCALABILITY_AND_EFFICIENCY,
            frozenset(
                {
                    SemanticCoordinate.EXPERIMENT,
                    SemanticCoordinate.CONDITION,
                    SemanticCoordinate.METHOD,
                    SemanticCoordinate.SUPPORT,
                    SemanticCoordinate.SEED,
                }
            ),
        ),
    )
)


def experiment_relevance(experiment: ExperimentName) -> frozenset[SemanticCoordinate]:
    return CELL_RELEVANCE_BY_EXPERIMENT[experiment]


ConditionLabel = EvaluationConditionName


@dataclass(frozen=True, slots=True)
class RegisteredCondition:
    labels: tuple[ConditionLabel, ...]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ConditionRegistrationError("registered condition must contain at least one label")


@dataclass(frozen=True, slots=True)
class RegisteredConditions:
    entries: tuple[RegisteredCondition, ...]

    def __post_init__(self) -> None:
        if len(set(self.entries)) != len(self.entries):
            raise ConditionRegistrationError("registered conditions must be distinct")

    def __len__(self) -> Index:
        return len(self.entries)


class ConditionRegistrationError(ValueError):
    pass
