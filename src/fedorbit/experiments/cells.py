from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

from fedorbit.types import ExperimentName, SemanticCoordinate


def experiment_relevance(experiment: ExperimentName) -> frozenset[SemanticCoordinate]:
    common = frozenset({SemanticCoordinate.EXPERIMENT, SemanticCoordinate.SEED})
    by_experiment: Mapping[ExperimentName, frozenset[SemanticCoordinate]] = OrderedDict(
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
                        SemanticCoordinate.SUPPORT,
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
                        SemanticCoordinate.DATASET,
                        SemanticCoordinate.DIRECTED_PAIR,
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
                        SemanticCoordinate.DATASET,
                        SemanticCoordinate.DIRECTED_PAIR,
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
                ExperimentName.SCALABILITY_AND_EFFICIENCY,
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
        )
    )
    return by_experiment.get(experiment, common)


@dataclass(frozen=True, slots=True)
class ConditionLabel:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ConditionRegistrationError("registered conditions must be non-empty")


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

    def __len__(self) -> int:
        return len(self.entries)


class ConditionRegistrationError(ValueError):
    pass
