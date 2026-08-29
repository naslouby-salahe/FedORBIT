from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

from fedorbit.domain.enums import ExperimentName


def experiment_relevance(experiment: ExperimentName) -> frozenset[str]:
    common = frozenset({"experiment", "seed"})
    by_experiment: Mapping[ExperimentName, frozenset[str]] = OrderedDict(
        (
            (
                ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION,
                frozenset({"experiment", "seed", "condition", "support"}),
            ),
            (
                ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION,
                frozenset({"experiment", "seed", "dataset", "directed_pair"}),
            ),
            (
                ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT,
                frozenset({"experiment", "dataset", "source_client", "seed"}),
            ),
            (
                ExperimentName.SOURCE_RESPONSE_ESTIMATOR_PILOT,
                frozenset({"experiment", "dataset", "directed_pair", "method", "seed"}),
            ),
            (
                ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION,
                frozenset({"experiment", "dataset", "directed_pair", "method", "seed"}),
            ),
            (
                ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
                frozenset({"experiment", "dataset", "directed_pair", "method", "support", "seed"}),
            ),
            (
                ExperimentName.MECHANISM_ABLATIONS,
                frozenset(
                    {
                        "experiment",
                        "dataset",
                        "directed_pair",
                        "method",
                        "support",
                        "condition",
                        "seed",
                    }
                ),
            ),
            (
                ExperimentName.SPARSITY_AND_DENSE_FALLBACK,
                frozenset({"experiment", "dataset", "directed_pair", "method", "support", "seed"}),
            ),
            (
                ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY,
                frozenset({"experiment", "dataset", "directed_pair", "method", "support", "seed"}),
            ),
            (
                ExperimentName.SECONDARY_CROSS_MODALITY_GENERALIZATION,
                frozenset({"experiment", "dataset", "directed_pair", "method", "seed"}),
            ),
            (
                ExperimentName.SEMANTIC_SUFFICIENCY_FRONTIER,
                frozenset(
                    {
                        "experiment",
                        "dataset",
                        "directed_pair",
                        "method",
                        "support",
                        "condition",
                        "seed",
                    }
                ),
            ),
            (
                ExperimentName.WEAK_SIGNAL_SUPPORT_AND_HETEROGENEITY_BOUNDARIES,
                frozenset(
                    {
                        "experiment",
                        "dataset",
                        "directed_pair",
                        "method",
                        "support",
                        "condition",
                        "seed",
                    }
                ),
            ),
            (
                ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK,
                frozenset({"experiment", "dataset", "directed_pair", "method", "support", "seed"}),
            ),
            (
                ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION,
                frozenset({"experiment", "dataset", "directed_pair", "method", "support", "seed"}),
            ),
            (
                ExperimentName.MULTI_SOURCE_SELECTION_VALIDATION,
                frozenset(
                    {
                        "experiment",
                        "dataset",
                        "directed_pair",
                        "method",
                        "support",
                        "condition",
                        "seed",
                    }
                ),
            ),
            (
                ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT,
                frozenset({"experiment", "dataset", "directed_pair", "method", "seed"}),
            ),
            (
                ExperimentName.SCALABILITY_AND_EFFICIENCY,
                frozenset({"experiment", "dataset", "directed_pair", "method", "support", "seed"}),
            ),
        )
    )
    return by_experiment.get(experiment, common)
