from __future__ import annotations

from fedorbit.baselines.correspondence import (
    CommittedMapAction,
    CouplingDestroyedMatrices,
    CouplingDestructionError,
    committed_map_action,
    coupling_destroyed_matrices,
)
from fedorbit.baselines.fairness import (
    ComparatorResources,
    FairnessViolationError,
    assert_identical_resources,
    assert_registered_method_name,
)
from fedorbit.baselines.local import (
    FixedMatrixActionSolution,
    FixedMatrixOptimizationError,
    linear_objective_row,
    optimize_against_fixed_matrix,
)
from fedorbit.baselines.summaries import (
    CoarseBlockSummary,
    coarse_block_mean_matrix,
    coarse_block_min_matrix,
    matched_resource_rectangular_lower_bounds,
    orbit_mean_matrix,
)

__all__ = [
    "CoarseBlockSummary",
    "CommittedMapAction",
    "ComparatorResources",
    "CouplingDestroyedMatrices",
    "CouplingDestructionError",
    "FairnessViolationError",
    "FixedMatrixActionSolution",
    "FixedMatrixOptimizationError",
    "assert_identical_resources",
    "assert_registered_method_name",
    "coarse_block_mean_matrix",
    "coarse_block_min_matrix",
    "committed_map_action",
    "coupling_destroyed_matrices",
    "linear_objective_row",
    "matched_resource_rectangular_lower_bounds",
    "optimize_against_fixed_matrix",
    "orbit_mean_matrix",
]
