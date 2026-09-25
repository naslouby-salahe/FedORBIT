from __future__ import annotations

import highspy
import numpy as np
import pytest

from fedorbit.config.loading import active_config
from fedorbit.methods.baselines import optimize_against_fixed_matrix
from fedorbit.optimization.correspondence import build_padded_block_structure
from fedorbit.optimization.dense_ccp import (
    AssignmentVariableLayout,
    barycenter_start,
    solve_lifted_lp,
)
from fedorbit.optimization.exact_sparse import run_support_master_lp
from fedorbit.optimization.objective import (
    CurriculumAction,
    RobustActionProblem,
    SupportCoordinateSet,
)
from fedorbit.types import CoarseGroup


def test_every_highs_lp_path_passes_registered_tolerances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: dict[str, float | bool | int | str] = {}
    original = highspy.Highs.setOptionValue

    def record_option(self: highspy.Highs, name: str, value: float | int | str | bool) -> None:
        configured[name] = value
        original(self, name, value)

    monkeypatch.setattr(highspy.Highs, "setOptionValue", record_option)
    group = CoarseGroup.DISRUPTION
    blocks = build_padded_block_structure((group,), {group: 2}, {group: 2})
    problem = RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=np.zeros((2, 2), dtype=np.float64),
        upper_response_matrix=np.zeros((2, 2), dtype=np.float64),
        target_importance=np.asarray((1.0, 0.0), dtype=np.float64),
        coordinate_caps=np.asarray((0.25, 0.0), dtype=np.float64),
        linear_costs=np.asarray((0.01, 0.0), dtype=np.float64),
        total_budget=0.50,
        principal_support=2,
    )

    run_support_master_lp(
        problem,
        SupportCoordinateSet(problem, (0,)),
        (np.asarray((0.1, 0.0), dtype=np.float64),),
    )
    optimize_against_fixed_matrix(problem, np.zeros((2, 2), dtype=np.float64))
    layout = AssignmentVariableLayout.build(blocks)
    alpha = CurriculumAction(problem, np.asarray((0.2, 0.0), dtype=np.float64))
    solve_lifted_lp(problem, alpha, layout, 0.0, barycenter_start(layout))

    settings = active_config().solvers.exact_sparse
    assert configured["primal_feasibility_tolerance"] == settings.lp_primal_feasibility_tolerance
    assert configured["dual_feasibility_tolerance"] == settings.lp_dual_feasibility_tolerance
    assert configured["optimality_tolerance"] == settings.lp_optimality_tolerance
