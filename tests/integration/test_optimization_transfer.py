from __future__ import annotations

import numpy as np
import pytest

from fedorbit.methods.assimilation import PreTestLifecycle
from fedorbit.methods.assimilation import TestOpeningRuleError as OpeningRuleError
from fedorbit.optimization.assignment import solve_minimum_cost_assignment


def test_solver_pipeline_returns_lexicographic_optimum_under_tie() -> None:
    costs = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    result = solve_minimum_cost_assignment(costs, 1e-12)
    assert result.column_for_row == (0, 1)
    assert result.objective_value == 0.0


def test_transfer_pipeline_keeps_test_closed_until_all_pretest_phases_complete() -> None:
    lifecycle = PreTestLifecycle()
    for phase in (
        "source_selection_finalized",
        "action_finalized",
        "confirmation_decision_finalized",
        "assimilation_settled",
    ):
        lifecycle.complete_phase(phase)
    with pytest.raises(OpeningRuleError):
        lifecycle.open_test()
    lifecycle.complete_phase("pre_test_artifacts_committed")
    lifecycle.open_test()
    lifecycle.assert_opened()
