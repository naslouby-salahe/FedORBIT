from __future__ import annotations

import pytest

from fedorbit.transfer.assimilation import PreTestLifecycle, TestOpeningRuleError


def test_transfer_pipeline_keeps_test_closed_until_all_pretest_phases_complete() -> None:
    lifecycle = PreTestLifecycle()
    for phase in (
        "source_selection_finalized",
        "action_finalized",
        "confirmation_decision_finalized",
        "assimilation_settled",
    ):
        lifecycle.complete_phase(phase)
    with pytest.raises(TestOpeningRuleError):
        lifecycle.open_test()
    lifecycle.complete_phase("pre_test_artifacts_committed")
    lifecycle.open_test()
    lifecycle.assert_opened()
