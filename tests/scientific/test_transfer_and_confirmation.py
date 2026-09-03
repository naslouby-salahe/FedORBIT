from __future__ import annotations

import pytest

from fedorbit.methods.assimilation import PreTestLifecycle
from fedorbit.methods.assimilation import TestOpeningRuleError as OpeningRuleError


def test_confirmation_lifecycle_fails_closed_until_artifacts_are_committed() -> None:
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
