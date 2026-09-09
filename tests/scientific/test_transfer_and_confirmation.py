from __future__ import annotations

import pytest

from fedorbit.methods.assimilation import PreTestLifecycle, PreTestPhase
from fedorbit.methods.assimilation import TestOpeningRuleError as OpeningRuleError


def test_confirmation_lifecycle_fails_closed_until_artifacts_are_committed() -> None:
    lifecycle = PreTestLifecycle()
    for phase in (
        PreTestPhase.SOURCE_SELECTION_FINALIZED,
        PreTestPhase.ACTION_FINALIZED,
        PreTestPhase.CONFIRMATION_DECISION_FINALIZED,
        PreTestPhase.ASSIMILATION_SETTLED,
    ):
        lifecycle.complete_phase(phase)
    with pytest.raises(OpeningRuleError):
        lifecycle.open_test()
