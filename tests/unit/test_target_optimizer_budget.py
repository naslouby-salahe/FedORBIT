from __future__ import annotations

from typing import cast

import pytest

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.runtime import RandomSeed
from fedorbit.methods.target import (
    OptimizerBudgetError,
    TargetOptimizerBudgetCategory,
    TargetOptimizerStepLedger,
)
from fedorbit.types import DirectedPairName, TransferMethod


def test_target_optimizer_ledger_prevents_cross_category_borrowing() -> None:
    ledger = TargetOptimizerStepLedger(
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        DirectedPairName("source -> target"),
        cast(RandomSeed, 1),
    )
    ledger.consume(
        TargetOptimizerBudgetCategory.CONFIRMATION_CANDIDATES,
        active_config().scientific.target_optimizer_budget.reserved.confirmation_candidates,
    )

    with pytest.raises(OptimizerBudgetError):
        ledger.consume(
            TargetOptimizerBudgetCategory.CONFIRMATION_CANDIDATES,
            active_config().scientific.confirmation.optimizer_steps_per_shadow,
        )

    assert (
        ledger.remaining(TargetOptimizerBudgetCategory.LIVE_ASSIMILATION)
        == active_config().scientific.target_optimizer_budget.reserved.live_assimilation
    )


def test_target_optimizer_ledger_rejects_steps_beyond_all_reserved_categories() -> None:
    ledger = TargetOptimizerStepLedger(
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        DirectedPairName("source -> target"),
        cast(RandomSeed, 1),
    )
    for category in TargetOptimizerBudgetCategory:
        ledger.consume(category, ledger.remaining(category))

    with pytest.raises(OptimizerBudgetError):
        ledger.consume(
            TargetOptimizerBudgetCategory.LIVE_ASSIMILATION,
            active_config().scientific.confirmation.accepted_live_assimilation_steps,
        )
