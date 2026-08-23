from __future__ import annotations

import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.transfer.optimizer_budget import (
    BudgetCategory,
    OptimizerBudgetError,
    TargetOptimizerStepLedger,
)


@pytest.fixture
def config() -> FedorbitConfig:
    return load_fedorbit_config()


def test_reserved_categories_sum_to_total_cap(config: FedorbitConfig) -> None:
    ledger = TargetOptimizerStepLedger.from_config(config)
    budget = config.scientific.target_optimizer_budget
    assert sum(ledger.reserved_steps.values()) == (
        budget.maximum_steps_per_method_pair_seed_before_test
    )


def test_target_response_reserve_matches_registered_derivation(
    config: FedorbitConfig,
) -> None:
    diagnostic = config.scientific.target_response_diagnostic
    expected = 8 * diagnostic.paired_replicates * 2 * diagnostic.shadow_optimizer_steps
    ledger = TargetOptimizerStepLedger.from_config(config)
    assert (
        ledger.reserved_steps[BudgetCategory.TARGET_RESPONSE_DIAGNOSTIC]
        == expected
        == config.scientific.target_optimizer_budget.reserved.target_response_diagnostic
    )


def test_consumption_is_tracked_per_category(config: FedorbitConfig) -> None:
    ledger = TargetOptimizerStepLedger.from_config(config)
    ledger.consume(BudgetCategory.TARGET_RESPONSE_DIAGNOSTIC, 100)
    ledger.consume(BudgetCategory.LIVE_ASSIMILATION, 50)
    assert ledger.remaining(BudgetCategory.TARGET_RESPONSE_DIAGNOSTIC) == 3100
    assert ledger.remaining(BudgetCategory.LIVE_ASSIMILATION) == 450
    assert ledger.total_consumed == 150


def test_unused_budget_never_transfers_across_categories_or_methods(
    config: FedorbitConfig,
) -> None:
    ledger = TargetOptimizerStepLedger.from_config(config)
    with pytest.raises(OptimizerBudgetError):
        ledger.consume(
            BudgetCategory.CONFIRMATION_CANDIDATES,
            ledger.reserved_steps[BudgetCategory.CONFIRMATION_CANDIDATES] + 1,
        )
    ledger.consume(BudgetCategory.NONTRANSFERABLE_SAFETY_RESERVE, 0)
    fresh = TargetOptimizerStepLedger.from_config(config)
    ledger.consume(BudgetCategory.TARGET_RESPONSE_DIAGNOSTIC, 3200)
    assert fresh.remaining(BudgetCategory.TARGET_RESPONSE_DIAGNOSTIC) == 3200


def test_negative_consumption_rejected(config: FedorbitConfig) -> None:
    ledger = TargetOptimizerStepLedger.from_config(config)
    with pytest.raises(OptimizerBudgetError):
        ledger.consume(BudgetCategory.LIVE_ASSIMILATION, -1)


def test_capacity_check_without_mutation(config: FedorbitConfig) -> None:
    ledger = TargetOptimizerStepLedger.from_config(config)
    with pytest.raises(OptimizerBudgetError):
        ledger.require_capacity(BudgetCategory.LIVE_ASSIMILATION, 501)
    assert ledger.remaining(BudgetCategory.LIVE_ASSIMILATION) == 500
