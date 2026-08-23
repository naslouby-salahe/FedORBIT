from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fedorbit.config.models import FedorbitConfig


class BudgetCategory(StrEnum):
    TARGET_RESPONSE_DIAGNOSTIC = "target_response_diagnostic"
    CONFIRMATION_CANDIDATES = "confirmation_candidates"
    LIVE_ASSIMILATION = "live_assimilation"
    NONTRANSFERABLE_SAFETY_RESERVE = "nontransferable_safety_reserve"


class OptimizerBudgetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TargetOptimizerStepLedger:
    maximum_total_steps: int
    reserved_steps: dict[BudgetCategory, int]
    consumed_steps: dict[BudgetCategory, int]

    @classmethod
    def from_config(cls, config: FedorbitConfig) -> TargetOptimizerStepLedger:
        budget = config.scientific.target_optimizer_budget
        diagnostic = config.scientific.target_response_diagnostic
        expected_diagnostic_reserve = (
            8 * diagnostic.paired_replicates * 2 * diagnostic.shadow_optimizer_steps
        )
        if budget.reserved.target_response_diagnostic != expected_diagnostic_reserve:
            raise OptimizerBudgetError(
                "configured target-response reserve "
                f"{budget.reserved.target_response_diagnostic} does not equal the registered "
                f"derivation {expected_diagnostic_reserve}"
            )
        if (
            budget.reserved.target_response_diagnostic
            + budget.reserved.confirmation_candidates
            + budget.reserved.live_assimilation
            + budget.reserved.nontransferable_safety_reserve
            != budget.maximum_steps_per_method_pair_seed_before_test
        ):
            raise OptimizerBudgetError("reserved budgets do not sum to the total step cap")
        reserved = budget.reserved
        return cls(
            maximum_total_steps=budget.maximum_steps_per_method_pair_seed_before_test,
            reserved_steps={
                BudgetCategory.TARGET_RESPONSE_DIAGNOSTIC: reserved.target_response_diagnostic,
                BudgetCategory.CONFIRMATION_CANDIDATES: reserved.confirmation_candidates,
                BudgetCategory.LIVE_ASSIMILATION: reserved.live_assimilation,
                BudgetCategory.NONTRANSFERABLE_SAFETY_RESERVE: (
                    reserved.nontransferable_safety_reserve
                ),
            },
            consumed_steps=dict.fromkeys(BudgetCategory, 0),
        )

    def remaining(self, category: BudgetCategory) -> int:
        return self.reserved_steps[category] - self.consumed_steps[category]

    def consume(self, category: BudgetCategory, steps: int) -> None:
        if steps < 0:
            raise OptimizerBudgetError("consumed steps must be nonnegative")
        if steps > self.remaining(category):
            raise OptimizerBudgetError(
                f"category {category.value} budget exhausted: requested {steps} steps, "
                f"remaining {self.remaining(category)}"
            )
        self.consumed_steps[category] += steps

    def require_capacity(self, category: BudgetCategory, steps: int) -> None:
        if steps > self.remaining(category):
            raise OptimizerBudgetError(
                f"category {category.value} cannot absorb {steps} steps; "
                f"remaining {self.remaining(category)}"
            )

    @property
    def total_consumed(self) -> int:
        return sum(self.consumed_steps.values())
