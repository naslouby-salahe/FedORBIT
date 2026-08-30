from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fedorbit.config.context import active_config


class BudgetCategory(StrEnum):
    TARGET_RESPONSE_DIAGNOSTIC = "target_response_diagnostic"
    CONFIRMATION_CANDIDATES = "confirmation_candidates"
    LIVE_ASSIMILATION = "live_assimilation"
    NONTRANSFERABLE_SAFETY_RESERVE = "nontransferable_safety_reserve"


class OptimizerBudgetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OptimizerStepAllocation:
    target_response_diagnostic: int
    confirmation_candidates: int
    live_assimilation: int
    nontransferable_safety_reserve: int

    def for_category(self, category: BudgetCategory) -> int:
        if category == BudgetCategory.TARGET_RESPONSE_DIAGNOSTIC:
            return self.target_response_diagnostic
        if category == BudgetCategory.CONFIRMATION_CANDIDATES:
            return self.confirmation_candidates
        if category == BudgetCategory.LIVE_ASSIMILATION:
            return self.live_assimilation
        return self.nontransferable_safety_reserve

    def incremented(self, category: BudgetCategory, steps: int) -> OptimizerStepAllocation:
        if category == BudgetCategory.TARGET_RESPONSE_DIAGNOSTIC:
            return OptimizerStepAllocation(
                self.target_response_diagnostic + steps,
                self.confirmation_candidates,
                self.live_assimilation,
                self.nontransferable_safety_reserve,
            )
        if category == BudgetCategory.CONFIRMATION_CANDIDATES:
            return OptimizerStepAllocation(
                self.target_response_diagnostic,
                self.confirmation_candidates + steps,
                self.live_assimilation,
                self.nontransferable_safety_reserve,
            )
        if category == BudgetCategory.LIVE_ASSIMILATION:
            return OptimizerStepAllocation(
                self.target_response_diagnostic,
                self.confirmation_candidates,
                self.live_assimilation + steps,
                self.nontransferable_safety_reserve,
            )
        return OptimizerStepAllocation(
            self.target_response_diagnostic,
            self.confirmation_candidates,
            self.live_assimilation,
            self.nontransferable_safety_reserve + steps,
        )

    @property
    def total(self) -> int:
        return (
            self.target_response_diagnostic
            + self.confirmation_candidates
            + self.live_assimilation
            + self.nontransferable_safety_reserve
        )


@dataclass(frozen=True, slots=True)
class TargetOptimizerStepLedger:
    maximum_total_steps: int
    reserved_steps: OptimizerStepAllocation
    consumed_steps: OptimizerStepAllocation

    @classmethod
    def from_context(cls) -> TargetOptimizerStepLedger:
        config = active_config()
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
            reserved_steps=OptimizerStepAllocation(
                target_response_diagnostic=reserved.target_response_diagnostic,
                confirmation_candidates=reserved.confirmation_candidates,
                live_assimilation=reserved.live_assimilation,
                nontransferable_safety_reserve=reserved.nontransferable_safety_reserve,
            ),
            consumed_steps=OptimizerStepAllocation(0, 0, 0, 0),
        )

    def remaining(self, category: BudgetCategory) -> int:
        return self.reserved_steps.for_category(category) - self.consumed_steps.for_category(
            category
        )

    def consume(self, category: BudgetCategory, steps: int) -> TargetOptimizerStepLedger:
        if steps < 0:
            raise OptimizerBudgetError("consumed steps must be nonnegative")
        if steps > self.remaining(category):
            raise OptimizerBudgetError(
                f"category {category.value} budget exhausted: requested {steps} steps, "
                f"remaining {self.remaining(category)}"
            )
        return TargetOptimizerStepLedger(
            maximum_total_steps=self.maximum_total_steps,
            reserved_steps=self.reserved_steps,
            consumed_steps=self.consumed_steps.incremented(category, steps),
        )

    def require_capacity(self, category: BudgetCategory, steps: int) -> None:
        if steps > self.remaining(category):
            raise OptimizerBudgetError(
                f"category {category.value} cannot absorb {steps} steps; "
                f"remaining {self.remaining(category)}"
            )

    @property
    def total_consumed(self) -> int:
        return self.consumed_steps.total
