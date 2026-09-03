from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from numpy.typing import NDArray

from fedorbit.config.loading import active_config
from fedorbit.learning.training import BaseCheckpoint
from fedorbit.response.estimation import ShadowSettings
from fedorbit.response.pilot import PilotData
from fedorbit.response.uncertainty import FinalResponseEstimate, estimate_response_bands


class TargetImportanceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TransferNodeRisk:
    node_index: int
    is_actionable: bool
    meta_class_risk: float

    def __post_init__(self) -> None:
        if self.node_index < 0:
            raise TargetImportanceError(f"negative node index: {self.node_index}")
        if not math.isfinite(self.meta_class_risk):
            raise TargetImportanceError(
                f"node {self.node_index} META class risk is not finite: {self.meta_class_risk}"
            )
        if self.meta_class_risk < 0.0:
            raise TargetImportanceError(
                f"node {self.node_index} META class risk must be nonnegative"
            )


@dataclass(frozen=True, slots=True)
class TargetImportance:
    weights_by_node_index: Mapping[int, float]

    def __post_init__(self) -> None:
        for node_index, weight in self.weights_by_node_index.items():
            if node_index < 0:
                raise TargetImportanceError(f"negative node index: {node_index}")
            if not math.isfinite(weight):
                raise TargetImportanceError(f"node {node_index} importance is not finite")
            if weight < 0.0:
                raise TargetImportanceError(f"node {node_index} importance must be nonnegative")
        if self.weights_by_node_index:
            total = math.fsum(self.weights_by_node_index.values())
            absolute_tolerance = math.ulp(1.0) * max(1, len(self.weights_by_node_index))
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=absolute_tolerance):
                raise TargetImportanceError("target importance weights must sum to one")

    def weight_of(self, node_index: int) -> float:
        return self.weights_by_node_index[node_index]

    def as_vector(self, size: int) -> NDArray[np.float64]:
        vector = np.zeros(size, dtype=np.float64)
        for node_index, weight in self.weights_by_node_index.items():
            if node_index >= size:
                raise TargetImportanceError(f"node {node_index} outside vector size {size}")
            vector[node_index] = weight
        return vector

    @property
    def actionable_total(self) -> float:
        return sum(self.weights_by_node_index.values())


def build_target_importance(
    node_risks: tuple[TransferNodeRisk, ...],
) -> TargetImportance:
    floor = active_config().scientific.target_importance.class_risk_floor
    if floor <= 0.0:
        raise TargetImportanceError("class risk floor must be positive")
    seen: set[int] = set()
    floored: OrderedDict[int, float] = OrderedDict()
    zero_nodes: OrderedDict[int, float] = OrderedDict()
    for node_risk in node_risks:
        if node_risk.node_index in seen:
            raise TargetImportanceError(f"node {node_risk.node_index} reported more than once")
        seen.add(node_risk.node_index)
        if node_risk.is_actionable:
            floored[node_risk.node_index] = max(node_risk.meta_class_risk, floor)
        else:
            zero_nodes[node_risk.node_index] = 0.0
    if not floored:
        raise TargetImportanceError(
            "no actionable target nodes with META risk; target importance undefined"
        )
    total = sum(floored.values())
    weights = OrderedDict(
        (node_index, value / total) for node_index, value in sorted(floored.items())
    )
    combined = OrderedDict((*zero_nodes.items(), *weights.items()))
    ordered = OrderedDict((node_index, combined[node_index]) for node_index in sorted(seen))
    return TargetImportance(weights_by_node_index=ordered)


def estimate_target_response_diagnostic(
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: tuple[int, ...],
    seed: int,
) -> FinalResponseEstimate:
    diagnostic = active_config().scientific.target_response_diagnostic
    settings = ShadowSettings(
        diagnostic.intervention_magnitude,
        diagnostic.shadow_optimizer_steps,
        data.learning_rate,
        data.weight_decay,
    )
    return estimate_response_bands(
        model,
        checkpoint,
        data,
        (intervention_classes,),
        settings,
        seed,
        replicate_count=diagnostic.paired_replicates,
        bootstrap_resamples=diagnostic.simultaneous_bootstrap_resamples,
        confidence_level=diagnostic.confidence_level,
        seed_stage="target-local-diagnostic",
    )


class SelectionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceProposal:
    source_client_name: str
    certified_robust_value: float


@dataclass(frozen=True, slots=True)
class RankedProposal:
    rank: int
    proposal: SourceProposal


@dataclass(frozen=True, slots=True)
class SelectionAttempt:
    rank: int
    source_client_name: str
    accepted: bool


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    accepted_proposal: SourceProposal | None
    accepted_rank: int | None
    attempts: tuple[SelectionAttempt, ...]
    remained_local_only: bool


def rank_source_proposals(
    candidates: Sequence[SourceProposal],
) -> tuple[RankedProposal, ...]:
    config = active_config()
    action = config.scientific.action
    multi_source = config.scientific.multi_source_selection
    communication = abs(multi_source.communication_cost_coefficient_in_principal_ranking)
    confirmation = abs(multi_source.confirmation_cost_coefficient_in_principal_ranking)
    if communication > 0.0 or confirmation > 0.0:
        raise SelectionError(
            "principal ranking requires zero communication and confirmation cost coefficients"
        )
    seen_clients: set[str] = set()
    positive: list[SourceProposal] = []
    for candidate in candidates:
        if candidate.source_client_name in seen_clients:
            raise SelectionError(
                f"source client proposed more than once: {candidate.source_client_name}"
            )
        seen_clients.add(candidate.source_client_name)
        if candidate.certified_robust_value <= action.positive_source_value_threshold:
            continue
        positive.append(candidate)
    ordered = sorted(
        positive, key=lambda entry: (-entry.certified_robust_value, entry.source_client_name)
    )
    capped = ordered[: action.maximum_source_proposals_per_target]
    return tuple(
        RankedProposal(rank=index + 1, proposal=proposal) for index, proposal in enumerate(capped)
    )


def select_source_sequentially(
    ranked: Sequence[RankedProposal],
    confirmation_decision: Callable[[SourceProposal], bool],
) -> SelectionDecision:
    maximum = active_config().scientific.action.maximum_source_proposals_per_target
    attempts: list[SelectionAttempt] = []
    for ranked_proposal in sorted(ranked, key=lambda entry: entry.rank)[:maximum]:
        if len(attempts) >= maximum:
            break
        accepted = bool(confirmation_decision(ranked_proposal.proposal))
        attempts.append(
            SelectionAttempt(
                rank=ranked_proposal.rank,
                source_client_name=ranked_proposal.proposal.source_client_name,
                accepted=accepted,
            )
        )
        if accepted:
            return SelectionDecision(
                accepted_proposal=ranked_proposal.proposal,
                accepted_rank=ranked_proposal.rank,
                attempts=tuple(attempts),
                remained_local_only=False,
            )
    return SelectionDecision(
        accepted_proposal=None,
        accepted_rank=None,
        attempts=tuple(attempts),
        remained_local_only=True,
    )


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


class CurriculumError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CurriculumMultipliers:
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.values.numel() == 0:
            raise CurriculumError("curriculum multipliers must be a non-empty vector")
        if not bool(torch.isfinite(self.values).all()) or bool((self.values < 0.0).any()):
            raise CurriculumError("curriculum multipliers must be finite and nonnegative")
