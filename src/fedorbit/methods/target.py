from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np
import torch
from numpy.typing import NDArray

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.runtime import RandomSeed
from fedorbit.types import (
    Coefficient,
    ConceptCount,
    DirectedPairName,
    Index,
    RepetitionCount,
    Score,
    SourceClientName,
    StepCount,
    TransferMethod,
)


class TargetImportanceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TransferNodeRisk:
    node_index: Index
    is_actionable: bool
    meta_class_risk: Coefficient

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
    weights_by_node_index: Mapping[Index, Coefficient]

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

    def as_vector(self, size: Index) -> NDArray[np.float64]:
        vector = np.zeros(size, dtype=np.float64)
        for node_index, weight in self.weights_by_node_index.items():
            if node_index >= size:
                raise TargetImportanceError(f"node {node_index} outside vector size {size}")
            vector[node_index] = weight
        return vector


def build_target_importance(
    node_risks: tuple[TransferNodeRisk, ...],
) -> TargetImportance:
    floor = active_config().scientific.target_importance.class_risk_floor
    if floor <= 0.0:
        raise TargetImportanceError("class risk floor must be positive")
    seen: set[Index] = set()
    floored: OrderedDict[Index, Coefficient] = OrderedDict()
    zero_nodes: OrderedDict[Index, Coefficient] = OrderedDict()
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


class SelectionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceProposal:
    source_client_name: SourceClientName
    certified_robust_value: Score


@dataclass(frozen=True, slots=True)
class RankedProposal:
    rank: Index
    proposal: SourceProposal


@dataclass(frozen=True, slots=True)
class SelectionAttempt:
    rank: Index
    source_client_name: SourceClientName
    accepted: bool


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    accepted_proposal: SourceProposal | None
    accepted_rank: Index | None
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
    seen_clients: set[SourceClientName] = set()
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


SHADOW_PAIR_DIRECTIONS: RepetitionCount = 2


class OptimizerBudgetError(ValueError):
    pass


class TargetOptimizerBudgetCategory(StrEnum):
    TARGET_RESPONSE_DIAGNOSTIC = "target_response_diagnostic"
    CONFIRMATION_CANDIDATES = "confirmation_candidates"
    LIVE_ASSIMILATION = "live_assimilation"
    NONTRANSFERABLE_SAFETY_RESERVE = "nontransferable_safety_reserve"


@dataclass(frozen=True, slots=True)
class TargetOptimizerReservations:
    target_response_diagnostic: StepCount
    confirmation_candidates: StepCount
    live_assimilation: StepCount
    nontransferable_safety_reserve: StepCount

    @property
    def total_steps(self) -> StepCount:
        return (
            self.target_response_diagnostic
            + self.confirmation_candidates
            + self.live_assimilation
            + self.nontransferable_safety_reserve
        )

    def remaining(self, category: TargetOptimizerBudgetCategory) -> StepCount:
        match category:
            case TargetOptimizerBudgetCategory.TARGET_RESPONSE_DIAGNOSTIC:
                return self.target_response_diagnostic
            case TargetOptimizerBudgetCategory.CONFIRMATION_CANDIDATES:
                return self.confirmation_candidates
            case TargetOptimizerBudgetCategory.LIVE_ASSIMILATION:
                return self.live_assimilation
            case TargetOptimizerBudgetCategory.NONTRANSFERABLE_SAFETY_RESERVE:
                return self.nontransferable_safety_reserve

    def reduced(
        self,
        category: TargetOptimizerBudgetCategory,
        step_count: StepCount,
    ) -> TargetOptimizerReservations:
        remaining = self.remaining(category) - step_count
        match category:
            case TargetOptimizerBudgetCategory.TARGET_RESPONSE_DIAGNOSTIC:
                return replace(self, target_response_diagnostic=remaining)
            case TargetOptimizerBudgetCategory.CONFIRMATION_CANDIDATES:
                return replace(self, confirmation_candidates=remaining)
            case TargetOptimizerBudgetCategory.LIVE_ASSIMILATION:
                return replace(self, live_assimilation=remaining)
            case TargetOptimizerBudgetCategory.NONTRANSFERABLE_SAFETY_RESERVE:
                return replace(self, nontransferable_safety_reserve=remaining)


@dataclass(slots=True)
class TargetOptimizerStepLedger:
    method: TransferMethod
    directed_pair: DirectedPairName
    seed: RandomSeed
    _reservations: TargetOptimizerReservations = field(init=False)

    def __post_init__(self) -> None:
        budget = active_config().scientific.target_optimizer_budget
        reserved = budget.reserved
        reservations = TargetOptimizerReservations(
            reserved.target_response_diagnostic,
            reserved.confirmation_candidates,
            reserved.live_assimilation,
            reserved.nontransferable_safety_reserve,
        )
        if reservations.total_steps > budget.maximum_steps_per_method_pair_seed_before_test:
            raise OptimizerBudgetError("target optimizer reservations exceed the registered cap")
        self._reservations = reservations

    def consume(
        self,
        category: TargetOptimizerBudgetCategory,
        step_count: StepCount,
    ) -> None:
        if step_count < 0:
            raise OptimizerBudgetError("target optimizer step count must be nonnegative")
        remaining = self._reservations.remaining(category)
        if step_count > remaining:
            raise OptimizerBudgetError(
                f"target optimizer budget exceeded for {self.method.value}, "
                f"{self.directed_pair}, seed {self.seed}, category {category.value}: "
                f"requested {step_count}, remaining {remaining}"
            )
        self._reservations = self._reservations.reduced(category, step_count)

    def remaining(self, category: TargetOptimizerBudgetCategory) -> StepCount:
        return self._reservations.remaining(category)

    def assert_seed(self, seed: RandomSeed) -> None:
        if self.seed != seed:
            raise OptimizerBudgetError(
                f"target optimizer ledger seed {self.seed} does not match execution seed {seed}"
            )

    def assert_directed_pair(self, directed_pair: DirectedPairName) -> None:
        if self.directed_pair != directed_pair:
            raise OptimizerBudgetError(
                f"target optimizer ledger pair {self.directed_pair} does not match "
                f"execution pair {directed_pair}"
            )


def target_diagnostic_reserve(intervention_class_count: ConceptCount) -> StepCount:
    diagnostic = active_config().scientific.target_response_diagnostic
    reserve: StepCount = (
        intervention_class_count
        * diagnostic.paired_replicates
        * SHADOW_PAIR_DIRECTIONS
        * diagnostic.shadow_optimizer_steps
    )
    return reserve


def assert_target_diagnostic_reserve(intervention_class_count: ConceptCount) -> None:
    budget = active_config().scientific.target_optimizer_budget
    expected = target_diagnostic_reserve(intervention_class_count)
    if budget.reserved.target_response_diagnostic != expected:
        raise OptimizerBudgetError(
            "configured target-response reserve "
            f"{budget.reserved.target_response_diagnostic} does not equal the derivation "
            f"{expected} for {intervention_class_count} intervention classes"
        )


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
