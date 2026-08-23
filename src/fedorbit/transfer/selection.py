from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fedorbit.config.models import FedorbitConfig


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
    config: FedorbitConfig,
    candidates: Sequence[SourceProposal],
) -> tuple[RankedProposal, ...]:
    action = config.scientific.action
    multi_source = config.scientific.multi_source_selection
    if multi_source.communication_cost_coefficient_in_principal_ranking != 0.0:
        raise SelectionError("principal ranking requires a zero communication-cost coefficient")
    if multi_source.confirmation_cost_coefficient_in_principal_ranking != 0.0:
        raise SelectionError("principal ranking requires a zero confirmation-cost coefficient")
    seen_clients: set[str] = set()
    positive: list[SourceProposal] = []
    for candidate in candidates:
        if candidate.source_client_name in seen_clients:
            raise SelectionError(
                f"source client proposed more than once: {candidate.source_client_name}"
            )
        seen_clients.add(candidate.source_client_name)
        if not candidate.certified_robust_value > action.positive_source_value_threshold:
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
    config: FedorbitConfig,
    ranked: Sequence[RankedProposal],
    confirmation_decision: Callable[[SourceProposal], bool],
) -> SelectionDecision:
    maximum = config.scientific.action.maximum_source_proposals_per_target
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
