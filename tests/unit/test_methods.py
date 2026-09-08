from __future__ import annotations

from fedorbit.config.loading import configured, load_fedorbit_config
from fedorbit.methods.target import (
    SelectionError,
    SourceProposal,
    rank_source_proposals,
    select_source_sequentially,
    without_confirmation_selection,
)
from fedorbit.types import Score, SourceClientName


def _proposal(name: SourceClientName, value: Score) -> SourceProposal:
    return SourceProposal(source_client_name=name, certified_robust_value=value)


def test_nonpositive_proposals_discarded_before_ranking() -> None:
    ranked = rank_source_proposals(
        (
            _proposal(SourceClientName("edge"), Score(-0.5)),
            _proposal(SourceClientName("windows"), Score(0.0)),
            _proposal(SourceClientName("linux"), Score(0.25)),
        ),
    )
    assert len(ranked) == 1
    assert ranked[0].rank == 1
    assert ranked[0].proposal.source_client_name == "linux"


def test_descending_value_order_with_stable_client_name_ties() -> None:
    ranked = rank_source_proposals(
        (
            _proposal(SourceClientName("ton_network"), Score(0.3)),
            _proposal(SourceClientName("zeta"), Score(0.5)),
            _proposal(SourceClientName("alpha"), Score(0.5)),
        ),
    )
    assert [entry.proposal.source_client_name for entry in ranked] == [
        "alpha",
        "zeta",
        "ton_network",
    ]
    assert [entry.rank for entry in ranked] == [1, 2, 3]


def test_maximum_proposal_cap_enforced_from_configuration() -> None:
    config = load_fedorbit_config()
    maximum = config.scientific.action.maximum_source_proposals_per_target
    assert maximum == 3
    candidates = tuple(
        _proposal(SourceClientName(f"source_{i}"), Score(0.9 - i * 0.05)) for i in range(6)
    )
    ranked = rank_source_proposals(candidates)
    assert len(ranked) == maximum
    assert all(entry.rank <= maximum for entry in ranked)


def test_sequential_confirmation_stops_at_first_accept() -> None:
    ranked = rank_source_proposals(
        (
            _proposal(SourceClientName("first"), Score(0.9)),
            _proposal(SourceClientName("second"), Score(0.7)),
            _proposal(SourceClientName("third"), Score(0.5)),
        ),
    )
    decision = select_source_sequentially(
        ranked,
        lambda proposal: proposal.source_client_name == SourceClientName("second"),
    )
    assert not decision.remained_local_only
    assert decision.accepted_rank == 2
    assert decision.accepted_proposal is not None
    assert decision.accepted_proposal.source_client_name == "second"
    assert [attempt.rank for attempt in decision.attempts] == [1, 2]
    assert [attempt.accepted for attempt in decision.attempts] == [False, True]


def test_no_accepted_candidate_remains_local_only() -> None:
    ranked = rank_source_proposals(
        tuple(
            _proposal(SourceClientName(f"source_{i}"), Score(0.5 - i * 0.1))
            for i in range(4)
        ),
    )
    decision = select_source_sequentially(ranked, lambda _proposal: False)
    assert decision.remained_local_only
    assert decision.accepted_proposal is None
    assert decision.accepted_rank is None
    assert [attempt.rank for attempt in decision.attempts] == [1, 2, 3]


def test_empty_candidates_remain_local_only() -> None:
    ranked = rank_source_proposals(())
    assert ranked == ()
    decision = select_source_sequentially(ranked, lambda _p: True)
    assert decision.remained_local_only
    assert decision.attempts == ()


def test_without_confirmation_selection_accepts_the_first_positive_proposal() -> None:
    ranked = rank_source_proposals(
        (
            _proposal(SourceClientName("first"), Score(0.9)),
            _proposal(SourceClientName("second"), Score(0.5)),
        ),
    )
    decision = without_confirmation_selection(ranked)
    assert not decision.remained_local_only
    assert decision.accepted_rank == 1
    assert decision.accepted_proposal is not None
    assert decision.accepted_proposal.source_client_name == "first"


def test_principal_cost_coefficients_are_zero_and_validated() -> None:
    config = load_fedorbit_config()
    multi_source = config.scientific.multi_source_selection
    assert multi_source.communication_cost_coefficient_in_principal_ranking == 0.0
    assert multi_source.confirmation_cost_coefficient_in_principal_ranking == 0.0
    nonzero = config.model_copy(deep=True)
    object.__setattr__(
        nonzero.scientific.multi_source_selection,
        "communication_cost_coefficient_in_principal_ranking",
        0.5,
    )
    try:
        with configured(nonzero):
            rank_source_proposals((_proposal(SourceClientName("edge"), Score(0.4)),))
    except SelectionError:
        pass
    else:
        raise AssertionError("nonzero cost coefficient must be rejected in principal ranking")


def test_ranking_is_deterministic() -> None:
    candidates = (
        _proposal(SourceClientName("b"), Score(0.4)),
        _proposal(SourceClientName("a"), Score(0.6)),
        _proposal(SourceClientName("c"), Score(0.2)),
    )
    first = rank_source_proposals(candidates)
    second = rank_source_proposals(tuple(reversed(candidates)))
    assert first == second
