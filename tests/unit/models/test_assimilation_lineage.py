from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fedorbit.methods.assimilation as assimilation_module
from fedorbit.config.loading import active_config
from fedorbit.learning.scoring import LocalClassCount, ScoringRequest, score_model
from fedorbit.learning.training import ClassWeights
from fedorbit.methods.assimilation import (
    AssimilationCoordinates,
    PreConfirmStatePair,
    PreConfirmTargetState,
    apply_accepted_assimilation,
    capture_pre_confirm_pair,
    settle_rejected_proposal,
)
from fedorbit.methods.target import (
    CurriculumMultipliers,
    TargetOptimizerBudgetCategory,
    TargetOptimizerStepLedger,
    TransferNodeRisk,
    build_target_importance,
)
from fedorbit.types import (
    ArtifactIdentifier,
    DirectedPairName,
    EvaluationConditionName,
    Sha256Digest,
    SourceClientName,
    TransferMethod,
)

_SEED = 1103
_PAIR = DirectedPairName("ton_iot_windows10_host -> ton_iot_network")


def _model() -> nn.Module:
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(((1.0, -1.0), (-1.0, 1.0))))
        model.bias.zero_()
    return model


def _features() -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.25), (0.25, 0.5), (0.75, 0.5)),
        dtype=torch.float32,
    )
    targets = torch.tensor((0, 1, 0, 1, 0, 1), dtype=torch.long)
    return features, targets


def _ledger() -> TargetOptimizerStepLedger:
    return TargetOptimizerStepLedger(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, _PAIR, _SEED)


def _coordinates() -> AssimilationCoordinates:
    return AssimilationCoordinates(
        target_client=SourceClientName("ton_iot_windows10_host"),
        directed_pair=_PAIR,
        condition=EvaluationConditionName("principal"),
        seed=_SEED,
        clean_pretransfer_checkpoint_artifact_id=ArtifactIdentifier("checkpoint"),
        source_packet_artifact_id=ArtifactIdentifier("packet"),
        action_artifact_sha256=Sha256Digest("a" * 64),
    )


def _parameters(model: nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in model.parameters())


def _run_assimilation(
    multipliers: CurriculumMultipliers,
    pre_confirm: PreConfirmTargetState,
    seed: int = _SEED,
) -> tuple[nn.Module, TargetOptimizerStepLedger]:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pre_confirm.restore_into(model, optimizer)
    features, targets = _features()
    ledger = _ledger()
    apply_accepted_assimilation(
        model,
        optimizer,
        pre_confirm,
        features,
        targets,
        ClassWeights(torch.ones(2, dtype=torch.float32)),
        multipliers,
        seed,
        _coordinates(),
        ledger,
        batch_size=4,
    )
    return model, ledger


def _curriculum_multipliers(alpha: tuple[float, ...]) -> CurriculumMultipliers:
    return CurriculumMultipliers(
        torch.tensor([1.0 + value for value in alpha], dtype=torch.float32)
    )


def test_accepted_assimilation_discards_shadow_state_and_restarts_from_the_clean_checkpoint() -> (
    None
):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pair = capture_pre_confirm_pair(model, optimizer)
    assert isinstance(pair, PreConfirmStatePair)
    clean = pair.baseline
    features, targets = _features()
    multipliers = _curriculum_multipliers((0.2, 0.0))
    reference_model, reference_ledger = _run_assimilation(multipliers, clean)

    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(features), targets)
        loss.backward()
        optimizer.step()
    shadow_parameters = _parameters(model)
    assert any(
        not torch.equal(actual, expected)
        for actual, expected in zip(shadow_parameters, _parameters(reference_model), strict=True)
    )

    steps = apply_accepted_assimilation(
        model,
        optimizer,
        clean,
        features,
        targets,
        ClassWeights(torch.ones(2, dtype=torch.float32)),
        multipliers,
        _SEED,
        _coordinates(),
        _ledger(),
        batch_size=4,
    )
    expected_steps = active_config().scientific.confirmation.accepted_live_assimilation_steps
    assert expected_steps == 500
    assert steps == expected_steps
    assert reference_ledger.remaining(TargetOptimizerBudgetCategory.LIVE_ASSIMILATION) == 0
    for actual, expected in zip(_parameters(model), _parameters(reference_model), strict=True):
        assert torch.equal(actual, expected)


def test_accepted_assimilation_applies_registered_multipliers_without_renormalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[torch.Tensor] = []
    original = assimilation_module.minibatch_objective

    def record(
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: ClassWeights,
        probability_floor: float,
        curriculum_multipliers: torch.Tensor,
    ) -> torch.Tensor:
        recorded.append(curriculum_multipliers.detach().clone())
        return original(logits, targets, class_weights, probability_floor, curriculum_multipliers)

    monkeypatch.setattr(assimilation_module, "minibatch_objective", record)
    alpha = (0.25, 0.0)
    multipliers = _curriculum_multipliers(alpha)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    clean = capture_pre_confirm_pair(model, optimizer).baseline
    features, targets = _features()
    apply_accepted_assimilation(
        model,
        optimizer,
        clean,
        features,
        targets,
        ClassWeights(torch.ones(2, dtype=torch.float32)),
        multipliers,
        _SEED,
        _coordinates(),
        _ledger(),
        batch_size=4,
    )
    assert len(recorded) == 500
    for observed in recorded:
        assert torch.equal(observed, multipliers.values)
    assert torch.equal(multipliers.values, torch.tensor([1.25, 1.0], dtype=torch.float32))


def test_rejected_proposal_settles_on_the_clean_pre_confirm_state() -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    clean = capture_pre_confirm_pair(model, optimizer).baseline
    clean_parameters = _parameters(model)
    features, targets = _features()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(features), targets).backward()
        optimizer.step()
    settle_rejected_proposal(model, optimizer, clean)
    for actual, expected in zip(_parameters(model), clean_parameters, strict=True):
        assert torch.equal(actual, expected)


def test_accepted_assimilation_requires_the_matching_ledger_identity() -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    clean = capture_pre_confirm_pair(model, optimizer).baseline
    features, targets = _features()
    wrong_seed_ledger = TargetOptimizerStepLedger(
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, _PAIR, _SEED + 1
    )
    try:
        apply_accepted_assimilation(
            model,
            optimizer,
            clean,
            features,
            targets,
            ClassWeights(torch.ones(2, dtype=torch.float32)),
            _curriculum_multipliers((0.2, 0.0)),
            _SEED,
            _coordinates(),
            wrong_seed_ledger,
            batch_size=4,
        )
    except ValueError:
        consumed = wrong_seed_ledger.remaining(TargetOptimizerBudgetCategory.LIVE_ASSIMILATION)
        registered = active_config().scientific.target_optimizer_budget.reserved.live_assimilation
        assert consumed == registered
    else:
        raise AssertionError("assimilation accepted a ledger for a different seed")


def test_meta_risk_refresh_rebuilds_target_importance_from_new_meta_risks() -> None:
    initial = build_target_importance(
        (
            TransferNodeRisk(0, True, 0.5),
            TransferNodeRisk(1, True, 0.5),
        )
    )
    refreshed = build_target_importance(
        (
            TransferNodeRisk(0, False, 0.9),
            TransferNodeRisk(1, True, 0.1),
        )
    )
    assert initial.as_vector(2).sum() == 1.0
    assert refreshed.as_vector(2).sum() == 1.0
    assert refreshed.as_vector(2)[0] == 0.0
    assert refreshed.as_vector(2)[1] == 1.0
    assert not torch.equal(torch.tensor(initial.as_vector(2)), torch.tensor(refreshed.as_vector(2)))


def test_scoring_route_consumes_the_registered_meta_split() -> None:
    model = _model()
    features, targets = _features()
    score = score_model(
        ScoringRequest(
            model,
            features,
            targets,
            LocalClassCount(2),
        )
    )
    assert score.macro_cross_entropy.value > 0.0
    assert score.class_conditional_cross_entropy.values
    assert SimpleNamespace(model=model).model is model
