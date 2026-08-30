from __future__ import annotations

import pytest
import torch

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.models.network_classifier import NetworkFlowClassifier
from fedorbit.training.losses import ClassWeights
from fedorbit.training.trainer import make_adamw
from fedorbit.transfer.assimilation import (
    AssimilationCoordinates,
    AssimilationError,
    PreConfirmTargetState,
    PreTestLifecycle,
    apply_accepted_assimilation,
    capture_pre_confirm_pair,
    settle_rejected_proposal,
)
from fedorbit.transfer.assimilation import TestOpeningRuleError as OpeningRuleError
from fedorbit.transfer.curriculum import CurriculumMultipliers


def _model_and_optimizer() -> tuple[NetworkFlowClassifier, torch.optim.AdamW]:
    model = NetworkFlowClassifier(3, 2, 0.0)
    model.initialize(torch.Generator().manual_seed(7))
    return model, make_adamw(model, 1e-3, 0.0)


def test_pre_confirm_capture_restores_model_optimizer_and_rng() -> None:
    model, optimizer = _model_and_optimizer()
    state = PreConfirmTargetState.capture(model, optimizer)
    before = tuple(entry.value.clone() for entry in state.model_state.tensors)
    for parameter in model.parameters():
        parameter.data.add_(1.0)
    state.restore_into(model, optimizer)
    after = tuple(value.detach().cpu() for value in model.state_dict().values())
    assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))


def test_capture_pair_uses_independent_identical_clean_snapshots() -> None:
    model, optimizer = _model_and_optimizer()
    states = capture_pre_confirm_pair(model, optimizer)
    baseline = states.baseline
    curriculum = states.curriculum
    assert baseline is not curriculum
    assert baseline.optimizer_state.payload == curriculum.optimizer_state.payload
    assert torch.equal(baseline.rng_state.cpu, curriculum.rng_state.cpu)
    assert len(baseline.rng_state.cuda) == len(curriculum.rng_state.cuda)
    assert all(
        torch.equal(left, right)
        for left, right in zip(baseline.rng_state.cuda, curriculum.rng_state.cuda, strict=True)
    )
    assert tuple(entry.name for entry in baseline.model_state.tensors) == tuple(
        entry.name for entry in curriculum.model_state.tensors
    )
    assert all(
        torch.equal(left.value, right.value)
        for left, right in zip(
            baseline.model_state.tensors,
            curriculum.model_state.tensors,
            strict=True,
        )
    )


def test_rejected_proposal_restores_clean_state() -> None:
    model, optimizer = _model_and_optimizer()
    state = PreConfirmTargetState.capture(model, optimizer)
    for parameter in model.parameters():
        parameter.data.mul_(0.0)
    settle_rejected_proposal(model, optimizer, state)
    current = tuple(value.detach().cpu() for value in model.state_dict().values())
    expected = tuple(entry.value for entry in state.model_state.tensors)
    assert all(torch.equal(left, right) for left, right in zip(current, expected, strict=True))


def test_live_assimilation_uses_registered_step_count_and_coordinates() -> None:
    config = load_fedorbit_config()
    model, optimizer = _model_and_optimizer()
    clean = PreConfirmTargetState.capture(model, optimizer)
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(32, 3, generator=generator)
    targets = torch.tensor([0, 1] * 16)
    coordinates = AssimilationCoordinates(
        target_client="target",
        directed_pair="source -> target",
        condition="principal",
        seed="1103",
        clean_pretransfer_checkpoint_artifact_id="checkpoint-1",
        source_packet_artifact_id="packet-1",
        action_artifact_sha256="d" * 64,
    )
    steps = apply_accepted_assimilation(
        model,
        optimizer,
        clean,
        features,
        targets,
        ClassWeights.from_targets(targets, 2),
        CurriculumMultipliers(torch.tensor([1.25, 1.0])),
        1103,
        coordinates,
        batch_size=16,
    )
    assert steps == config.scientific.confirmation.accepted_live_assimilation_steps


def test_assimilation_coordinates_are_total_and_method_free() -> None:
    with pytest.raises(AssimilationError):
        AssimilationCoordinates(
            target_client="target",
            directed_pair="source -> target",
            condition="principal",
            seed="1103",
            clean_pretransfer_checkpoint_artifact_id="checkpoint-1",
            source_packet_artifact_id="packet-1",
            action_artifact_sha256="",
        )


def test_pre_test_lifecycle_fails_closed_and_enforces_order() -> None:
    lifecycle = PreTestLifecycle()
    with pytest.raises(OpeningRuleError):
        lifecycle.complete_phase("action_finalized")
    lifecycle.complete_phase("source_selection_finalized")
    lifecycle.complete_phase("action_finalized")
    lifecycle.complete_phase("confirmation_decision_finalized")
    lifecycle.complete_phase("assimilation_settled")
    with pytest.raises(OpeningRuleError):
        lifecycle.open_test()
    lifecycle.complete_phase("pre_test_artifacts_committed")
    grant = lifecycle.open_test()
    assert grant.completed_phases[-1] == "pre_test_artifacts_committed"
    lifecycle.assert_opened()
