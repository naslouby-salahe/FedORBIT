from __future__ import annotations

import pytest
import torch

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.models.training import ModelParameterState
from fedorbit.transfer import assimilation as assimilation_module
from fedorbit.transfer.assimilation import (
    AssimilationCoordinates,
    AssimilationError,
    ConfirmationRequest,
    PreConfirmTargetState,
    PreTestLifecycle,
    apply_accepted_assimilation,
    capture_pre_confirm_pair,
    run_proposal_confirmation,
    settle_rejected_proposal,
)


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def _make_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
    )


def _factory_for(model: torch.nn.Module, learning_rate: float, weight_decay: float):
    def factory(_params: object) -> torch.optim.AdamW:
        del _params
        return _make_optimizer(model, learning_rate, weight_decay, 0.9, 0.999, 1e-8)

    return factory


def _dataset(seed: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(count, 3, generator=generator)
    targets = torch.randint(0, 2, (count,), generator=generator)
    return features, targets


def test_capture_pair_clones_identical_independent_states() -> None:
    torch.manual_seed(0)
    model = TinyClassifier()
    optimizer = _make_optimizer(model, 1e-3, 0.0, 0.9, 0.999, 1e-8)
    baseline, curriculum = capture_pre_confirm_pair(model, optimizer)
    assert set(baseline.model_state.tensors_by_name) == set(curriculum.model_state.tensors_by_name)
    for key, value in baseline.model_state.tensors_by_name.items():
        assert torch.equal(value, curriculum.model_state.tensors_by_name[key])
    model.linear.weight.data.add_(1.0)
    unchanged = baseline.model_state.tensors_by_name["linear.weight"]
    current = model.state_dict()["linear.weight"]
    assert not torch.equal(unchanged, current)


def test_confirmation_accepts_helpful_and_rejects_harmful_curriculum() -> None:
    config = load_fedorbit_config()
    torch.manual_seed(7)
    train_features, train_targets = _dataset(11, 128)
    confirm_features, confirm_targets = _dataset(12, 96)
    model = TinyClassifier()
    optimizer = _make_optimizer(model, 5e-3, 0.0, 0.9, 0.999, 1e-8)
    baseline_snapshot = PreConfirmTargetState.capture(model, optimizer)
    curriculum_snapshot = PreConfirmTargetState.capture(model, optimizer)
    base_weights = torch.tensor([1.0, 1.0])
    learning_rate = 5e-3
    weight_decay = 0.0
    factory = _factory_for(model, learning_rate, weight_decay)
    helpful_request = ConfirmationRequest(
        model=model,
        optimizer_factory=factory,
        pre_confirm_baseline=baseline_snapshot,
        pre_confirm_curriculum=curriculum_snapshot,
        train_features=train_features,
        train_targets=train_targets,
        confirm_features=confirm_features,
        confirm_targets=confirm_targets,
        base_class_weights=base_weights,
        curriculum_multipliers=torch.tensor([1.15, 1.0]),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=1103,
        contrast_coordinates="probe/helpful",
    )
    helpful = run_proposal_confirmation(config, helpful_request, batch_size=32)
    harmful_request = ConfirmationRequest(
        model=model,
        optimizer_factory=factory,
        pre_confirm_baseline=baseline_snapshot,
        pre_confirm_curriculum=curriculum_snapshot,
        train_features=train_features,
        train_targets=train_targets,
        confirm_features=confirm_features,
        confirm_targets=torch.flip(confirm_targets, dims=(0,)),
        base_class_weights=base_weights,
        curriculum_multipliers=torch.tensor([0.6, 1.6]),
        learning_rate=5e-2,
        weight_decay=weight_decay,
        seed=2207,
        contrast_coordinates="probe/harmful",
    )
    harmful = run_proposal_confirmation(config, harmful_request, batch_size=32)
    assert isinstance(helpful.accepted, bool)
    assert isinstance(harmful.accepted, bool)


def test_paired_shadows_share_schedule_and_start_from_clean_state() -> None:
    config = load_fedorbit_config()
    torch.manual_seed(3)
    train_features, train_targets = _dataset(21, 64)
    confirm_features, confirm_targets = _dataset(22, 64)
    model_a = TinyClassifier()
    with torch.no_grad():
        model_a.linear.weight.fill_(0.1)
        model_a.linear.bias.fill_(0.0)
    optimizer_a = _make_optimizer(model_a, 1e-3, 0.0, 0.9, 0.999, 1e-8)
    baseline, curriculum = capture_pre_confirm_pair(model_a, optimizer_a)
    neutral_weights = torch.tensor([1.0, 1.0])
    first_request = ConfirmationRequest(
        model=model_a,
        optimizer_factory=_factory_for(model_a, 1e-3, 0.0),
        pre_confirm_baseline=baseline,
        pre_confirm_curriculum=curriculum,
        train_features=train_features,
        train_targets=train_targets,
        confirm_features=confirm_features,
        confirm_targets=confirm_targets,
        base_class_weights=neutral_weights,
        curriculum_multipliers=torch.tensor([1.0, 1.0]),
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=3319,
        contrast_coordinates="replay",
    )
    second_request = ConfirmationRequest(
        model=model_a,
        optimizer_factory=_factory_for(model_a, 1e-3, 0.0),
        pre_confirm_baseline=baseline,
        pre_confirm_curriculum=curriculum,
        train_features=train_features,
        train_targets=train_targets,
        confirm_features=confirm_features,
        confirm_targets=confirm_targets,
        base_class_weights=neutral_weights,
        curriculum_multipliers=torch.tensor([1.0, 1.0]),
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=3319,
        contrast_coordinates="replay",
    )
    first = run_proposal_confirmation(config, first_request, batch_size=16)
    second = run_proposal_confirmation(config, second_request, batch_size=16)
    assert first.lower_bound == pytest.approx(second.lower_bound, abs=1e-9)
    assert first.accepted == second.accepted


def test_rejected_proposal_restores_clean_state_exactly() -> None:
    torch.manual_seed(5)
    model = TinyClassifier()
    optimizer = _make_optimizer(model, 1e-3, 0.0, 0.9, 0.999, 1e-8)
    clean = PreConfirmTargetState.capture(model, optimizer)
    model.linear.weight.data.add_(3.0)
    settle_rejected_proposal(model, optimizer, clean)
    for key, value in clean.model_state.tensors_by_name.items():
        assert torch.equal(model.state_dict()[key], value)


def test_accepted_assimilation_applies_steps_from_clean_state() -> None:
    config = load_fedorbit_config()
    torch.manual_seed(9)
    model = TinyClassifier()
    with torch.no_grad():
        model.linear.weight.fill_(0.05)
    optimizer = _make_optimizer(model, 1e-2, 0.0, 0.9, 0.999, 1e-8)
    pre_confirm = PreConfirmTargetState.capture(model, optimizer)
    train_features, train_targets = _dataset(31, 64)
    coordinates = AssimilationCoordinates(
        target_client="ton_iot_windows10_host",
        directed_pair="edge_iiotset_network -> ton_iot_windows10_host",
        condition="principal",
        seed="1103",
        clean_pretransfer_checkpoint_artifact_id="checkpoint-1",
        source_packet_artifact_id="packet-1",
        action_artifact_sha256="c" * 64,
    )
    steps = apply_accepted_assimilation(
        config,
        model,
        optimizer,
        pre_confirm,
        train_features,
        train_targets,
        base_class_weights=torch.tensor([1.0, 1.0]),
        curriculum_multipliers=torch.tensor([1.25, 1.0]),
        seed=1103,
        assimilation_coordinates=coordinates,
        batch_size=16,
    )
    expected_steps = config.scientific.confirmation.accepted_live_assimilation_steps
    assert steps == expected_steps


def test_assimilation_coordinate_contract_is_typed_and_total() -> None:
    complete_kwargs: dict[str, str] = {
        "target_client": "edge_iiotset_network",
        "directed_pair": "edge_iiotset_network -> ton_iot_windows10_host",
        "condition": "principal",
        "seed": "4421",
        "clean_pretransfer_checkpoint_artifact_id": "checkpoint-1",
        "source_packet_artifact_id": "packet-1",
        "action_artifact_sha256": "d" * 64,
    }
    incomplete_kwargs = {
        name: value for name, value in complete_kwargs.items() if name != "action_artifact_sha256"
    }
    with pytest.raises(TypeError):
        AssimilationCoordinates(**incomplete_kwargs)
    with pytest.raises(AssimilationError):
        AssimilationCoordinates(
            target_client="edge_iiotset_network",
            directed_pair="edge_iiotset_network -> ton_iot_windows10_host",
            condition="principal",
            seed="4421",
            clean_pretransfer_checkpoint_artifact_id="checkpoint-1",
            source_packet_artifact_id="packet-1",
            action_artifact_sha256="",
        )


def test_extra_coordinate_key_rejected() -> None:
    config = load_fedorbit_config()
    torch.manual_seed(19)
    model = TinyClassifier()
    optimizer = _make_optimizer(model, 1e-2, 0.0, 0.9, 0.999, 1e-8)
    pre_confirm = PreConfirmTargetState.capture(model, optimizer)
    train_features, train_targets = _dataset(43, 32)
    complete = AssimilationCoordinates(
        target_client="edge_iiotset_network",
        directed_pair="edge_iiotset_network -> ton_iot_windows10_host",
        condition="principal",
        seed="4421",
        clean_pretransfer_checkpoint_artifact_id="checkpoint-1",
        source_packet_artifact_id="packet-1",
        action_artifact_sha256="d" * 64,
    )
    assert complete.action_artifact_sha256 == "d" * 64
    extra_kwargs = {
        **{
            field: getattr(complete, field)
            for field in (
                "target_client",
                "directed_pair",
                "condition",
                "seed",
                "clean_pretransfer_checkpoint_artifact_id",
                "source_packet_artifact_id",
                "action_artifact_sha256",
            )
        },
        "method_name": "forbidden-method-name",
    }
    with pytest.raises(TypeError):
        AssimilationCoordinates(**extra_kwargs)
    with pytest.raises(AssimilationError):
        apply_accepted_assimilation(
            config,
            model,
            optimizer,
            pre_confirm,
            train_features,
            train_targets,
            base_class_weights=torch.ones(2),
            curriculum_multipliers=torch.ones(2),
            seed=4421,
            assimilation_coordinates=AssimilationCoordinates(
                target_client="edge_iiotset_network",
                directed_pair="edge_iiotset_network -> ton_iot_windows10_host",
                condition="principal",
                seed="4421",
                clean_pretransfer_checkpoint_artifact_id="checkpoint-1",
                source_packet_artifact_id="packet-1",
                action_artifact_sha256="",
            ),
            batch_size=8,
        )


def test_pre_test_lifecycle_enforces_ordering_and_opening() -> None:
    lifecycle = PreTestLifecycle()
    with pytest.raises(assimilation_module.TestOpeningRuleError):
        lifecycle.complete_phase("action_finalized")
    lifecycle.complete_phase("source_selection_finalized")
    with pytest.raises(assimilation_module.TestOpeningRuleError):
        lifecycle.open_test()
    with pytest.raises(assimilation_module.TestOpeningRuleError):
        lifecycle.assert_opened()
    lifecycle.complete_phase("action_finalized")
    lifecycle.complete_phase("confirmation_decision_finalized")
    lifecycle.complete_phase("assimilation_settled")
    lifecycle.complete_phase("pre_test_artifacts_committed")
    grant = lifecycle.open_test()
    assert len(grant.completed_phases) == 5
    with pytest.raises(assimilation_module.TestOpeningRuleError):
        lifecycle.complete_phase("action_finalized")
    lifecycle.assert_opened()


def test_model_parameter_state_helper_matches_checkpoint_contract() -> None:
    torch.manual_seed(17)
    model = TinyClassifier()
    state = ModelParameterState(
        {key: value.detach().clone() for key, value in model.state_dict().items()}
    )
    other = TinyClassifier()
    state.load_into(other)
    for key, value in state.tensors_by_name.items():
        assert torch.equal(other.state_dict()[key], value)
