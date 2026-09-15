from __future__ import annotations

import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fedorbit.learning.pilot as learning_pilot
import fedorbit.learning.training as learning_training
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.learning.pilot import (
    PilotConfiguration,
    PilotData,
    PilotFitResult,
    pilot_grid,
    run_base_model_pilot,
    select_pilot_configuration,
)
from fedorbit.learning.training import (
    BaseCheckpoint,
    ClassWeights,
    ModelParameterState,
    OptimizerState,
    RngState,
    SelectedHyperparameters,
    TrainingOutcome,
    make_adamw,
    minibatch_objective,
    per_example_weighted_cross_entropy,
    train_base_model,
)
from fedorbit.types import ConceptCount, DatasetId, FeatureCount, Floor, Fraction, RandomSeed, Score


def _outcome(configuration: PilotConfiguration, metric: float) -> TrainingOutcome:
    checkpoint = BaseCheckpoint(
        epoch=0,
        valid_macro_cross_entropy=metric,
        state_dict=ModelParameterState(()),
        optimizer_state=OptimizerState(b""),
        rng_state=RngState(torch.get_rng_state().clone(), ()),
        selected_hyperparameters=configuration.hyperparameters(),
        train_class_weights=ClassWeights(torch.ones(2)),
    )
    return TrainingOutcome(checkpoint, 1)


def _fits(
    configuration: PilotConfiguration, values: tuple[float, float, float]
) -> tuple[PilotFitResult, ...]:
    return tuple(
        PilotFitResult(configuration, seed, _outcome(configuration, value))
        for seed, value in zip((101, 202, 303), values, strict=True)
    )


def test_pilot_grid_is_exact_registered_cartesian_product() -> None:
    config = load_fedorbit_config()
    grid = pilot_grid()
    assert len(grid) == 12
    assert len(set(grid)) == 12
    assert {item.learning_rate for item in grid} == {0.0003, 0.001, 0.003}
    assert {item.weight_decay for item in grid} == {0.0, 0.0001}
    assert {item.dropout for item in grid} == {0.0, 0.1}
    assert tuple(config.scientific.randomness.pilot_seeds) == (101, 202, 303)


def reference_learning_rate() -> float:
    return load_fedorbit_config().scientific.base_model_pilot.reference_learning_rate


def test_pilot_selection_uses_registered_tie_order() -> None:
    grid = pilot_grid()
    results = tuple(fit for candidate in grid for fit in _fits(candidate, (1.0, 1.0, 1.0)))
    selection = select_pilot_configuration(results)
    assert selection.configuration == PilotConfiguration(reference_learning_rate(), 0.0, 0.0)
    assert selection.median_valid_macro_cross_entropy == pytest.approx(1.0)


def test_pilot_selection_rejects_incomplete_grid() -> None:
    candidate = PilotConfiguration(reference_learning_rate(), 0.0, 0.0)
    with pytest.raises(ValueError):
        select_pilot_configuration(_fits(candidate, (1.0, 1.0, 1.0)))


def test_class_weights_have_the_registered_inverse_frequency_normalization() -> None:
    targets = torch.tensor((0, 0, 0, 1), dtype=torch.long)
    weights = ClassWeights.from_targets(targets, 2)

    assert weights.values.tolist() == pytest.approx((2 / 3, 2.0))
    assert float(weights.values[targets].mean()) == pytest.approx(1.0)


def test_minibatch_objective_means_weighted_losses_over_examples() -> None:
    logits = torch.zeros((2, 2), dtype=torch.float32)
    targets = torch.tensor((0, 1), dtype=torch.long)
    class_weights = ClassWeights(torch.tensor((2.0, 3.0), dtype=torch.float32))
    multipliers = torch.tensor((4.0, 5.0), dtype=torch.float32)

    losses = per_example_weighted_cross_entropy(logits, targets, class_weights, 1e-12, multipliers)
    expected_losses = torch.tensor((8.0 * math.log(2.0), 15.0 * math.log(2.0)))
    expected_objective = (8.0 * math.log(2.0) + 15.0 * math.log(2.0)) / 2

    assert torch.allclose(losses, expected_losses)
    assert float(minibatch_objective(logits, targets, class_weights, 1e-12, multipliers)) == (
        pytest.approx(expected_objective)
    )


def test_base_training_keeps_final_batch_and_earliest_near_tied_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = SimpleNamespace(
        dataloader_workers=0,
        label_smoothing=0.0,
        maximum_epochs=3,
        batch_size=2,
        gradient_clip_global_l2_norm=1.0,
        adamw=SimpleNamespace(
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            amsgrad=False,
        ),
        checkpoint=SimpleNamespace(tie_tolerance=0.1),
        early_stopping=SimpleNamespace(
            minimum_improvement=0.1,
            patience_completed_epochs=1,
        ),
    )
    monkeypatch.setattr(
        learning_training,
        "active_config",
        lambda: SimpleNamespace(
            scientific=SimpleNamespace(
                training=training,
                metrics=SimpleNamespace(probability_log_floor=1e-12),
            )
        ),
    )
    observed_batch_sizes: list[int] = []
    original_step = learning_training.optimizer_step

    def count_step(optimizer: torch.optim.Optimizer) -> None:
        observed_batch_sizes.append(1)
        original_step(optimizer)

    metric_values = iter((1.0, 0.95))

    def near_tied_metric(
        logits: torch.Tensor,
        targets: torch.Tensor,
        probability_log_floor: Floor,
    ) -> Score:
        del logits, targets, probability_log_floor
        return next(metric_values)

    monkeypatch.setattr(learning_training, "optimizer_step", count_step)
    monkeypatch.setattr(learning_training, "macro_cross_entropy", near_tied_metric)

    outcome = train_base_model(
        nn.Linear(2, 2),
        torch.tensor(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))),
        torch.tensor((0, 1, 0)),
        torch.tensor(((1.0, 0.0), (0.0, 1.0))),
        torch.tensor((0, 1)),
        ClassWeights(torch.ones(2)),
        101,
        SelectedHyperparameters(1e-3, 0.0, 0.0),
    )

    assert len(observed_batch_sizes) == 4
    assert outcome.completed_epochs == 2
    assert outcome.checkpoint.epoch == 0


def test_base_checkpoint_restores_paired_shadow_next_step_exactly() -> None:
    torch.manual_seed(67)
    source = nn.Linear(2, 2)
    optimizer = make_adamw(source, 1e-3, 0.0)
    inputs = torch.tensor(((1.0, 0.0), (0.0, 1.0)))
    source(inputs).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoint = BaseCheckpoint(
        epoch=3,
        valid_macro_cross_entropy=0.25,
        state_dict=ModelParameterState.capture(source),
        optimizer_state=OptimizerState.capture(optimizer),
        rng_state=RngState.capture(),
        selected_hyperparameters=SelectedHyperparameters(1e-3, 0.0, 0.1),
        train_class_weights=ClassWeights(torch.tensor((0.5, 1.5))),
    )
    first = nn.Linear(2, 2)
    second = nn.Linear(2, 2)
    first_optimizer = make_adamw(first, 1e-3, 0.0)
    second_optimizer = make_adamw(second, 1e-3, 0.0)

    checkpoint.restore(first, first_optimizer)
    first_random = torch.rand(3)
    first(inputs).sum().backward()
    first_optimizer.step()
    checkpoint.restore(second, second_optimizer)
    second_random = torch.rand(3)
    second(inputs).sum().backward()
    second_optimizer.step()

    assert checkpoint.epoch == 3
    assert checkpoint.selected_hyperparameters == SelectedHyperparameters(1e-3, 0.0, 0.1)
    assert torch.equal(checkpoint.train_class_weights.values, torch.tensor((0.5, 1.5)))
    assert torch.equal(first_random, second_random)
    assert all(
        torch.equal(first_value, second_value)
        for first_value, second_value in zip(first.parameters(), second.parameters(), strict=True)
    )


def test_base_model_pilot_expands_only_train_valid_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = PilotData(
        train_features=torch.tensor(((1.0, 0.0), (0.0, 1.0))),
        train_targets=torch.tensor((0, 1)),
        valid_features=torch.tensor(((1.0, 1.0), (0.0, 0.0))),
        valid_targets=torch.tensor((0, 1)),
        n_classes=2,
    )
    calls: list[PilotConfiguration] = []

    def fake_train(
        model: nn.Module,
        train_features: torch.Tensor,
        train_targets: torch.Tensor,
        valid_features: torch.Tensor,
        valid_targets: torch.Tensor,
        class_weights: ClassWeights,
        seed: RandomSeed,
        selected_hyperparameters: SelectedHyperparameters,
    ) -> TrainingOutcome:
        del model, class_weights, seed
        assert train_features is data.train_features
        assert train_targets is data.train_targets
        assert valid_features is data.valid_features
        assert valid_targets is data.valid_targets
        calls.append(
            PilotConfiguration(
                selected_hyperparameters.learning_rate,
                selected_hyperparameters.weight_decay,
                selected_hyperparameters.dropout_probability,
            )
        )
        return _outcome(calls[-1], 1.0)

    def fake_classifier(
        dataset: DatasetId,
        input_dimension: FeatureCount,
        n_classes: ConceptCount,
        dropout_probability: Fraction,
        seed: RandomSeed,
        device: torch.device | None = None,
    ) -> nn.Module:
        del dataset, dropout_probability, seed, device
        return nn.Linear(input_dimension, n_classes)

    monkeypatch.setattr(learning_pilot, "principal_determinism", nullcontext)
    monkeypatch.setattr(learning_pilot, "create_classifier", fake_classifier)
    monkeypatch.setattr(learning_pilot, "train_base_model", fake_train)

    results = run_base_model_pilot(data, DatasetId.TON_IOT_WINDOWS10_HOST)

    assert set(PilotData.__dataclass_fields__) == {
        "train_features",
        "train_targets",
        "valid_features",
        "valid_targets",
        "n_classes",
    }
    assert len(results) == len(pilot_grid()) * 3
    assert calls == [configuration for configuration in pilot_grid() for _ in range(3)]
