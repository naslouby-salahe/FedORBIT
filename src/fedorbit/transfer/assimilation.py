from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch

from fedorbit.config.models import FedorbitConfig, TrainingConfig
from fedorbit.domain.enums import RngNamespace
from fedorbit.models.training import ModelParameterState, OptimizerState
from fedorbit.runtime.seeds import derive_seed32
from fedorbit.transfer.confirmation import (
    ConfirmReplicateOutcomes,
    hierarchical_bootstrap_lower_bound,
)


class AssimilationError(ValueError):
    pass


class TestOpeningRuleError(RuntimeError):
    pass


ASSIMILATION_COORDINATE_KEYS = (
    "target_client",
    "directed_pair",
    "condition",
    "seed",
    "clean_pretransfer_checkpoint_artifact_id",
    "source_packet_artifact_id",
    "action_artifact_sha256",
)


@dataclass(frozen=True, slots=True)
class AssimilationCoordinates:
    target_client: str
    directed_pair: str
    condition: str
    seed: str
    clean_pretransfer_checkpoint_artifact_id: str
    source_packet_artifact_id: str
    action_artifact_sha256: str

    def __post_init__(self) -> None:
        values = {
            "target_client": self.target_client,
            "directed_pair": self.directed_pair,
            "condition": self.condition,
            "seed": self.seed,
            "clean_pretransfer_checkpoint_artifact_id": (
                self.clean_pretransfer_checkpoint_artifact_id
            ),
            "source_packet_artifact_id": self.source_packet_artifact_id,
            "action_artifact_sha256": self.action_artifact_sha256,
        }
        for name, value in values.items():
            if not value:
                raise AssimilationError(
                    f"assimilation coordinate {name} must be a non-empty string"
                )


@dataclass(frozen=True, slots=True)
class ShadowBatch:
    features: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True, slots=True)
class PreConfirmTargetState:
    model_state: ModelParameterState
    optimizer_state: OptimizerState
    rng_state: torch.Tensor

    @classmethod
    def capture(cls, model: torch.nn.Module, optimizer: torch.optim.AdamW) -> PreConfirmTargetState:
        return cls(
            model_state=ModelParameterState(
                {key: value.detach().clone() for key, value in model.state_dict().items()}
            ),
            optimizer_state=optimizer.state_dict(),
            rng_state=torch.get_rng_state(),
        )

    def restore_into(self, model: torch.nn.Module, optimizer: torch.optim.AdamW) -> None:
        self.model_state.load_into(model)
        optimizer.load_state_dict(self.optimizer_state)
        torch.set_rng_state(self.rng_state)


def capture_pre_confirm_pair(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
) -> tuple[PreConfirmTargetState, PreConfirmTargetState]:
    baseline = PreConfirmTargetState.capture(model, optimizer)
    curriculum = PreConfirmTargetState.capture(model, optimizer)
    return baseline, curriculum


def _confirmation_batches_for_replicate(
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    seed: int,
    contrast_coordinates: str,
    replicate_index: int,
    horizon: int,
) -> list[ShadowBatch]:
    rng_seed = derive_seed32(
        seed,
        RngNamespace.CONFIRMATION_SCHEDULE,
        {"coordinates": contrast_coordinates, "replicate": replicate_index},
    )
    generator = torch.Generator().manual_seed(rng_seed)
    train_size = int(features.shape[0])
    batches: list[ShadowBatch] = []
    permutation = torch.randperm(train_size, generator=generator)
    position = 0
    while len(batches) < horizon:
        if position >= train_size:
            permutation = torch.randperm(train_size, generator=generator)
            position = 0
        indices = permutation[position : position + batch_size]
        batches.append(ShadowBatch(features=features[indices], targets=targets[indices]))
        position += int(indices.shape[0])
    return batches


def _step_shadow(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    state: PreConfirmTargetState,
    batches: list[ShadowBatch],
    class_weight_vector: torch.Tensor,
    training_settings: TrainingConfig,
) -> None:
    state.restore_into(model, optimizer)
    criterion = torch.nn.CrossEntropyLoss(
        reduction="none", label_smoothing=training_settings.label_smoothing
    )
    model.train()
    for batch in batches:
        model.zero_grad()
        logits = model(batch.features.float())
        loss_per_example = criterion(logits, batch.targets)
        weights = class_weight_vector[batch.targets]
        loss = (loss_per_example * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), training_settings.gradient_clip_global_l2_norm
        )
        optimizer.step()


def _confirm_class_losses(
    model: torch.nn.Module,
    confirm_features: torch.Tensor,
    confirm_targets: torch.Tensor,
    class_count: int,
    log_floor: float,
) -> tuple[torch.Tensor, ...]:
    model.eval()
    with torch.no_grad():
        logits = model(confirm_features.float())
        probabilities = torch.softmax(logits, dim=1)
        picked = probabilities.gather(1, confirm_targets.unsqueeze(1)).squeeze(1)
        losses = -torch.log(torch.clamp(picked, min=log_floor))
    classes: list[list[float]] = [[] for _ in range(class_count)]
    for index in range(int(confirm_targets.shape[0])):
        classes[int(confirm_targets[index])].append(float(losses[index]))
    tensors: list[torch.Tensor] = []
    for examples in classes:
        if not examples:
            raise AssimilationError("CONFIRM evaluation class has zero examples")
        tensors.append(torch.tensor(examples, dtype=torch.float64))
    return tuple(tensors)


@dataclass(frozen=True, slots=True)
class ConfirmationVerdict:
    accepted: bool
    lower_bound: float
    acceptance_threshold: float


def run_proposal_confirmation(
    config: FedorbitConfig,
    model: torch.nn.Module,
    optimizer_factory: Callable[[object], torch.optim.AdamW],
    pre_confirm_baseline: PreConfirmTargetState,
    pre_confirm_curriculum: PreConfirmTargetState,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    confirm_features: torch.Tensor,
    confirm_targets: torch.Tensor,
    base_class_weights: torch.Tensor,
    curriculum_multipliers: torch.Tensor,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    contrast_coordinates: str,
    batch_size: int | None = None,
) -> ConfirmationVerdict:
    confirmation = config.scientific.confirmation
    training = config.scientific.training
    effective_batch = batch_size if batch_size is not None else training.batch_size
    if effective_batch <= 0:
        raise AssimilationError("confirmation batch size must be positive")
    if confirm_features.shape[0] == 0 or confirm_targets.shape[0] == 0:
        raise AssimilationError("CONFIRM split is empty")

    del learning_rate, weight_decay

    def make_optimizer() -> torch.optim.AdamW:
        return optimizer_factory(model.parameters())

    curriculum_weight_vector = base_class_weights * curriculum_multipliers
    log_floor = config.scientific.metrics.probability_log_floor
    class_count = int(base_class_weights.shape[0])
    replicated: list[ConfirmReplicateOutcomes] = []
    for replicate_index in range(confirmation.paired_replicates):
        batches = _confirmation_batches_for_replicate(
            train_features,
            train_targets,
            effective_batch,
            seed,
            contrast_coordinates,
            replicate_index,
            confirmation.optimizer_steps_per_shadow,
        )
        baseline_optimizer = make_optimizer()
        _step_shadow(
            model,
            baseline_optimizer,
            pre_confirm_baseline,
            batches,
            base_class_weights,
            training,
        )
        baseline_losses = _confirm_class_losses(
            model, confirm_features, confirm_targets, class_count, log_floor
        )
        curriculum_optimizer = make_optimizer()
        _step_shadow(
            model,
            curriculum_optimizer,
            pre_confirm_curriculum,
            batches,
            curriculum_weight_vector,
            training,
        )
        curriculum_losses = _confirm_class_losses(
            model, confirm_features, confirm_targets, class_count, log_floor
        )
        replicated.append(ConfirmReplicateOutcomes(baseline_losses, curriculum_losses))
    lower_bound = hierarchical_bootstrap_lower_bound(
        config, tuple(replicated), seed, contrast_coordinates
    )
    threshold = confirmation.lower_bound_acceptance_threshold_relative_macro_ce
    return ConfirmationVerdict(
        accepted=lower_bound >= threshold,
        lower_bound=lower_bound,
        acceptance_threshold=threshold,
    )


def settle_rejected_proposal(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    pre_confirm: PreConfirmTargetState,
) -> None:
    pre_confirm.restore_into(model, optimizer)


def _assimilation_batches(
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
    total_steps: int,
) -> Iterator[ShadowBatch]:
    train_size = int(features.shape[0])
    produced = 0
    permutation = torch.randperm(train_size, generator=generator)
    position = 0
    while produced < total_steps:
        if position >= train_size:
            permutation = torch.randperm(train_size, generator=generator)
            position = 0
        indices = permutation[position : position + batch_size]
        yield ShadowBatch(features=features[indices], targets=targets[indices])
        position += int(indices.shape[0])
        produced += 1


def apply_accepted_assimilation(
    config: FedorbitConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    pre_confirm: PreConfirmTargetState,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    base_class_weights: torch.Tensor,
    curriculum_multipliers: torch.Tensor,
    seed: int,
    assimilation_coordinates: AssimilationCoordinates,
    batch_size: int | None = None,
) -> int:
    training = config.scientific.training
    confirmation = config.scientific.confirmation
    effective_batch = batch_size if batch_size is not None else training.batch_size
    if effective_batch <= 0:
        raise AssimilationError("assimilation batch size must be positive")
    total_steps = confirmation.accepted_live_assimilation_steps
    coordinates_payload = {
        name: getattr(assimilation_coordinates, name) for name in ASSIMILATION_COORDINATE_KEYS
    }
    rng_seed = derive_seed32(seed, RngNamespace.ASSIMILATION_SCHEDULE, coordinates_payload)
    generator = torch.Generator().manual_seed(rng_seed)
    pre_confirm.restore_into(model, optimizer)
    curriculum_weight_vector = base_class_weights * curriculum_multipliers
    criterion = torch.nn.CrossEntropyLoss(
        reduction="none", label_smoothing=training.label_smoothing
    )
    model.train()
    steps_executed = 0
    for batch in _assimilation_batches(
        train_features, train_targets, effective_batch, generator, total_steps
    ):
        model.zero_grad()
        logits = model(batch.features.float())
        loss_per_example = criterion(logits, batch.targets)
        weights = curriculum_weight_vector[batch.targets]
        loss = (loss_per_example * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_global_l2_norm)
        optimizer.step()
        steps_executed += 1
    if steps_executed != total_steps:
        raise AssimilationError(
            f"live assimilation executed {steps_executed} of {total_steps} steps"
        )
    return steps_executed


_PRE_TEST_PHASES = (
    "source_selection_finalized",
    "action_finalized",
    "confirmation_decision_finalized",
    "assimilation_settled",
    "pre_test_artifacts_committed",
)


@dataclass(frozen=True, slots=True)
class TestAccessGrant:
    completed_phases: tuple[str, ...]


class PreTestLifecycle:
    def __init__(self) -> None:
        self._completed: list[str] = []
        self._opened = False

    def complete_phase(self, phase: str) -> None:
        if phase not in _PRE_TEST_PHASES:
            raise TestOpeningRuleError(f"unknown pre-TEST phase: {phase}")
        if phase in self._completed:
            raise TestOpeningRuleError(f"phase already finalized: {phase}")
        index = _PRE_TEST_PHASES.index(phase)
        missing = [name for name in _PRE_TEST_PHASES[:index] if name not in self._completed]
        if missing:
            raise TestOpeningRuleError(f"phases out of order; missing {missing} before {phase}")
        self._completed.append(phase)

    @property
    def opened(self) -> bool:
        return self._opened

    def require_closed_before(self, phase: str) -> None:
        if self._opened:
            raise TestOpeningRuleError(f"TEST already opened before {phase}")

    def open_test(self) -> TestAccessGrant:
        missing = [name for name in _PRE_TEST_PHASES if name not in self._completed]
        if missing:
            raise TestOpeningRuleError(f"TEST opened early; missing phases: {missing}")
        self._opened = True
        return TestAccessGrant(completed_phases=tuple(_PRE_TEST_PHASES))

    def assert_opened(self) -> None:
        if not self._opened:
            raise TestOpeningRuleError("TEST read attempted before TEST opening rule satisfied")
