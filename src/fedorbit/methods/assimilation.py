from __future__ import annotations

import itertools
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import torch

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.runtime import RandomSeed, SeedDerivationRequest, derive_seed32
from fedorbit.learning.training import (
    ClassWeights,
    ModelParameterState,
    OptimizerState,
    RngState,
    SelectedHyperparameters,
    backward_value,
    make_adamw,
    minibatch_objective,
    optimizer_step,
)
from fedorbit.methods.confirmation import (
    ConfirmReplicateOutcomes,
    hierarchical_bootstrap_lower_bound,
)
from fedorbit.methods.target import CurriculumMultipliers
from fedorbit.response.estimation import shadow_batch_schedule
from fedorbit.types import RngNamespace, StableJsonPayload


class AssimilationError(ValueError):
    pass


class TestOpeningRuleError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PreConfirmStatePair:
    baseline: PreConfirmTargetState
    curriculum: PreConfirmTargetState


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
        for name in ASSIMILATION_COORDINATE_KEYS:
            if not getattr(self, name):
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
    rng_state: RngState

    @classmethod
    def capture(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.AdamW,
    ) -> PreConfirmTargetState:
        return cls(
            model_state=ModelParameterState.capture(model),
            optimizer_state=OptimizerState.capture(optimizer),
            rng_state=RngState.capture(),
        )

    def restore_into(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.AdamW,
    ) -> None:
        self.model_state.load_into(model)
        self.optimizer_state.load_into(optimizer)
        self.rng_state.restore()


def capture_pre_confirm_pair(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
) -> PreConfirmStatePair:
    return PreConfirmStatePair(
        PreConfirmTargetState.capture(model, optimizer),
        PreConfirmTargetState.capture(model, optimizer),
    )


def _confirmation_batches_for_replicate(
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    seed: RandomSeed,
    contrast_coordinates: str,
    replicate_index: int,
    horizon: int,
) -> tuple[ShadowBatch, ...]:
    rng_seed = derive_seed32(
        SeedDerivationRequest(
            seed,
            RngNamespace.CONFIRMATION_SCHEDULE,
            cast(
                StableJsonPayload,
                OrderedDict(coordinates=contrast_coordinates, replicate=replicate_index),
            ),
        )
    )
    generator = torch.Generator().manual_seed(rng_seed)
    train_size = int(features.shape[0])
    if train_size <= 0:
        raise AssimilationError("confirmation TRAIN split is empty")
    schedule = shadow_batch_schedule(train_size, batch_size, generator)
    return tuple(
        ShadowBatch(features[indices], targets[indices])
        for indices in itertools.islice(schedule, horizon)
    )


def _step_shadow(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    state: PreConfirmTargetState,
    batches: tuple[ShadowBatch, ...],
    class_weights: ClassWeights,
    multipliers: CurriculumMultipliers,
) -> None:
    config = active_config()
    state.restore_into(model, optimizer)
    model.train()
    device = next(model.parameters()).device
    for batch in batches:
        targets = batch.targets.to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.features.to(device=device, dtype=torch.float32))
        loss = minibatch_objective(
            logits,
            targets,
            class_weights,
            config.scientific.metrics.probability_log_floor,
            multipliers.values,
        )
        if not bool(torch.isfinite(loss)):
            raise AssimilationError("non-finite confirmation shadow loss")
        backward_value(loss)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.scientific.training.gradient_clip_global_l2_norm,
            norm_type=2.0,
        )
        optimizer_step(optimizer)


def _confirm_class_losses(
    model: torch.nn.Module,
    confirm_features: torch.Tensor,
    confirm_targets: torch.Tensor,
    class_count: int,
    log_floor: float,
) -> tuple[torch.Tensor, ...]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(confirm_features.to(device=device, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).to(dtype=torch.float64)
        targets = confirm_targets.to(device=device, dtype=torch.long)
        picked = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        losses = -torch.log(torch.clamp(picked, min=log_floor)).cpu()
    target_values = confirm_targets.to(dtype=torch.long).cpu()
    tensors: list[torch.Tensor] = []
    for class_index in range(class_count):
        class_losses = losses[target_values == class_index]
        if class_losses.numel() == 0:
            raise AssimilationError("CONFIRM evaluation class has zero examples")
        tensors.append(class_losses.to(dtype=torch.float64))
    return tuple(tensors)


@dataclass(frozen=True, slots=True)
class ConfirmationVerdict:
    accepted: bool
    lower_bound: float
    acceptance_threshold: float


@dataclass(frozen=True, slots=True)
class ConfirmationRequest:
    model: torch.nn.Module
    pre_confirm_baseline: PreConfirmTargetState
    pre_confirm_curriculum: PreConfirmTargetState
    train_features: torch.Tensor
    train_targets: torch.Tensor
    confirm_features: torch.Tensor
    confirm_targets: torch.Tensor
    base_class_weights: ClassWeights
    curriculum_multipliers: CurriculumMultipliers
    selected_hyperparameters: SelectedHyperparameters
    seed: RandomSeed
    contrast_coordinates: str


def run_proposal_confirmation(
    request: ConfirmationRequest,
    batch_size: int | None = None,
) -> ConfirmationVerdict:
    config = active_config()
    confirmation = config.scientific.confirmation
    effective_batch = batch_size or config.scientific.training.batch_size
    if effective_batch <= 0:
        raise AssimilationError("confirmation batch size must be positive")
    if request.confirm_features.shape[0] == 0 or request.confirm_targets.shape[0] == 0:
        raise AssimilationError("CONFIRM split is empty")
    model = request.model
    class_count = int(request.base_class_weights.values.shape[0])
    neutral = CurriculumMultipliers(torch.ones_like(request.base_class_weights.values))
    replicated: list[ConfirmReplicateOutcomes] = []
    for replicate_index in range(confirmation.paired_replicates):
        batches = _confirmation_batches_for_replicate(
            request.train_features,
            request.train_targets,
            effective_batch,
            request.seed,
            request.contrast_coordinates,
            replicate_index,
            confirmation.optimizer_steps_per_shadow,
        )
        baseline_optimizer = make_adamw(
            model,
            request.selected_hyperparameters.learning_rate,
            request.selected_hyperparameters.weight_decay,
        )
        _step_shadow(
            model,
            baseline_optimizer,
            request.pre_confirm_baseline,
            batches,
            request.base_class_weights,
            neutral,
        )
        baseline_losses = _confirm_class_losses(
            model,
            request.confirm_features,
            request.confirm_targets,
            class_count,
            config.scientific.metrics.probability_log_floor,
        )
        curriculum_optimizer = make_adamw(
            model,
            request.selected_hyperparameters.learning_rate,
            request.selected_hyperparameters.weight_decay,
        )
        _step_shadow(
            model,
            curriculum_optimizer,
            request.pre_confirm_curriculum,
            batches,
            request.base_class_weights,
            request.curriculum_multipliers,
        )
        curriculum_losses = _confirm_class_losses(
            model,
            request.confirm_features,
            request.confirm_targets,
            class_count,
            config.scientific.metrics.probability_log_floor,
        )
        replicated.append(ConfirmReplicateOutcomes(baseline_losses, curriculum_losses))
    lower_bound = hierarchical_bootstrap_lower_bound(
        tuple(replicated),
        request.seed,
        request.contrast_coordinates,
    )
    threshold = confirmation.lower_bound_acceptance_threshold_relative_macro_ce
    return ConfirmationVerdict(lower_bound >= threshold, lower_bound, threshold)


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
    if train_size <= 0:
        raise AssimilationError("assimilation TRAIN split is empty")
    schedule = shadow_batch_schedule(train_size, batch_size, generator)
    for indices in itertools.islice(schedule, total_steps):
        yield ShadowBatch(features[indices], targets[indices])


def apply_accepted_assimilation(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    pre_confirm: PreConfirmTargetState,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    base_class_weights: ClassWeights,
    curriculum_multipliers: CurriculumMultipliers,
    seed: RandomSeed,
    assimilation_coordinates: AssimilationCoordinates,
    batch_size: int | None = None,
) -> int:
    config = active_config()
    effective_batch = batch_size or config.scientific.training.batch_size
    if effective_batch <= 0:
        raise AssimilationError("assimilation batch size must be positive")
    total_steps = config.scientific.confirmation.accepted_live_assimilation_steps
    coordinates_payload = OrderedDict(
        (name, getattr(assimilation_coordinates, name)) for name in ASSIMILATION_COORDINATE_KEYS
    )
    rng_seed = derive_seed32(
        SeedDerivationRequest(
            seed,
            RngNamespace.ASSIMILATION_SCHEDULE,
            coordinates_payload,
        )
    )
    generator = torch.Generator().manual_seed(rng_seed)
    pre_confirm.restore_into(model, optimizer)
    model.train()
    device = next(model.parameters()).device
    steps_executed = 0
    for batch in _assimilation_batches(
        train_features,
        train_targets,
        effective_batch,
        generator,
        total_steps,
    ):
        targets = batch.targets.to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.features.to(device=device, dtype=torch.float32))
        loss = minibatch_objective(
            logits,
            targets,
            base_class_weights,
            config.scientific.metrics.probability_log_floor,
            curriculum_multipliers.values,
        )
        if not bool(torch.isfinite(loss)):
            raise AssimilationError("non-finite live assimilation loss")
        backward_value(loss)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.scientific.training.gradient_clip_global_l2_norm,
            norm_type=2.0,
        )
        optimizer_step(optimizer)
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
        return TestAccessGrant(tuple(_PRE_TEST_PHASES))

    def assert_opened(self) -> None:
        if not self._opened:
            raise TestOpeningRuleError("TEST read attempted before TEST opening rule satisfied")
