from __future__ import annotations

from dataclasses import dataclass, fields

from fedorbit.domain.enums import TransferMethod


class FairnessViolationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ComparatorResources:
    source_packet_id: str
    target_checkpoint_artifact_id: str
    target_importance_vector_sha256: str
    action_budget_cap: float
    support_cap: int
    seed: int
    confirmation_opportunity: bool
    live_assimilation_step_allowance: int
    test_access_granted: bool
    extra_target_labels: bool
    additional_tuning_seeds: tuple[int, ...]
    local_base_checkpoint_favorable: bool

    def validate_contract(self) -> None:
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.test_access_granted:
            raise FairnessViolationError(
                "comparator resources must never include pre-decision TEST access"
            )
        if self.extra_target_labels:
            raise FairnessViolationError("comparator resources must never add target labels")
        if self.additional_tuning_seeds:
            raise FairnessViolationError("comparator resources must never add tuning seeds")
        if self.local_base_checkpoint_favorable:
            raise FairnessViolationError(
                "comparator resources must never grant a more favorable base checkpoint"
            )


def assert_identical_resources(
    method_name: str,
    reference: ComparatorResources,
    candidate: ComparatorResources,
) -> None:
    for field in fields(ComparatorResources):
        if getattr(reference, field.name) != getattr(candidate, field.name):
            raise FairnessViolationError(
                f"method {method_name} received different {field.name} from the principal bundle"
            )


REGISTERED_METHOD_NAMES = frozenset(method.value for method in TransferMethod)


def assert_registered_method_name(name: str) -> None:
    if name not in REGISTERED_METHOD_NAMES:
        raise FairnessViolationError(f"unregistered comparator name: {name}")
