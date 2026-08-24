from __future__ import annotations

from enum import StrEnum

from fedorbit.domain.enums import ClientRole


class ResourceKind(StrEnum):
    TRAIN = "TRAIN"
    META = "META"
    VALID = "VALID"
    CONFIRM = "CONFIRM"
    TEST = "TEST"
    LOCAL_FINE_LABELS = "local fine labels"
    LOCAL_COARSE_GROUP_MEMBERSHIP = "local coarse-group membership"
    LOCAL_PREPROCESSING = "local preprocessing"
    LOCAL_CLASSIFIER = "local classifier"
    LOCAL_OPTIMIZER_CHECKPOINT = "local optimizer/checkpoint"
    LOCAL_INTERVENTION_EXPERIMENTS = "local intervention experiments"
    SHARED_COARSE_GROUPS = "shared coarse groups"
    ANONYMOUS_SOURCE_PACKET = "anonymous source packet"
    LOCAL_MODEL_STATE = "local model state"
    LOCAL_IMPORTANCE_VECTOR = "local importance vector"


SOURCE_LOCAL_WHITELIST = frozenset(
    {
        ResourceKind.TRAIN,
        ResourceKind.META,
        ResourceKind.VALID,
        ResourceKind.LOCAL_FINE_LABELS,
        ResourceKind.LOCAL_COARSE_GROUP_MEMBERSHIP,
        ResourceKind.LOCAL_PREPROCESSING,
        ResourceKind.LOCAL_CLASSIFIER,
        ResourceKind.LOCAL_OPTIMIZER_CHECKPOINT,
        ResourceKind.LOCAL_INTERVENTION_EXPERIMENTS,
    }
)

TARGET_LOCAL_WHITELIST = frozenset(
    {
        ResourceKind.TRAIN,
        ResourceKind.META,
        ResourceKind.VALID,
        ResourceKind.CONFIRM,
        ResourceKind.LOCAL_FINE_LABELS,
        ResourceKind.SHARED_COARSE_GROUPS,
        ResourceKind.ANONYMOUS_SOURCE_PACKET,
        ResourceKind.LOCAL_MODEL_STATE,
        ResourceKind.LOCAL_IMPORTANCE_VECTOR,
    }
)


class StrictResourceViolationError(PermissionError):
    pass


class StrictResourcePolicy:
    def assert_source_allowed(self, resource: ResourceKind) -> None:
        if resource not in SOURCE_LOCAL_WHITELIST:
            raise StrictResourceViolationError(
                f"source-local resource outside the whitelist: {resource.value}"
            )

    def assert_target_allowed(self, resource: ResourceKind, transfer_finalized: bool) -> None:
        if resource == ResourceKind.TEST:
            if not transfer_finalized:
                raise StrictResourceViolationError(
                    "TEST is permitted only after the transfer decision is fully finalized"
                )
            return
        if resource not in TARGET_LOCAL_WHITELIST:
            raise StrictResourceViolationError(
                f"target-local resource outside the whitelist: {resource.value}"
            )

    def assert_role_allowed(
        self,
        role: ClientRole,
        resource: ResourceKind,
        transfer_finalized: bool = False,
    ) -> None:
        if role == ClientRole.SOURCE:
            self.assert_source_allowed(resource)
            return
        if role == ClientRole.TARGET:
            self.assert_target_allowed(resource, transfer_finalized)
            return
        raise StrictResourceViolationError(f"unknown client role: {role.value}")
