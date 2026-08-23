from __future__ import annotations

import pytest

from fedorbit.domain.enums import ClientRole
from fedorbit.strict_interface.resources import (
    ResourceKind,
    StrictResourcePolicy,
    StrictResourceViolationError,
)


def test_source_resource_whitelist() -> None:
    policy = StrictResourcePolicy()
    for resource in (
        ResourceKind.TRAIN,
        ResourceKind.META,
        ResourceKind.VALID,
        ResourceKind.LOCAL_FINE_LABELS,
        ResourceKind.LOCAL_COARSE_GROUP_MEMBERSHIP,
        ResourceKind.LOCAL_PREPROCESSING,
        ResourceKind.LOCAL_CLASSIFIER,
        ResourceKind.LOCAL_OPTIMIZER_CHECKPOINT,
        ResourceKind.LOCAL_INTERVENTION_EXPERIMENTS,
    ):
        policy.assert_role_allowed(ClientRole.SOURCE, resource)
    for resource in (ResourceKind.CONFIRM, ResourceKind.TEST, ResourceKind.ANONYMOUS_SOURCE_PACKET):
        with pytest.raises(StrictResourceViolationError):
            policy.assert_role_allowed(ClientRole.SOURCE, resource, transfer_finalized=True)


def test_target_resource_whitelist_and_test_gate() -> None:
    policy = StrictResourcePolicy()
    for resource in (
        ResourceKind.TRAIN,
        ResourceKind.META,
        ResourceKind.VALID,
        ResourceKind.CONFIRM,
        ResourceKind.LOCAL_FINE_LABELS,
        ResourceKind.SHARED_COARSE_GROUPS,
        ResourceKind.ANONYMOUS_SOURCE_PACKET,
        ResourceKind.LOCAL_MODEL_STATE,
        ResourceKind.LOCAL_IMPORTANCE_VECTOR,
    ):
        policy.assert_role_allowed(ClientRole.TARGET, resource)
    with pytest.raises(StrictResourceViolationError):
        policy.assert_role_allowed(ClientRole.TARGET, ResourceKind.TEST)
    policy.assert_role_allowed(ClientRole.TARGET, ResourceKind.TEST, transfer_finalized=True)


def test_non_transfer_roles_fail_closed() -> None:
    policy = StrictResourcePolicy()
    for role in (ClientRole.PRIMARY, ClientRole.SECONDARY):
        with pytest.raises(StrictResourceViolationError):
            policy.assert_role_allowed(role, ResourceKind.TRAIN)
