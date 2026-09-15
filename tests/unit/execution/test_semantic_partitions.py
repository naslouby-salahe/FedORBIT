from __future__ import annotations

from typing import cast

from fedorbit.experiments.transfer import semantic_partition_bucket_of
from fedorbit.types import CoarseGroup, SemanticPartitionId, SemanticPartitionSpecification


def test_principal_partition_is_identity_on_coarse_groups() -> None:
    mapping = semantic_partition_bucket_of(SemanticPartitionId.PRINCIPAL_THREE_COARSE_GROUPS)
    assert mapping is not None
    assert mapping[CoarseGroup.DISRUPTION] == CoarseGroup.DISRUPTION
    assert mapping[CoarseGroup.EXPLOITATION] == CoarseGroup.EXPLOITATION


def test_oracle_fine_singleton_is_not_aliased_to_principal_partition() -> None:
    assert semantic_partition_bucket_of(SemanticPartitionId.ORACLE_FINE_SINGLETON_GROUPS) is None


def test_unknown_partition_is_rejected() -> None:
    partition = cast(SemanticPartitionSpecification, "invented-partition")
    assert semantic_partition_bucket_of(partition) is None
