from __future__ import annotations

from dataclasses import dataclass

import torch

from fedorbit.domain.enums import ClientRole, CoarseGroup, RngNamespace
from fedorbit.runtime.seeds import derive_seed32


class AnonymityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AnonymousNodeOrder:
    permutation: tuple[int, ...]
    display_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = tuple(range(len(self.permutation)))
        if tuple(sorted(self.permutation)) != expected:
            raise AnonymityError("anonymous node permutation is not a bijection")
        expected_ids = tuple(f"node-{index:04d}" for index in range(1, len(self.permutation) + 1))
        if self.display_ids != expected_ids:
            raise AnonymityError("anonymous node identifiers are not canonical sequential IDs")

    def reorder[T](self, values: tuple[T, ...]) -> tuple[T, ...]:
        if len(values) != len(self.permutation):
            raise AnonymityError("value count does not match anonymous node count")
        return tuple(values[index] for index in self.permutation)


def anonymous_node_order(
    node_count: int,
    base_seed: int,
    endpoint: ClientRole,
    coarse_group: CoarseGroup,
    coordinate: object,
) -> AnonymousNodeOrder:
    if node_count <= 0:
        raise AnonymityError("anonymous node order requires at least one node")
    if endpoint not in (ClientRole.SOURCE, ClientRole.TARGET):
        raise AnonymityError("anonymous ordering endpoint must be source or target")
    seed = derive_seed32(
        base_seed,
        RngNamespace.ANONYMOUS_NODE_ORDER,
        {
            "endpoint": endpoint.value,
            "coarse_group": coarse_group.value,
            "coordinate": coordinate,
        },
    )
    generator = torch.Generator().manual_seed(seed)
    permutation = tuple(int(index) for index in torch.randperm(node_count, generator=generator))
    display_ids = tuple(f"node-{index:04d}" for index in range(1, node_count + 1))
    return AnonymousNodeOrder(permutation, display_ids)
