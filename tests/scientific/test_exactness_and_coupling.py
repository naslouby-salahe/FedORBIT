from __future__ import annotations

from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    build_padded_block_structure,
    compare_correspondences_lexicographically,
)
from fedorbit.types import CoarseGroup


def test_correspondence_order_is_deterministic_for_exact_comparisons() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 3},
        {CoarseGroup.DISRUPTION: 3},
    )
    first = BlockCorrespondence(blocks, (0, 1, 2))
    second = BlockCorrespondence(blocks, (0, 2, 1))
    assert compare_correspondences_lexicographically(first, second) < 0
