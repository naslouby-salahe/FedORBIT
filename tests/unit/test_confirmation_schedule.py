from __future__ import annotations

import itertools
from typing import cast

import torch

from fedorbit.methods.confirmation import confirmation_schedule
from fedorbit.types import ContrastCoordinates, RandomSeed


def test_confirmation_schedule_is_seeded_and_retains_partial_batch() -> None:
    coordinates = cast(ContrastCoordinates, "pair-a-to-b:fedorbit:s=2")
    first = tuple(
        itertools.islice(
            confirmation_schedule(10, 4, cast(RandomSeed, 1103), coordinates, 0),
            4,
        )
    )
    repeated = tuple(
        itertools.islice(
            confirmation_schedule(10, 4, cast(RandomSeed, 1103), coordinates, 0),
            4,
        )
    )
    next_replicate = tuple(
        itertools.islice(
            confirmation_schedule(10, 4, cast(RandomSeed, 1103), coordinates, 1),
            3,
        )
    )

    assert [batch.shape[0] for batch in first] == [4, 4, 2, 4]
    assert all(torch.equal(left, right) for left, right in zip(first, repeated, strict=True))
    observed_indices = cast(list[int], torch.cat(first[:3]).tolist())
    assert sorted(observed_indices) == list(range(10))
    assert not torch.equal(first[0], next_replicate[0])
