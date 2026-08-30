from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SyntheticRandomRequest:
    seed: int


@dataclass(frozen=True, slots=True)
class SyntheticRandomStream:
    generator: np.random.Generator


def create_float64_random_stream(request: SyntheticRandomRequest) -> SyntheticRandomStream:
    return SyntheticRandomStream(np.random.Generator(np.random.PCG64(request.seed)))
