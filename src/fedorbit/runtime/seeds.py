from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from fedorbit.domain.enums import RngNamespace
from fedorbit.domain.serialization import stable_json

SEED32_MODULUS = 2**32


class SeedDerivationError(ValueError):
    pass


def derive_seed32(base_seed: int, namespace: RngNamespace, stable_coordinates: object) -> int:
    coordinates_text = stable_json(stable_coordinates)
    payload = f"FedORBIT|{base_seed}|{namespace.value}|{coordinates_text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % SEED32_MODULUS


@dataclass(frozen=True, slots=True)
class SeedPlan:
    base_seed: int
    coordinates_json: str
    streams: tuple[tuple[RngNamespace, int], ...]

    def seed_for(self, namespace: RngNamespace) -> int:
        for registered, seed in self.streams:
            if registered == namespace:
                return seed
        raise SeedDerivationError(f"namespace not in plan: {namespace}")


def seed_plan(base_seed: int, coordinates: object) -> SeedPlan:
    coordinates_json_value = stable_json(coordinates)
    return SeedPlan(
        base_seed=base_seed,
        coordinates_json=coordinates_json_value,
        streams=tuple(
            (namespace, derive_seed32(base_seed, namespace, coordinates))
            for namespace in RngNamespace
        ),
    )


def numpy_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def statistical_bootstrap_stream(
    statistical_seed: int, contrast_coordinates: object
) -> np.random.Generator:
    stream_seed = derive_seed32(
        statistical_seed, RngNamespace.STATISTICAL_BOOTSTRAP, contrast_coordinates
    )
    return numpy_generator(stream_seed)
