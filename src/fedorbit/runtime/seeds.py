from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from fedorbit.domain.enums import RngNamespace
from fedorbit.domain.serialization import StableJsonPayload, stable_json

SEED32_MODULUS = 2**32


class SeedDerivationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RandomSeed:
    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.value < SEED32_MODULUS:
            raise SeedDerivationError("random seed must be in the unsigned 32-bit range")


@dataclass(frozen=True, slots=True)
class SeedDerivationRequest:
    base_seed: RandomSeed
    namespace: RngNamespace
    stable_coordinates: StableJsonPayload


@dataclass(frozen=True, slots=True)
class DerivedSeed:
    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.value < SEED32_MODULUS:
            raise SeedDerivationError("derived seed must be in the unsigned 32-bit range")


def derive_seed32(request: SeedDerivationRequest) -> DerivedSeed:
    coordinates_text = stable_json(request.stable_coordinates)
    payload = f"FedORBIT|{request.base_seed.value}|{request.namespace.value}|{coordinates_text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return DerivedSeed(int(digest[:8], 16) % SEED32_MODULUS)


@dataclass(frozen=True, slots=True)
class SeedPlan:
    base_seed: RandomSeed
    coordinates_json: str
    streams: tuple[SeedStream, ...]

    def seed_for(self, namespace: RngNamespace) -> DerivedSeed:
        for stream in self.streams:
            if stream.namespace == namespace:
                return stream.seed
        raise SeedDerivationError(f"namespace not in plan: {namespace}")


@dataclass(frozen=True, slots=True)
class SeedStream:
    namespace: RngNamespace
    seed: DerivedSeed


@dataclass(frozen=True, slots=True)
class SeedPlanRequest:
    base_seed: RandomSeed
    coordinates: StableJsonPayload


def seed_plan(request: SeedPlanRequest) -> SeedPlan:
    coordinates_json_value = stable_json(request.coordinates)
    return SeedPlan(
        base_seed=request.base_seed,
        coordinates_json=coordinates_json_value,
        streams=tuple(
            SeedStream(
                namespace,
                derive_seed32(
                    SeedDerivationRequest(request.base_seed, namespace, request.coordinates)
                ),
            )
            for namespace in RngNamespace
        ),
    )


@dataclass(frozen=True, slots=True)
class NumpyGeneratorRequest:
    seed: DerivedSeed


@dataclass(frozen=True, slots=True)
class NumpyGeneratorStream:
    generator: np.random.Generator


def numpy_generator(request: NumpyGeneratorRequest) -> NumpyGeneratorStream:
    return NumpyGeneratorStream(np.random.default_rng(request.seed.value))


@dataclass(frozen=True, slots=True)
class TorchGeneratorRequest:
    seed: DerivedSeed


@dataclass(frozen=True, slots=True)
class TorchGeneratorStream:
    generator: torch.Generator


def torch_generator(request: TorchGeneratorRequest) -> TorchGeneratorStream:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(request.seed.value)
    return TorchGeneratorStream(generator)


@dataclass(frozen=True, slots=True)
class StatisticalBootstrapRequest:
    statistical_seed: RandomSeed
    contrast_coordinates: StableJsonPayload


def statistical_bootstrap_stream(request: StatisticalBootstrapRequest) -> NumpyGeneratorStream:
    stream_seed = derive_seed32(
        SeedDerivationRequest(
            request.statistical_seed,
            RngNamespace.STATISTICAL_BOOTSTRAP,
            request.contrast_coordinates,
        )
    )
    return numpy_generator(NumpyGeneratorRequest(stream_seed))
