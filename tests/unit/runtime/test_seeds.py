from __future__ import annotations

import hashlib

import pytest
import torch

from fedorbit.infrastructure.runtime import (
    SEED32_MODULUS,
    DerivedSeed,
    NumpyGeneratorRequest,
    RandomSeed,
    SeedDerivationError,
    SeedDerivationRequest,
    SeedPlan,
    SeedPlanRequest,
    SeedStream,
    StatisticalBootstrapRequest,
    TorchGeneratorRequest,
    derive_seed32,
    numpy_generator,
    seed_plan,
    statistical_bootstrap_stream,
    torch_generator,
)
from fedorbit.types import RngNamespace, StableSerializationError, stable_json

FIXED_COORDINATES = {
    "experiment": "Primary Strict Cross-Telemetry Transfer",
    "pair": ["edge_iiotset_network", "ton_iot_network"],
    "support": 2,
    "threshold": 0.005,
}


def _independent_expected_seed(base_seed: int, namespace: str, coordinates: object) -> int:
    payload = f"FedORBIT|{base_seed}|{namespace}|{stable_json(coordinates)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % SEED32_MODULUS


def test_derivation_matches_spec_formula() -> None:
    for base_seed in (101, 1103, 300, 5531):
        for namespace in RngNamespace:
            actual = derive_seed32(
                SeedDerivationRequest(RandomSeed(base_seed), namespace, FIXED_COORDINATES)
            ).value
            expected = _independent_expected_seed(base_seed, namespace.value, FIXED_COORDINATES)
            assert actual == expected
            assert 0 <= actual < SEED32_MODULUS


def test_derivation_is_stable_across_calls() -> None:
    first = derive_seed32(
        SeedDerivationRequest(RandomSeed(1103), RngNamespace.SPLIT, FIXED_COORDINATES)
    )
    second = derive_seed32(
        SeedDerivationRequest(RandomSeed(1103), RngNamespace.SPLIT, FIXED_COORDINATES)
    )
    assert first == second


def test_distinct_namespaces_yield_distinct_seeds() -> None:
    seeds = {
        derive_seed32(SeedDerivationRequest(RandomSeed(1103), namespace, FIXED_COORDINATES))
        for namespace in RngNamespace
    }
    assert len(seeds) == len(RngNamespace)


def test_distinct_base_seeds_yield_distinct_seeds() -> None:
    seeds = {
        derive_seed32(
            SeedDerivationRequest(RandomSeed(base), RngNamespace.SPLIT, FIXED_COORDINATES)
        )
        for base in (101, 202, 303, 1103)
    }
    assert len(seeds) == 4


def test_distinct_coordinates_yield_distinct_seeds() -> None:
    first = derive_seed32(
        SeedDerivationRequest(RandomSeed(1103), RngNamespace.SPLIT, {"pair": "a"})
    )
    second = derive_seed32(
        SeedDerivationRequest(RandomSeed(1103), RngNamespace.SPLIT, {"pair": "b"})
    )
    assert first != second


def test_stable_json_sorts_keys_and_omits_whitespace() -> None:
    rendered = stable_json({"z": 1, "a": {"y": 2, "b": 3}})
    assert rendered == '{"a":{"b":3,"y":2},"z":1}'
    assert " " not in rendered


def test_stable_json_shortest_float_form() -> None:
    assert stable_json({"value": 0.1}) == '{"value":0.1}'
    assert stable_json({"value": 0.005}) == '{"value":0.005}'


def test_stable_json_rejects_non_finite_floats() -> None:
    with pytest.raises(StableSerializationError):
        stable_json({"value": float("nan")})
    with pytest.raises(StableSerializationError):
        stable_json({"value": float("inf")})


def test_stable_json_encodes_as_utf8() -> None:
    rendered = stable_json({"experiment": "FedORBIT"})
    rendered.encode("utf-8")


def test_seed_plan_covers_every_namespace() -> None:
    plan = seed_plan(SeedPlanRequest(RandomSeed(1103), FIXED_COORDINATES))
    assert len(plan.streams) == len(RngNamespace)
    for namespace in RngNamespace:
        assert plan.seed_for(namespace) == derive_seed32(
            SeedDerivationRequest(RandomSeed(1103), namespace, FIXED_COORDINATES)
        )


def test_seed_plan_rejects_unknown_namespace() -> None:
    partial = SeedPlan(
        base_seed=RandomSeed(1103),
        coordinates_json="{}",
        streams=(SeedStream(RngNamespace.SPLIT, DerivedSeed(1)),),
    )
    with pytest.raises(SeedDerivationError):
        partial.seed_for(RngNamespace.DENSE_START)


def test_scoped_numpy_generator_replays_identical_choices() -> None:
    first = numpy_generator(NumpyGeneratorRequest(DerivedSeed(42))).generator.integers(
        0, 2**31, size=8
    )
    second = numpy_generator(NumpyGeneratorRequest(DerivedSeed(42))).generator.integers(
        0, 2**31, size=8
    )
    assert list(first) == list(second)
    different = numpy_generator(NumpyGeneratorRequest(DerivedSeed(43))).generator.integers(
        0, 2**31, size=8
    )
    assert list(first) != list(different)


def test_scoped_torch_generator_replays_identical_choices() -> None:
    first = torch.rand(
        8, generator=torch_generator(TorchGeneratorRequest(DerivedSeed(42))).generator
    )
    second = torch.rand(
        8, generator=torch_generator(TorchGeneratorRequest(DerivedSeed(42))).generator
    )
    assert torch.equal(first, second)
    different = torch.rand(
        8, generator=torch_generator(TorchGeneratorRequest(DerivedSeed(43))).generator
    )
    assert not torch.equal(first, different)


def test_statistical_bootstrap_streams_are_contrast_scoped() -> None:
    first = statistical_bootstrap_stream(
        StatisticalBootstrapRequest(RandomSeed(300), {"contrast": "A", "family": "utility"})
    ).generator
    second = statistical_bootstrap_stream(
        StatisticalBootstrapRequest(RandomSeed(300), {"contrast": "A", "family": "utility"})
    ).generator
    assert list(first.integers(0, 2**31, size=8)) == list(second.integers(0, 2**31, size=8))
    other = statistical_bootstrap_stream(
        StatisticalBootstrapRequest(RandomSeed(300), {"contrast": "B", "family": "utility"})
    ).generator
    assert list(first.integers(0, 2**31, size=8)) != list(other.integers(0, 2**31, size=8))


def test_statistical_seed_is_used_only_for_bootstrap_streams() -> None:
    plan = seed_plan(SeedPlanRequest(RandomSeed(300), FIXED_COORDINATES))
    bootstrap = plan.seed_for(RngNamespace.STATISTICAL_BOOTSTRAP)
    split = plan.seed_for(RngNamespace.SPLIT)
    assert bootstrap != split
