from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from fedorbit.config.loading import active_config
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.validation import (
    PrimitiveFixtureKind,
    execute_primitive_validation,
)
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import ArtifactState, ExperimentName, OverwritePolicy

type FixturePayload = Mapping[str, JsonValue]
type FixtureCell = Mapping[str, JsonValue]


def _as_int(payload: Mapping[str, JsonValue], key: str) -> int:
    value = payload[key]
    assert isinstance(value, int)
    return value


def _as_sequence(payload: Mapping[str, JsonValue], key: str) -> tuple[JsonValue, ...]:
    value = payload[key]
    assert isinstance(value, list)
    return tuple(value)


def _cells(payload: FixturePayload) -> tuple[FixtureCell, ...]:
    cells = _as_sequence(payload, "cells")
    for cell in cells:
        assert isinstance(cell, dict)
    return tuple(cast(FixtureCell, cell) for cell in cells)


def _pattern(cell: FixtureCell) -> str:
    return "-".join(str(size) for size in cast(list[int], cell["block_pattern"]))


def _fixture(tmp_path: Path) -> FixturePayload:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    manifest = execute_primitive_validation(store, layout, OverwritePolicy.REPLACE)
    assert store.resolve(manifest.artifact_id).state == ArtifactState.COMPLETED
    return cast(
        FixturePayload, json.loads(Path(manifest.payload_paths[0]).read_text(encoding="utf-8"))
    )


def test_primitive_validation_persists_the_realized_fixture_cell_plan(tmp_path: Path) -> None:
    payload = _fixture(tmp_path)
    definition = build_catalogue().definition(ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION)
    block_patterns = active_config().generators.exact_separator_theorem.block_patterns
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    assert _as_int(payload, "registered_planned_cells") == definition.derived_planned_cells == 0
    assert _as_sequence(payload, "registered_seeds") == tuple(definition.seeds)
    assert _as_int(payload, "realized_cell_count") == len(_cells(payload))
    assert _as_int(payload, "realized_cell_count") == len(block_patterns) * (
        1 + len(confirmatory_seeds)
    )
    assert _as_int(payload, "hand_fixture_seed") == (
        active_config().experiments.mathematical_primitive_validation.hand_fixture_seed
    )
    assert _as_sequence(payload, "property_check_seeds") == tuple(confirmatory_seeds)


def test_primitive_validation_uses_the_registered_seed_role(tmp_path: Path) -> None:
    payload = _fixture(tmp_path)
    hand_fixture_seed = _as_int(payload, "hand_fixture_seed")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    cells = _cells(payload)
    hand_cells = tuple(
        cell for cell in cells if cell["cell_kind"] == PrimitiveFixtureKind.HAND_FIXTURE.value
    )
    property_cells = tuple(
        cell for cell in cells if cell["cell_kind"] == PrimitiveFixtureKind.PROPERTY_CHECK.value
    )
    block_patterns = tuple(
        "-".join(str(size) for size in pattern)
        for pattern in active_config().generators.exact_separator_theorem.block_patterns
    )
    assert len(hand_cells) == len(block_patterns)
    assert {cast(int, cell["seed"]) for cell in hand_cells} == {hand_fixture_seed}
    assert {_pattern(cell) for cell in hand_cells} == set(block_patterns)
    assert len(property_cells) == len(block_patterns) * len(confirmatory_seeds)
    observed = {(cast(int, cell["seed"]), _pattern(cell)) for cell in property_cells}
    assert observed == {
        (seed, pattern) for pattern in block_patterns for seed in confirmatory_seeds
    }
    assert hand_fixture_seed not in confirmatory_seeds


def test_primitive_validation_cells_carry_a_typed_state(tmp_path: Path) -> None:
    payload = _fixture(tmp_path)
    for cell in _cells(payload):
        assert cell["state"] == ArtifactState.COMPLETED.value
        assert cell["invalid_reason"] is None
        assert cell["cell_kind"] in {kind.value for kind in PrimitiveFixtureKind}
        assert cell["fixture_error_within_tolerance"] is True
        assert cast(int, cell["orbit_size"]) >= 1
        assert cast(int, cell["active_image_map_count"]) >= 1
        assert cell["null_padding_present"] is True


def test_primitive_validation_reuses_identical_fixture_evidence(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    first = execute_primitive_validation(store, layout, OverwritePolicy.REPLACE)
    second = execute_primitive_validation(store, layout, OverwritePolicy.REUSE)
    assert first.artifact_id == second.artifact_id
    assert len(store.all_manifests()) == 1
