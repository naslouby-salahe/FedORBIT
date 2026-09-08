from __future__ import annotations

from fedorbit.experiments.catalogue import build_catalogue


def test_execution_plan_exposes_complete_registered_catalogue() -> None:
    catalogue = build_catalogue()
    names = catalogue.registered_names()
    assert len(names) == 26
    assert len(set(names)) == 26
    assert all(catalogue.definition(name).classification.value for name in names)
