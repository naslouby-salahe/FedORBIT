from __future__ import annotations

from fedorbit.experiments.catalogue import build_catalogue


def test_registered_experiments_are_deterministic_and_unique() -> None:
    catalogue = build_catalogue()
    assert catalogue == build_catalogue()
    assert len(set(catalogue.registered_names())) == len(catalogue)
