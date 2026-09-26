from __future__ import annotations

import numpy as np
import pytest

from fedorbit.detection.gaussian import (
    DegenerateCovarianceError,
    scorer_from_moments,
    scorer_from_rows,
    shrunk_precision,
)
from fedorbit.detection.moments import (
    aggregate_equal_weight,
    combine,
    pool_with_partners,
    summarise,
)


def test_combine_equals_stacked_rows() -> None:
    generator = np.random.default_rng(0)
    first = generator.normal(1.0, 2.0, size=(40, 5))
    second = generator.normal(-1.0, 0.5, size=(70, 5))
    combined = combine([(40.0, summarise(first)), (70.0, summarise(second))])
    stacked = summarise(np.vstack([first, second]))
    assert combined.weight == pytest.approx(110.0)
    assert np.allclose(combined.mean, stacked.mean)
    assert np.allclose(combined.covariance, stacked.covariance)


def test_equal_weight_aggregate_ignores_partner_row_counts() -> None:
    generator = np.random.default_rng(1)
    small = generator.normal(0.0, 1.0, size=(10, 3))
    large = generator.normal(4.0, 1.0, size=(1000, 3))
    aggregate = aggregate_equal_weight([summarise(small), summarise(large)])
    assert np.allclose(aggregate.mean, (small.mean(axis=0) + large.mean(axis=0)) / 2.0)


def test_pool_with_partners_uses_pseudo_row_weight() -> None:
    generator = np.random.default_rng(2)
    local = summarise(generator.normal(0.0, 1.0, size=(30, 4)))
    partners = summarise(generator.normal(3.0, 1.0, size=(500, 4)))
    pooled = pool_with_partners(local, partners, 90.0)
    assert pooled.weight == pytest.approx(120.0)
    assert np.allclose(pooled.mean, (30.0 * local.mean + 90.0 * partners.mean) / 120.0)


def test_shrunk_precision_matches_closed_form_on_two_dimensions() -> None:
    second_moment = np.array([[2.0, 0.5], [0.5, 1.0]])
    count = 10.0
    trace = 3.0
    trace_of_square = float((second_moment**2).sum())
    numerator = (1.0 - 2.0 / 2.0) * trace_of_square + trace**2
    denominator = (count + 1.0 - 2.0 / 2.0) * (trace_of_square - trace**2 / 2.0)
    shrinkage = min(1.0, numerator / denominator)
    expected = np.linalg.inv(
        (1.0 - shrinkage) * second_moment + shrinkage * np.eye(2) * (trace / 2.0)
    )
    assert np.allclose(shrunk_precision(second_moment, count), expected)


def test_isotropic_second_moment_shrinks_fully_to_target() -> None:
    precision = shrunk_precision(np.eye(3) * 2.0, 5.0)
    assert np.allclose(precision, np.eye(3) * 0.5)


def test_scorer_is_zero_at_centre_and_grows_with_distance() -> None:
    generator = np.random.default_rng(3)
    rows = generator.normal(0.0, 1.0, size=(200, 4))
    scorer = scorer_from_rows(rows, rows.mean(axis=0), rows.std(axis=0) + 1e-6)
    near = scorer.score(rows.mean(axis=0, keepdims=True))
    far = scorer.score(rows.mean(axis=0, keepdims=True) + 8.0)
    assert near[0] < 1e-9
    assert far[0] > 100.0


def test_scorer_from_moments_agrees_with_rows_when_moments_are_the_rows() -> None:
    generator = np.random.default_rng(4)
    rows = generator.normal(2.0, 3.0, size=(300, 5))
    from_rows = scorer_from_rows(rows, rows.mean(axis=0), rows.std(axis=0) + 1e-6)
    from_moments = scorer_from_moments(summarise(rows), 1e-6)
    probe = generator.normal(2.0, 3.0, size=(20, 5))
    assert np.allclose(from_rows.score(probe), from_moments.score(probe), rtol=1e-6)


def test_singular_shrunk_covariance_is_reported() -> None:
    with pytest.raises(DegenerateCovarianceError):
        shrunk_precision(np.zeros((3, 3)), 100.0)
