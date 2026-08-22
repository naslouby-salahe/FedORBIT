from __future__ import annotations

import numpy as np
import pytest

from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.eligibility import transfer_eligibility
from fedorbit.datasets.feature_quality import (
    MISSING_TOKEN_VOCABULARY,
    categorical_vocabulary,
    evaluate_feature_quality,
    is_missing_token,
    numeric_zero_is_not_missing,
)
from fedorbit.datasets.splits import (
    assign_duplicate_groups_chronologically,
    duplicate_group_midpoint_fraction,
    interval_edges,
    split_for_duplicate_group,
)
from fedorbit.domain.enums import Split


def test_split_intervals_exact_from_config(fedorbit_config: FedorbitConfig) -> None:
    edges = interval_edges(fedorbit_config)
    expected = {
        Split.TRAIN: (0.0, 0.55),
        Split.META: (0.55, 0.70),
        Split.VALID: (0.70, 0.80),
        Split.CONFIRM: (0.80, 0.90),
        Split.TEST: (0.90, 1.0),
    }
    assert {split: (lower, upper) for split, lower, upper in edges} == expected


def test_split_edges_are_contiguous_and_ordered(fedorbit_config: FedorbitConfig) -> None:
    edges = interval_edges(fedorbit_config)
    previous_upper = 0.0
    for _, lower, upper in edges:
        assert lower == pytest.approx(previous_upper)
        assert upper > lower
        previous_upper = upper
    assert previous_upper == pytest.approx(1.0)


def test_split_for_duplicate_group_midpoint(fedorbit_config: FedorbitConfig) -> None:
    assert split_for_duplicate_group(fedorbit_config, 0.1) == Split.TRAIN
    assert split_for_duplicate_group(fedorbit_config, 0.60) == Split.META
    assert split_for_duplicate_group(fedorbit_config, 0.75) == Split.VALID
    assert split_for_duplicate_group(fedorbit_config, 0.85) == Split.CONFIRM
    assert split_for_duplicate_group(fedorbit_config, 0.95) == Split.TEST


def test_split_rejects_out_of_range_midpoint(fedorbit_config: FedorbitConfig) -> None:
    from fedorbit.datasets.splits import SplitError

    with pytest.raises(SplitError):
        split_for_duplicate_group(fedorbit_config, -0.1)
    with pytest.raises(SplitError):
        split_for_duplicate_group(fedorbit_config, 1.5)


def test_midpoint_fraction_is_average() -> None:
    assert duplicate_group_midpoint_fraction(0.2, 0.4) == pytest.approx(0.3)


def test_duplicate_groups_assigned_chronologically(fedorbit_config: FedorbitConfig) -> None:
    assignments = assign_duplicate_groups_chronologically(
        fedorbit_config,
        (("g1", 0.1), ("g2", 0.9), ("g3", 0.75)),
    )
    assert assignments["g1"] == Split.TRAIN
    assert assignments["g2"] == Split.TEST
    assert assignments["g3"] == Split.VALID


def test_source_eligibility_requires_train_and_meta(fedorbit_config: FedorbitConfig) -> None:
    threshold = fedorbit_config.scientific.preprocessing.feature_missing_or_nonfinite_drop_threshold
    eligible = transfer_eligibility(
        fedorbit_config,
        source_train_support=threshold + 0.1,
        source_meta_support=threshold + 0.1,
        target_meta_support=threshold + 0.1,
        target_confirm_support=threshold + 0.1,
        target_test_support=threshold + 0.1,
    )
    assert eligible.source_eligible
    assert eligible.target_eligible
    partial = transfer_eligibility(
        fedorbit_config,
        source_train_support=threshold - 0.1,
        source_meta_support=threshold + 0.1,
        target_meta_support=threshold + 0.1,
        target_confirm_support=threshold + 0.1,
        target_test_support=threshold + 0.1,
    )
    assert not partial.source_eligible
    assert not partial.source_train_support_passes
    assert not partial.source_eligible


def test_target_eligibility_requires_meta_confirm_test(fedorbit_config: FedorbitConfig) -> None:
    threshold = fedorbit_config.scientific.preprocessing.feature_missing_or_nonfinite_drop_threshold
    partial = transfer_eligibility(
        fedorbit_config,
        source_train_support=threshold + 0.1,
        source_meta_support=threshold + 0.1,
        target_meta_support=threshold + 0.1,
        target_confirm_support=threshold - 0.1,
        target_test_support=threshold + 0.1,
    )
    assert not partial.target_eligible
    assert not partial.target_confirm_support_passes


def test_missing_token_vocabulary_exact() -> None:
    assert frozenset({"", "0", "0.0", "nan", "none", "null"}) == MISSING_TOKEN_VOCABULARY


def test_is_missing_token_case_insensitive_and_categorical_scoped() -> None:
    assert is_missing_token("NAN", categorical=False)
    assert is_missing_token("None", categorical=False)
    assert is_missing_token("", categorical=False)
    assert is_missing_token("0", categorical=True)
    assert is_missing_token("0.0", categorical=True)
    assert not is_missing_token("0", categorical=False)
    assert not is_missing_token("0.0", categorical=False)


def test_numeric_zero_is_not_missing() -> None:
    assert numeric_zero_is_not_missing(0.0)
    assert not numeric_zero_is_not_missing(float("nan"))
    assert not numeric_zero_is_not_missing(0.5)


def test_feature_quality_drops_above_threshold(fedorbit_config: FedorbitConfig) -> None:
    report = evaluate_feature_quality(
        fedorbit_config,
        feature_names=("good", "bad", "finite", "extra1", "extra2", "extra3"),
        categorical_features=frozenset(),
        train_values={
            "good": np.array([1.0, 2.0, 3.0]),
            "bad": np.array([np.nan, np.nan, 1.0]),
            "finite": np.array([1.0, 1.0, 1.0]),
            "extra1": np.array([1.0, 1.0, 1.0]),
            "extra2": np.array([1.0, 1.0, 1.0]),
            "extra3": np.array([1.0, 1.0, 1.0]),
        },
    )
    good = next(f for f in report.candidate_features if f.name == "good")
    bad = next(f for f in report.candidate_features if f.name == "bad")
    assert good.dropped is False
    assert bad.dropped is True
    assert report.dropped_feature_count == 1
    assert not report.client_invalid


def test_client_invalid_when_dropped_fraction_exceeds(fedorbit_config: FedorbitConfig) -> None:
    report = evaluate_feature_quality(
        fedorbit_config,
        feature_names=("f1", "f2", "f3"),
        categorical_features=frozenset(),
        train_values={
            "f1": np.array([np.nan, np.nan]),
            "f2": np.array([np.nan, np.nan]),
            "f3": np.array([1.0, 1.0]),
        },
    )
    assert report.client_invalid
    assert "dropped-feature fraction" in (report.client_invalid_reason or "")


def test_client_invalid_when_zero_candidate_features(
    fedorbit_config: FedorbitConfig,
) -> None:
    report = evaluate_feature_quality(
        fedorbit_config,
        feature_names=(),
        categorical_features=frozenset(),
        train_values={},
    )
    assert report.client_invalid
    assert report.candidate_count_before_filtering == 0


def test_missing_indicator_threshold(fedorbit_config: FedorbitConfig) -> None:
    report = evaluate_feature_quality(
        fedorbit_config,
        feature_names=("sparse", "dense"),
        categorical_features=frozenset(),
        train_values={
            "sparse": np.concatenate((np.array([np.nan, np.nan]), np.full(98, 1.0))),
            "dense": np.full(100, 1.0),
        },
    )
    sparse = next(f for f in report.candidate_features if f.name == "sparse")
    dense = next(f for f in report.candidate_features if f.name == "dense")
    assert sparse.missing_indicator
    assert not dense.missing_indicator


def test_categorical_vocabulary_ordered() -> None:
    vocabulary = categorical_vocabulary(("Zeta", "alpha", "Beta"))
    assert vocabulary[0] == "<ABSENT>"
    assert vocabulary[1] == "<RARE>"
    assert vocabulary[2] == "<UNK>"
    assert vocabulary[3:] == tuple(
        sorted(("Zeta", "alpha", "Beta"), key=lambda t: t.encode("utf-8"))
    )
