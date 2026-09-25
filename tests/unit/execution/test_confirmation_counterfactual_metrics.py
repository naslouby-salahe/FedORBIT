from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import fedorbit.experiments.transfer as transfer
from fedorbit.analysis.metrics import ClassEntropySet, CrossEntropy
from fedorbit.experiments.catalogue import ExperimentExecutionRequest, build_catalogue
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.learning.scoring import ScoreArtifact
from fedorbit.methods.assimilation import ConfirmationVerdict
from fedorbit.types import (
    DatasetId,
    ExperimentName,
    MetricId,
    OverwritePolicy,
    RandomSeed,
    directed_pair_name,
)


def _score(value: float) -> ScoreArtifact:
    entropy = CrossEntropy(value)
    return ScoreArtifact((), ClassEntropySet((entropy,)), entropy)


@pytest.mark.parametrize(
    ("counterfactual_ce", "expected_rate"),
    ((0.8, 1.0), (0.995, 0.0)),
)
def test_beneficial_rejected_rate_uses_no_confirmation_counterfactual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    counterfactual_ce: float,
    expected_rate: float,
) -> None:
    config = build_catalogue()
    request = ExperimentExecutionRequest(
        ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY,
        config.definition(ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY),
        OverwritePolicy.REUSE,
    )
    source, target = tuple(DatasetId)[:2]
    pair = directed_pair_name(source, target)
    captured: list[tuple[object, ...]] = []

    def capture(*args: object, **_kwargs: object) -> None:
        captured.append(args)

    monkeypatch.setattr(transfer, "persist_primary_transfer_metric", capture)
    persist_metrics = cast(
        Callable[..., None],
        getattr(transfer, "_persist_confirmation_safety_indicators"),  # noqa: B009
    )
    persist_metrics(
        ArtifactStore(tmp_path / "outputs"),
        build_layout(tmp_path),
        request,
        pair,
        source,
        target,
        cast(RandomSeed, 1103),
        CrossEntropy(1.0),
        _score(1.0),
        _score(counterfactual_ce),
        [ConfirmationVerdict(False, 0.0, 0.0)],
        (),
    )

    values = {
        cast(MetricId, call[8]): call[9]
        for call in captured
        if len(call) > 9 and call[8] == MetricId.BENEFICIAL_REJECTED_RATE
    }
    assert values == {MetricId.BENEFICIAL_REJECTED_RATE: expected_rate}
    assert values[MetricId.BENEFICIAL_REJECTED_RATE] is not None
