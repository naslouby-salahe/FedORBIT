from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from fedorbit.analysis.metrics import InvalidEvaluationDataError
from fedorbit.config.models import FedorbitConfig
from fedorbit.infrastructure.failures import FedorbitValidationError
from fedorbit.learning.scoring import (
    LocalClassCount,
    ScoringRequest,
    score_model,
)


class FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.table = nn.Parameter(logits, requires_grad=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[0] != self.table.shape[0]:
            raise AssertionError("feature batch size differs from the fixed logit table")
        return self.table


def _reference_class_entropies(
    logit_rows: tuple[tuple[float, ...], ...],
    target_rows: tuple[int, ...],
    class_count: int,
    log_floor: float,
) -> tuple[float, ...]:
    selected: list[list[float]] = [[] for _ in range(class_count)]
    for logits, target in zip(logit_rows, target_rows, strict=True):
        exponentials = tuple(math.exp(value) for value in logits)
        probability = exponentials[target] / sum(exponentials)
        selected[target].append(-math.log(max(probability, log_floor)))
    if any(not values for values in selected):
        raise AssertionError("reference construction requires every class to be present")
    return tuple(sum(values) / len(values) for values in selected)


def _scoring_request(
    logit_rows: tuple[tuple[float, ...], ...],
    target_rows: tuple[int, ...],
    class_count: int,
) -> ScoringRequest:
    logits = torch.tensor(logit_rows, dtype=torch.float32)
    targets = torch.tensor(target_rows, dtype=torch.long)
    features = torch.zeros((len(target_rows), logits.shape[1]), dtype=torch.float32)
    return ScoringRequest(
        model=FixedLogitModel(logits),
        features=features,
        targets=targets,
        local_class_count=LocalClassCount(class_count),
    )


def test_macro_cross_entropy_is_unweighted_over_the_fixed_evaluation_class_set(
    fedorbit_config: FedorbitConfig,
) -> None:
    log_floor = fedorbit_config.scientific.metrics.probability_log_floor
    logit_rows = (
        (2.0, -1.0, 0.5),
        (2.0, -1.0, 0.5),
        (2.0, -1.0, 0.5),
        (-2.0, 3.0, 0.0),
        (0.0, 0.0, 4.0),
        (0.0, 0.0, -4.0),
    )
    target_rows = (0, 0, 0, 1, 2, 2)
    expected = _reference_class_entropies(logit_rows, target_rows, 3, log_floor)
    artifact = score_model(_scoring_request(logit_rows, target_rows, 3))
    observed = tuple(entry.value for entry in artifact.class_conditional_cross_entropy.values)
    assert observed == pytest.approx(expected)
    assert artifact.macro_cross_entropy.value == pytest.approx(sum(expected) / 3)
    row_weighted = sum(
        -math.log(
            max(
                math.exp(logits[target]) / sum(math.exp(value) for value in logits),
                log_floor,
            )
        )
        for logits, target in zip(logit_rows, target_rows, strict=True)
    ) / len(target_rows)
    assert artifact.macro_cross_entropy.value != pytest.approx(row_weighted)


def test_zero_example_fixed_evaluation_class_invalidates_the_cell(
    fedorbit_config: FedorbitConfig,
) -> None:
    del fedorbit_config
    with pytest.raises(InvalidEvaluationDataError) as raised:
        score_model(
            _scoring_request(
                ((1.0, -1.0, 0.0), (1.0, -1.0, 0.0)),
                (0, 1),
                3,
            )
        )
    assert raised.value.class_index == 2
    assert isinstance(raised.value.reason, str)
    assert raised.value.reason
    assert isinstance(raised.value, FedorbitValidationError)


def test_scoring_rejects_a_split_without_examples_for_the_first_fixed_class(
    fedorbit_config: FedorbitConfig,
) -> None:
    del fedorbit_config
    with pytest.raises(InvalidEvaluationDataError) as raised:
        score_model(
            _scoring_request(
                ((0.0, 2.0), (0.0, 2.0)),
                (1, 1),
                2,
            )
        )
    assert raised.value.class_index == 0
