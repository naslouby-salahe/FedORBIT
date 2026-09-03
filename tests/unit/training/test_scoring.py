from __future__ import annotations

import torch

from fedorbit.learning.scoring import LocalClassCount, ScoringRequest, score_model


def test_score_model_builds_typed_score_artifact() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(((1.0, 0.0), (0.0, 1.0))))
    artifact = score_model(
        ScoringRequest(
            model=model,
            features=torch.tensor(((1.0, 0.0), (0.0, 1.0))),
            targets=torch.tensor((0, 1)),
            local_class_count=LocalClassCount(2),
        )
    )
    assert len(artifact.rows) == 2
    assert artifact.rows[0].target.value == 0
    assert artifact.macro_cross_entropy.value >= 0.0
