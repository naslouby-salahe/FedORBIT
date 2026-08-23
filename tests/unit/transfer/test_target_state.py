from __future__ import annotations

import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.transfer.target_state import (
    TargetImportanceError,
    TransferNodeRisk,
    build_target_importance,
)


def _risks() -> tuple[TransferNodeRisk, ...]:
    return (
        TransferNodeRisk(node_index=0, is_actionable=True, meta_class_risk=0.20),
        TransferNodeRisk(node_index=1, is_actionable=True, meta_class_risk=0.08),
        TransferNodeRisk(node_index=2, is_actionable=True, meta_class_risk=1.0e-9),
        TransferNodeRisk(node_index=3, is_actionable=False, meta_class_risk=0.55),
        TransferNodeRisk(node_index=4, is_actionable=False, meta_class_risk=0.0),
    )


@pytest.fixture
def config() -> FedorbitConfig:
    return load_fedorbit_config()


def test_importance_uses_meta_risk_only_with_floor_and_normalization(
    config: FedorbitConfig,
) -> None:
    floor = config.scientific.target_importance.class_risk_floor
    assert floor == 1e-4
    importance = build_target_importance(config, _risks())
    floored_total = 0.20 + 0.08 + floor
    assert importance.weight_of(0) == pytest.approx(0.20 / floored_total)
    assert importance.weight_of(1) == pytest.approx(0.08 / floored_total)
    assert importance.weight_of(2) == pytest.approx(floor / floored_total)
    assert importance.actionable_total == pytest.approx(1.0)


def test_normal_and_null_nodes_have_zero_importance(config: FedorbitConfig) -> None:
    importance = build_target_importance(config, _risks())
    assert importance.weight_of(3) == 0.0
    assert importance.weight_of(4) == 0.0


def test_vector_expansion_places_zero_coordinates(config: FedorbitConfig) -> None:

    importance = build_target_importance(config, _risks())
    vector = importance.as_vector(6)
    assert vector.shape == (6,)
    assert vector[5] == 0.0
    assert vector[0] > 0.0
    with pytest.raises(TargetImportanceError):
        importance.as_vector(2)


def test_duplicate_node_reports_rejected(config: FedorbitConfig) -> None:
    with pytest.raises(TargetImportanceError):
        build_target_importance(
            config,
            (*_risks(), TransferNodeRisk(0, True, 0.1)),
        )


def test_negative_risk_report_rejected() -> None:
    with pytest.raises(TargetImportanceError):
        build_target_importance(
            load_fedorbit_config(),
            (TransferNodeRisk(0, True, -0.1),),
        )


def test_nonfinite_risk_report_rejected_at_construction() -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(TargetImportanceError):
            TransferNodeRisk(0, True, bad)


def test_negative_node_index_rejected_at_construction() -> None:
    with pytest.raises(TargetImportanceError):
        TransferNodeRisk(-1, False, 0.1)


def test_no_actionable_nodes_is_insufficient_evidence(config: FedorbitConfig) -> None:
    with pytest.raises(TargetImportanceError):
        build_target_importance(
            config,
            (
                TransferNodeRisk(0, is_actionable=False, meta_class_risk=0.4),
                TransferNodeRisk(1, is_actionable=False, meta_class_risk=0.1),
            ),
        )


def test_construction_is_deterministic_and_sorted(config: FedorbitConfig) -> None:
    first = build_target_importance(config, tuple(reversed(_risks())))
    second = build_target_importance(config, _risks())
    assert first == second
    weights = [first.weight_of(node) for node in sorted(first.weights_by_node_index)]
    assert all(weight >= 0.0 for weight in weights)


def test_importance_feeds_robust_action_problem(config: FedorbitConfig) -> None:
    import numpy as np

    from fedorbit.domain.enums import CoarseGroup
    from fedorbit.orbit.correspondence import build_padded_block_structure
    from fedorbit.orbit.objective import RobustActionProblem, build_robust_action_problem

    importance = build_target_importance(config, _risks())
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 5},
        {CoarseGroup.DISRUPTION: 5},
    )
    size = blocks.total_padded_nodes
    problem = build_robust_action_problem(
        config,
        blocks,
        np.zeros((size, size)),
        np.zeros((size, size)),
        importance.as_vector(size),
        actionable_nodes=(0, 1, 2),
    )
    assert isinstance(problem, RobustActionProblem)
    assert np.all(problem.target_importance >= 0.0)
    assert problem.coordinate_caps[3] == 0.0
