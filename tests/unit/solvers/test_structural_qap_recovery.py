from __future__ import annotations

import numpy as np

from fedorbit.experiments.audit import RECOVERY_METHOD_ATTEMPTS
from fedorbit.experiments.synthetic import (
    ExactSeparatorInstanceRequest,
    generate_exact_separator_instance,
)
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    PaddedBlockStructure,
    build_padded_block_structure,
)
from fedorbit.optimization.exact_qap import (
    generic_exact_qap_correspondence,
    point_correspondence_commitment,
)
from fedorbit.types import CoarseGroup, TransferMethod


def _blocks() -> PaddedBlockStructure:
    groups = tuple(CoarseGroup)[:2]
    counts = {groups[0]: 2, groups[1]: 2}
    return build_padded_block_structure(groups, counts, counts)


def test_generic_exact_qap_correspondence_agrees_with_the_point_commitment() -> None:
    blocks = _blocks()
    source = np.asarray(
        (
            (-0.20, 0.05, 0.01, 0.00),
            (0.03, -0.18, 0.02, 0.04),
            (0.01, 0.02, -0.16, 0.06),
            (0.00, 0.05, 0.04, -0.14),
        ),
        dtype=np.float64,
    )
    assert source.shape == (blocks.total_padded_nodes, blocks.total_padded_nodes)
    commitment = point_correspondence_commitment(source, source, blocks)
    generic = generic_exact_qap_correspondence(source, source, blocks)
    assert commitment.certified
    assert generic.certified
    assert commitment.correspondence is not None
    assert generic.correspondence is not None
    assert generic.terminal_state is None
    assert commitment.correspondence.images == generic.correspondence.images
    assert generic.correspondence.blocks == blocks


def test_generic_exact_qap_recovery_is_a_distinct_registered_implementation() -> None:
    assert (
        RECOVERY_METHOD_ATTEMPTS[TransferMethod.POINT_CORRESPONDENCE_COMMITMENT]
        is not RECOVERY_METHOD_ATTEMPTS[TransferMethod.GENERIC_EXACT_QAP]
    )
    assert set(RECOVERY_METHOD_ATTEMPTS) == {
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
        TransferMethod.GENERIC_EXACT_QAP,
    }


def test_structural_qap_recovery_finds_the_oracle_map_on_a_diagonal_matrix() -> None:
    blocks = _blocks()
    source = np.asarray(
        (
            (-0.20, 0.00, 0.01, 0.02),
            (0.00, -0.15, 0.03, 0.01),
            (0.01, 0.03, -0.12, 0.00),
            (0.02, 0.01, 0.00, -0.10),
        ),
        dtype=np.float64,
    )
    oracle = BlockCorrespondence.lexicographically_smallest(blocks)
    for recovery in (point_correspondence_commitment, generic_exact_qap_correspondence):
        result = recovery(source, source, blocks)
        assert result.certified
        assert result.correspondence is not None
        assert result.correspondence.images == oracle.images


def test_generic_exact_qap_recovery_is_deterministic_for_a_registered_seed() -> None:
    blocks = _blocks()
    instance = generate_exact_separator_instance(ExactSeparatorInstanceRequest((2, 2), 1103))
    first = generic_exact_qap_correspondence(
        instance.lower_response_matrix, instance.lower_response_matrix, blocks
    )
    second = generic_exact_qap_correspondence(
        instance.lower_response_matrix, instance.lower_response_matrix, blocks
    )
    assert first.certified and second.certified
    assert first.correspondence is not None and second.correspondence is not None
    assert first.correspondence.images == second.correspondence.images
