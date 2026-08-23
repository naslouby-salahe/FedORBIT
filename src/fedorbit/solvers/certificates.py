from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fedorbit.orbit.correspondence import BlockCorrespondence, PaddedBlockStructure
from fedorbit.orbit.objective import CurriculumAction


class CertificateError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SeparatorWorkCertificate:
    active_image_candidates: int
    lap_calls: int

    def verify_against(
        self,
        blocks: PaddedBlockStructure,
        support_block_counts: tuple[int, ...],
    ) -> bool:
        from fedorbit.orbit.correspondence import (
            BlockNodeCounts,
            active_image_assignment_count,
        )

        expected_candidates = active_image_assignment_count(
            blocks, BlockNodeCounts(blocks=blocks, per_block=support_block_counts)
        )
        expected_lap_calls = expected_candidates * sum(
            1
            for block_index, size in enumerate(blocks.padded_size_tuple)
            if size - support_block_counts[block_index] > 0
        )
        return (
            self.active_image_candidates == expected_candidates
            and self.lap_calls == expected_lap_calls
        )


def verify_correspondence_certificate(
    correspondence: BlockCorrespondence,
    claimed_objective: float,
    action: CurriculumAction,
    objective_tolerance: float,
) -> bool:
    from fedorbit.orbit.objective import evaluate_objective

    recomputed = evaluate_objective(action, correspondence)
    return abs(recomputed - claimed_objective) <= objective_tolerance


def verify_exactness_certificate(
    solver_value: float,
    exhaustive_truth_value: float,
    exact_tolerance: float,
) -> bool:
    return abs(solver_value - exhaustive_truth_value) <= exact_tolerance


def require_valid_images(images: Sequence[int], blocks: PaddedBlockStructure) -> None:
    total = blocks.total_padded_nodes
    if sorted(images) != list(range(total)):
        raise CertificateError("certificate images are not a padded-space bijection")


def certificate_residual(claimed: float, recomputed: float) -> float:
    return float(np.abs(claimed - recomputed))
