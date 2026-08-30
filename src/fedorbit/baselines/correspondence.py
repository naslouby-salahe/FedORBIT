from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import RngNamespace
from fedorbit.orbit.correspondence import BlockCorrespondence, PaddedBlockStructure
from fedorbit.orbit.objective import CurriculumAction, RobustActionProblem
from fedorbit.runtime.seeds import derive_seed32


class CouplingDestructionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CouplingDestroyedMatrices:
    lower_response_matrix: NDArray[np.float64]
    upper_response_matrix: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CommittedMapAction:
    correspondence: BlockCorrespondence
    selected_action: CurriculumAction


def _block_pair_permutation(
    entry_count: int,
    seed: int,
    coordinates: str,
    block_pair_index: int,
) -> NDArray[np.intp]:
    rng_seed = derive_seed32(
        seed,
        RngNamespace.COUPLING_DESTRUCTION,
        OrderedDict[str, str | int](coordinates=coordinates, block_pair=block_pair_index),
    )
    generator = np.random.default_rng(rng_seed)
    return generator.permutation(entry_count)


def coupling_destroyed_matrices(
    blocks: PaddedBlockStructure,
    lower_response_matrix: NDArray[np.float64],
    upper_response_matrix: NDArray[np.float64],
    seed: int,
    contrast_coordinates: str,
) -> CouplingDestroyedMatrices:
    size = blocks.total_padded_nodes
    if lower_response_matrix.shape != (size, size):
        raise CouplingDestructionError("lower matrix shape mismatch")
    if upper_response_matrix.shape != (size, size):
        raise CouplingDestructionError("upper matrix shape mismatch")
    destroyed_lower = np.array(lower_response_matrix, dtype=np.float64, copy=True)
    destroyed_upper = np.array(upper_response_matrix, dtype=np.float64, copy=True)
    block_pair_index = 0
    for group_source in range(len(blocks.padded_size_tuple)):
        rows = blocks.block_index_range(group_source)
        for group_target in range(len(blocks.padded_size_tuple)):
            columns = blocks.block_index_range(group_target)
            lower_block = lower_response_matrix[np.ix_(rows, columns)]
            upper_block = upper_response_matrix[np.ix_(rows, columns)]
            flat_entries = lower_block.reshape(-1)
            permutation = _block_pair_permutation(
                flat_entries.shape[0],
                seed,
                contrast_coordinates,
                block_pair_index,
            )
            permuted_lower = flat_entries[permutation].reshape(lower_block.shape)
            permuted_upper = upper_block.reshape(-1)[permutation].reshape(upper_block.shape)
            destroyed_lower[np.ix_(rows, columns)] = permuted_lower
            destroyed_upper[np.ix_(rows, columns)] = permuted_upper
            block_pair_index += 1
    return CouplingDestroyedMatrices(destroyed_lower, destroyed_upper)


def committed_map_action(
    problem: RobustActionProblem,
    source_matrix: NDArray[np.float64],
    target_matrix: NDArray[np.float64],
    config: FedorbitConfig,
) -> CommittedMapAction:
    from fedorbit.baselines.local import optimize_against_fixed_matrix
    from fedorbit.solvers.exact_qap import point_correspondence_commitment

    result = point_correspondence_commitment(source_matrix, target_matrix, problem.blocks, config)
    correspondence = result.require_certified().correspondence
    committed = correspondence.permute_response_matrix(problem.lower_response_matrix)
    solution = optimize_against_fixed_matrix(problem, committed)
    return CommittedMapAction(correspondence, solution.selected_action)
