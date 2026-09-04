from __future__ import annotations

import pytest

from fedorbit.experiments.synthetic import (
    CouplingGenerationError,
    CouplingInstanceRequest,
    eligible_coupling_support_sizes,
    generate_coupling_instance,
)
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    build_padded_block_structure,
    compare_correspondences_lexicographically,
)
from fedorbit.types import CoarseGroup, CouplingCompatibility


def test_correspondence_order_is_deterministic_for_exact_comparisons() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 3},
        {CoarseGroup.DISRUPTION: 3},
    )
    first = BlockCorrespondence(blocks, (0, 1, 2))
    second = BlockCorrespondence(blocks, (0, 2, 1))
    assert compare_correspondences_lexicographically(first, second) < 0


def test_coupling_generator_produces_jointly_realizable_instances() -> None:
    request = CouplingInstanceRequest(
        compatibility=CouplingCompatibility.JOINTLY_REALIZABLE,
        response_heterogeneity=1.0,
        directed_asymmetry=0.0,
        response_sparsity=1.0,
        block_pattern=(2, 2),
        support_size=1,
        seed=1103,
        instance_index=0,
    )
    instance = generate_coupling_instance(request)
    assert instance.classification == CouplingCompatibility.JOINTLY_REALIZABLE


def test_coupling_generator_produces_incompatible_instances_with_multi_node_support() -> None:
    request = CouplingInstanceRequest(
        compatibility=CouplingCompatibility.INCOMPATIBLE,
        response_heterogeneity=2.0,
        directed_asymmetry=1.0,
        response_sparsity=1.0,
        block_pattern=(2, 2),
        support_size=2,
        seed=1103,
        instance_index=0,
    )
    instance = generate_coupling_instance(request)
    assert instance.classification == CouplingCompatibility.INCOMPATIBLE


def test_coupling_generator_is_seed_deterministic() -> None:
    request = CouplingInstanceRequest(
        compatibility=CouplingCompatibility.JOINTLY_REALIZABLE,
        response_heterogeneity=0.5,
        directed_asymmetry=0.5,
        response_sparsity=0.5,
        block_pattern=(2, 3),
        support_size=1,
        seed=2207,
        instance_index=1,
    )
    first = generate_coupling_instance(request)
    second = generate_coupling_instance(request)
    assert (first.lower_response_matrix == second.lower_response_matrix).all()
    assert (first.target_importance == second.target_importance).all()


def test_coupling_generator_raises_when_requested_class_is_unreachable() -> None:
    request = CouplingInstanceRequest(
        compatibility=CouplingCompatibility.INCOMPATIBLE,
        response_heterogeneity=1.0,
        directed_asymmetry=0.0,
        response_sparsity=1.0,
        block_pattern=(2, 2),
        support_size=1,
        seed=1103,
        instance_index=0,
    )
    with pytest.raises(CouplingGenerationError):
        generate_coupling_instance(request)


def test_single_active_support_node_is_always_jointly_realizable_never_incompatible() -> None:
    for seed in (1103, 2207, 3319, 4421, 5531):
        for block_pattern in ((2, 2), (2, 3), (3, 3)):
            for heterogeneity, asymmetry, sparsity in ((0.5, 0.0, 0.25), (2.0, 1.0, 1.0)):
                request = CouplingInstanceRequest(
                    compatibility=CouplingCompatibility.JOINTLY_REALIZABLE,
                    response_heterogeneity=heterogeneity,
                    directed_asymmetry=asymmetry,
                    response_sparsity=sparsity,
                    block_pattern=block_pattern,
                    support_size=1,
                    seed=seed,
                    instance_index=0,
                )
                instance = generate_coupling_instance(request)
                assert instance.classification == CouplingCompatibility.JOINTLY_REALIZABLE
                incompatible_request = CouplingInstanceRequest(
                    compatibility=CouplingCompatibility.INCOMPATIBLE,
                    response_heterogeneity=heterogeneity,
                    directed_asymmetry=asymmetry,
                    response_sparsity=sparsity,
                    block_pattern=block_pattern,
                    support_size=1,
                    seed=seed,
                    instance_index=0,
                )
                with pytest.raises(CouplingGenerationError):
                    generate_coupling_instance(incompatible_request)


def test_coupling_generator_constructs_joint_realizability_at_support_three() -> None:
    for seed in (2207, 3319, 4421, 6653, 7753, 8861, 9973):
        request = CouplingInstanceRequest(
            compatibility=CouplingCompatibility.JOINTLY_REALIZABLE,
            response_heterogeneity=0.5,
            directed_asymmetry=0.0,
            response_sparsity=1.0,
            block_pattern=(3, 3),
            support_size=3,
            seed=seed,
            instance_index=0,
        )
        instance = generate_coupling_instance(request)
        assert instance.classification == CouplingCompatibility.JOINTLY_REALIZABLE


def test_eligible_coupling_support_sizes_excludes_support_one_for_incompatible() -> None:
    registered = (1, 2, 3)
    assert eligible_coupling_support_sizes(
        CouplingCompatibility.JOINTLY_REALIZABLE, registered
    ) == (1, 2, 3)
    assert eligible_coupling_support_sizes(CouplingCompatibility.INCOMPATIBLE, registered) == (
        2,
        3,
    )
