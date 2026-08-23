from __future__ import annotations

from dataclasses import dataclass

from fedorbit.config.models import FedorbitConfig


@dataclass(frozen=True, slots=True)
class PairContrastEvidence:
    directed_pair: str
    mean_gain: float | None
    holm_p: float | None
    bca_lower: float | None
    strict_resource_valid: bool
    valid_seed_count: int


@dataclass(frozen=True, slots=True)
class TransferCriteriaDecision:
    supported: bool
    conditional: bool
    partially_supported: bool
    null_result: bool
    not_supported: bool
    successful_pairs: tuple[str, ...]
    harmful_pairs: tuple[str, ...]
    equal_pair_mean_gain: float | None
    reasons: tuple[str, ...]


def _successful_pair(
    config: FedorbitConfig,
    evidence: PairContrastEvidence,
) -> bool:
    criteria = config.scientific.claim_criteria.strict_cross_telemetry_utility
    materiality = config.scientific.materiality.realized_relative_macro_ce
    if evidence.mean_gain is None or evidence.holm_p is None or evidence.bca_lower is None:
        return False
    assert evidence.mean_gain is not None
    return (
        evidence.mean_gain >= materiality
        and evidence.holm_p < criteria.holm_adjusted_p_maximum
        and evidence.bca_lower > criteria.bca_lower_bound_strictly_greater_than
        and evidence.strict_resource_valid
    )


def evaluate_transfer_style_criteria(
    config: FedorbitConfig,
    evidence_by_pair: dict[str, PairContrastEvidence],
    removed_before_outcome_inspection: bool,
) -> TransferCriteriaDecision:
    claim = config.scientific.claim_criteria.strict_cross_telemetry_utility
    materiality = config.scientific.materiality.realized_relative_macro_ce
    harm_threshold = config.scientific.materiality.harmful_transfer_relative_macro_ce_gain
    required = claim.successful_primary_pairs_required

    all_pairs = sorted(evidence_by_pair)
    analyzable = {
        pair: evidence
        for pair, evidence in evidence_by_pair.items()
        if evidence.valid_seed_count > 0
    }
    reasons: list[str] = []

    successful = tuple(pair for pair in analyzable if _successful_pair(config, analyzable[pair]))
    harmful = tuple(
        pair
        for pair in analyzable
        if analyzable[pair].mean_gain is not None and analyzable[pair].mean_gain <= harm_threshold
    )

    scope_removed = len(all_pairs) - len(analyzable)
    if scope_removed > 1:
        reasons.append("more than one pair removed from claim scope")
        return TransferCriteriaDecision(
            False, False, False, False, True, (), (), None, tuple(reasons)
        )

    reduced_scope = removed_before_outcome_inspection and len(analyzable) == 3
    full_scope = not removed_before_outcome_inspection and len(analyzable) == 4

    equal_pair_values = tuple(analyzable[pair].mean_gain for pair in sorted(analyzable))
    all_present = all(value is not None for value in equal_pair_values)
    if all_present:
        numeric_values = [value for value in equal_pair_values if value is not None]
        equal_pair_mean_value: float | None = sum(numeric_values) / len(numeric_values)
    else:
        equal_pair_mean_value = None

    if any(not e.strict_resource_valid for e in evidence_by_pair.values()):
        reasons.append("strict-resource validation failed for a contributing run")
        return TransferCriteriaDecision(
            False, False, False, False, True, (), (), None, tuple(reasons)
        )

    if full_scope and len(successful) >= required and not harmful:
        if equal_pair_mean_value is not None and equal_pair_mean_value >= materiality:
            return TransferCriteriaDecision(
                True,
                False,
                False,
                False,
                False,
                successful,
                harmful,
                equal_pair_mean_value,
                tuple(reasons),
            )
        reasons.append("equal-pair mean below materiality")
    elif reduced_scope and len(successful) == 3 and not harmful:
        if equal_pair_mean_value is not None and equal_pair_mean_value >= materiality:
            return TransferCriteriaDecision(
                False,
                True,
                False,
                False,
                False,
                successful,
                harmful,
                equal_pair_mean_value,
                tuple(reasons),
            )
        reasons.append("reduced-scope equal-pair mean below materiality")

    positive_pairs = [
        pair
        for pair in analyzable
        if analyzable[pair].mean_gain is not None
        and analyzable[pair].mean_gain >= materiality
        and _holm_and_bca_pass(config, analyzable[pair])
        and analyzable[pair].strict_resource_valid
    ]
    no_material_benefit = not positive_pairs
    if harmful:
        return TransferCriteriaDecision(
            False,
            False,
            False,
            False,
            True,
            tuple(positive_pairs),
            harmful,
            equal_pair_mean_value,
            tuple(reasons),
        )
    if 0 < len(positive_pairs) <= 2:
        return TransferCriteriaDecision(
            False,
            False,
            True,
            False,
            False,
            tuple(positive_pairs),
            harmful,
            equal_pair_mean_value,
            tuple(reasons),
        )
    if no_material_benefit and not harmful:
        return TransferCriteriaDecision(
            False,
            False,
            False,
            True,
            False,
            (),
            harmful,
            equal_pair_mean_value,
            (*tuple(reasons), "no materially beneficial pair"),
        )
    reasons.append("criteria not met")
    return TransferCriteriaDecision(
        False,
        False,
        False,
        False,
        False,
        tuple(positive_pairs),
        harmful,
        equal_pair_mean_value,
        tuple(reasons),
    )


def _holm_and_bca_pass(config: FedorbitConfig, evidence: PairContrastEvidence) -> bool:
    claim = config.scientific.claim_criteria.strict_cross_telemetry_utility
    if evidence.holm_p is None or evidence.bca_lower is None:
        return False
    return (
        evidence.holm_p < claim.holm_adjusted_p_maximum
        and evidence.bca_lower > claim.bca_lower_bound_strictly_greater_than
    )
