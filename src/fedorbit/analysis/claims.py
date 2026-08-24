from __future__ import annotations

from dataclasses import dataclass

from fedorbit.analysis.comparisons import PairContrastEvidence, PairContrastEvidenceSet
from fedorbit.config.models import FedorbitConfig


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


def _holm_and_bca_pass(config: FedorbitConfig, evidence: PairContrastEvidence) -> bool:
    claim = config.scientific.claim_criteria.strict_cross_telemetry_utility
    if evidence.holm_p is None or evidence.bca_lower is None:
        return False
    return (
        evidence.holm_p < claim.holm_adjusted_p_maximum
        and evidence.bca_lower > claim.bca_lower_bound_strictly_greater_than
    )


def _successful_pair(config: FedorbitConfig, evidence: PairContrastEvidence) -> bool:
    materiality = config.scientific.materiality.realized_relative_macro_ce
    return (
        evidence.mean_gain is not None
        and evidence.mean_gain >= materiality
        and _holm_and_bca_pass(config, evidence)
        and evidence.strict_resource_valid
    )


def evaluate_transfer_style_criteria(
    config: FedorbitConfig,
    evidence_set: PairContrastEvidenceSet,
    removed_before_outcome_inspection: bool,
) -> TransferCriteriaDecision:
    claim = config.scientific.claim_criteria.strict_cross_telemetry_utility
    materiality = config.scientific.materiality.realized_relative_macro_ce
    harm_threshold = config.scientific.materiality.harmful_transfer_relative_macro_ce_gain
    evidence_by_pair = {entry.directed_pair: entry for entry in evidence_set.entries}
    all_pairs = tuple(sorted(evidence_by_pair))
    analyzable = tuple(
        evidence_by_pair[pair] for pair in all_pairs if evidence_by_pair[pair].valid_seed_count > 0
    )
    reasons: list[str] = []

    if len(all_pairs) - len(analyzable) > 1:
        return TransferCriteriaDecision(
            False,
            False,
            False,
            False,
            True,
            (),
            (),
            None,
            ("more than one pair removed from claim scope",),
        )
    if any(not evidence.strict_resource_valid for evidence in evidence_set.entries):
        return TransferCriteriaDecision(
            False,
            False,
            False,
            False,
            True,
            (),
            (),
            None,
            ("strict-resource validation failed for a contributing run",),
        )

    successful = tuple(
        evidence.directed_pair for evidence in analyzable if _successful_pair(config, evidence)
    )
    harmful = tuple(
        evidence.directed_pair
        for evidence in analyzable
        if evidence.mean_gain is not None and evidence.mean_gain <= harm_threshold
    )
    gains = tuple(evidence.mean_gain for evidence in analyzable)
    numeric_gains = tuple(gain for gain in gains if gain is not None)
    equal_pair_mean = (
        sum(numeric_gains) / len(numeric_gains)
        if len(numeric_gains) == len(gains) and numeric_gains
        else None
    )

    full_scope = not removed_before_outcome_inspection and len(analyzable) == 4
    reduced_scope = removed_before_outcome_inspection and len(analyzable) == 3
    if full_scope and len(successful) >= claim.successful_primary_pairs_required and not harmful:
        if equal_pair_mean is not None and equal_pair_mean >= materiality:
            return TransferCriteriaDecision(
                True, False, False, False, False, successful, harmful, equal_pair_mean, ()
            )
        reasons.append("equal-pair mean below materiality")
    elif reduced_scope and len(successful) == 3 and not harmful:
        if equal_pair_mean is not None and equal_pair_mean >= materiality:
            return TransferCriteriaDecision(
                False, True, False, False, False, successful, harmful, equal_pair_mean, ()
            )
        reasons.append("reduced-scope equal-pair mean below materiality")

    positive_pairs = tuple(
        evidence.directed_pair
        for evidence in analyzable
        if evidence.mean_gain is not None
        and evidence.mean_gain >= materiality
        and _holm_and_bca_pass(config, evidence)
        and evidence.strict_resource_valid
    )
    if harmful:
        return TransferCriteriaDecision(
            False,
            False,
            False,
            False,
            True,
            positive_pairs,
            harmful,
            equal_pair_mean,
            tuple(reasons),
        )
    if 0 < len(positive_pairs) <= 2:
        return TransferCriteriaDecision(
            False,
            False,
            True,
            False,
            False,
            positive_pairs,
            harmful,
            equal_pair_mean,
            tuple(reasons),
        )
    if not positive_pairs:
        return TransferCriteriaDecision(
            False,
            False,
            False,
            True,
            False,
            (),
            harmful,
            equal_pair_mean,
            (*tuple(reasons), "no materially beneficial pair"),
        )
    return TransferCriteriaDecision(
        False,
        False,
        False,
        False,
        False,
        positive_pairs,
        harmful,
        equal_pair_mean,
        (*tuple(reasons), "criteria not met"),
    )
