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


@dataclass(frozen=True, slots=True)
class MechanismRetentionEvidence:
    directed_pair: str
    full_mean_gain: float
    destroyed_mean_gain: float
    equivalence_holm_p: float | None
    valid_seed_count: int

    @property
    def retention(self) -> float | None:
        if self.full_mean_gain <= 0.0:
            return None
        return self.destroyed_mean_gain / self.full_mean_gain


@dataclass(frozen=True, slots=True)
class CouplingMechanismEvidence:
    theorem_classification_accuracy: float
    real_packet_gaps: tuple[float, ...]
    pair_gap_evidence: PairContrastEvidenceSet
    retention_evidence: tuple[MechanismRetentionEvidence, ...]


@dataclass(frozen=True, slots=True)
class CouplingMechanismDecision:
    supported: bool
    material_real_packet_fraction: float | None
    material_gap_pairs: tuple[str, ...]
    mechanism_retention_pairs: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SparseUnitEvidence:
    directed_pair: str
    seed: int
    support: int
    sparse_gain: float
    dense_gain: float
    valid: bool


@dataclass(frozen=True, slots=True)
class SparsePairGainEvidence:
    directed_pair: str
    support: int
    mean_gain: float
    valid_seed_count: int


@dataclass(frozen=True, slots=True)
class SparseOperationalDecision:
    supported: bool
    dense_closeness_fraction: float | None
    useful_supports: tuple[int, ...]
    exact_sparse_correct: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConfirmationPairEvidence:
    directed_pair: str
    harm_rate_no_confirm: float
    harm_rate_confirm: float
    coverage_loss: float
    valid_seed_count: int

    @property
    def absolute_risk_reduction(self) -> float:
        return self.harm_rate_no_confirm - self.harm_rate_confirm

    @property
    def relative_risk_reduction(self) -> float | None:
        if self.harm_rate_no_confirm <= 0.0:
            return None
        return self.absolute_risk_reduction / self.harm_rate_no_confirm


@dataclass(frozen=True, slots=True)
class ConfirmationSafetyDecision:
    supported: bool
    qualifying_pairs: tuple[str, ...]
    equal_pair_absolute_risk_reduction: float | None
    equal_pair_relative_risk_reduction: float | None
    reasons: tuple[str, ...]


def _holm_and_bca_pass(
    evidence: PairContrastEvidence,
    holm_maximum: float,
    bca_lower_minimum: float,
) -> bool:
    if evidence.holm_p is None or evidence.bca_lower is None:
        return False
    return evidence.holm_p < holm_maximum and evidence.bca_lower > bca_lower_minimum


def _evaluate_transfer_criteria(
    config: FedorbitConfig,
    evidence_set: PairContrastEvidenceSet,
    removed_before_outcome_inspection: bool,
    successful_pairs_required: int,
    holm_maximum: float,
    bca_lower_minimum: float,
) -> TransferCriteriaDecision:
    materiality = config.scientific.materiality.realized_relative_macro_ce
    harm_threshold = config.scientific.materiality.harmful_transfer_relative_macro_ce_gain
    minimum_seeds = config.scientific.statistics.minimum_valid_paired_seeds
    evidence_by_pair = {entry.directed_pair: entry for entry in evidence_set.entries}
    all_pairs = tuple(sorted(evidence_by_pair))
    analyzable = tuple(
        evidence_by_pair[pair]
        for pair in all_pairs
        if evidence_by_pair[pair].valid_seed_count >= minimum_seeds
    )
    removed_count = len(all_pairs) - len(analyzable)
    reasons: list[str] = []

    if removed_count > 1:
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
    if removed_count == 1 and not removed_before_outcome_inspection:
        return TransferCriteriaDecision(
            False,
            False,
            False,
            False,
            True,
            (),
            (),
            None,
            ("pair removal was not registered before outcome inspection",),
        )
    if any(not evidence.strict_resource_valid for evidence in analyzable):
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
        evidence.directed_pair
        for evidence in analyzable
        if evidence.mean_gain is not None
        and evidence.mean_gain >= materiality
        and _holm_and_bca_pass(evidence, holm_maximum, bca_lower_minimum)
        and evidence.strict_resource_valid
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

    full_scope = removed_count == 0
    reduced_scope = removed_count == 1 and removed_before_outcome_inspection
    if full_scope and len(successful) >= successful_pairs_required and not harmful:
        if equal_pair_mean is not None and equal_pair_mean >= materiality:
            return TransferCriteriaDecision(
                True, False, False, False, False, successful, harmful, equal_pair_mean, ()
            )
        reasons.append("equal-pair mean below materiality")
    elif reduced_scope and len(successful) == len(analyzable) and not harmful:
        if equal_pair_mean is not None and equal_pair_mean >= materiality:
            return TransferCriteriaDecision(
                False, True, False, False, False, successful, harmful, equal_pair_mean, ()
            )
        reasons.append("reduced-scope equal-pair mean below materiality")

    if harmful:
        return TransferCriteriaDecision(
            False,
            False,
            False,
            False,
            True,
            successful,
            harmful,
            equal_pair_mean,
            tuple(reasons),
        )
    if successful:
        return TransferCriteriaDecision(
            False,
            False,
            True,
            False,
            False,
            successful,
            harmful,
            equal_pair_mean,
            tuple(reasons),
        )
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


def evaluate_transfer_style_criteria(
    config: FedorbitConfig,
    evidence_set: PairContrastEvidenceSet,
    removed_before_outcome_inspection: bool,
) -> TransferCriteriaDecision:
    criteria = config.scientific.claim_criteria.strict_cross_telemetry_utility
    return _evaluate_transfer_criteria(
        config,
        evidence_set,
        removed_before_outcome_inspection,
        criteria.successful_primary_pairs_required,
        criteria.holm_adjusted_p_maximum,
        criteria.bca_lower_bound_strictly_greater_than,
    )


def evaluate_external_source_value(
    config: FedorbitConfig,
    evidence_set: PairContrastEvidenceSet,
    removed_before_outcome_inspection: bool,
) -> TransferCriteriaDecision:
    criteria = config.scientific.claim_criteria.external_source_value_vs_local_sir
    return _evaluate_transfer_criteria(
        config,
        evidence_set,
        removed_before_outcome_inspection,
        criteria.successful_primary_pairs_required,
        criteria.holm_adjusted_p_maximum,
        criteria.bca_lower_bound_strictly_greater_than,
    )


def evaluate_coupling_mechanism(
    config: FedorbitConfig,
    evidence: CouplingMechanismEvidence,
) -> CouplingMechanismDecision:
    criteria = config.scientific.claim_criteria.coupling_mechanism
    materiality = config.scientific.materiality.coupling_objective_units
    minimum_seeds = config.scientific.statistics.minimum_valid_paired_seeds
    tost_alpha = config.scientific.statistics.tost_alpha_per_one_sided_test
    reasons: list[str] = []
    if evidence.real_packet_gaps:
        material_fraction = sum(gap >= materiality for gap in evidence.real_packet_gaps) / len(
            evidence.real_packet_gaps
        )
    else:
        material_fraction = None
    material_pairs = tuple(
        pair.directed_pair
        for pair in evidence.pair_gap_evidence.entries
        if pair.valid_seed_count >= minimum_seeds
        and pair.mean_gain is not None
        and pair.mean_gain >= materiality
        and pair.holm_p is not None
        and pair.holm_p < criteria.holm_adjusted_p_maximum
    )
    retention_pairs = tuple(
        item.directed_pair
        for item in evidence.retention_evidence
        if item.valid_seed_count >= minimum_seeds
        and item.equivalence_holm_p is not None
        and item.equivalence_holm_p <= tost_alpha
        and item.retention is not None
        and item.retention >= criteria.destruction_positive_gain_retention_minimum
    )
    if (
        evidence.theorem_classification_accuracy
        < criteria.theorem_zero_strict_classification_accuracy_required
    ):
        reasons.append("designed-family theorem classification criterion failed")
    if (
        material_fraction is None
        or material_fraction < criteria.real_packet_fraction_with_material_gap_minimum
    ):
        reasons.append("real-packet coupling materiality fraction criterion failed")
    if len(material_pairs) < criteria.primary_pairs_with_material_mean_gap_required:
        reasons.append("too few primary pairs have material significant coupling gaps")
    if len(retention_pairs) >= criteria.primary_pairs_with_material_mean_gap_required:
        reasons.append("coupling destruction satisfies the mechanism-retention condition")
    return CouplingMechanismDecision(
        not reasons,
        material_fraction,
        material_pairs,
        retention_pairs,
        tuple(reasons),
    )


def evaluate_sparse_operational_relevance(
    config: FedorbitConfig,
    unit_evidence: tuple[SparseUnitEvidence, ...],
    pair_gain_evidence: tuple[SparsePairGainEvidence, ...],
    exact_sparse_correct: bool,
) -> SparseOperationalDecision:
    criteria = config.scientific.claim_criteria.sparse_operational_relevance
    minimum_seeds = config.scientific.statistics.minimum_valid_paired_seeds
    materiality = config.scientific.materiality.realized_relative_macro_ce
    compared = tuple(
        unit
        for unit in unit_evidence
        if unit.valid and unit.support == criteria.compared_sparse_support
    )
    dense_closeness_fraction = (
        sum(
            unit.dense_gain - unit.sparse_gain <= criteria.dense_minus_sparse_gain_maximum
            for unit in compared
        )
        / len(compared)
        if compared
        else None
    )
    candidate_supports = (
        config.scientific.action.principal_sparse_support,
        criteria.compared_sparse_support,
    )
    useful_supports = tuple(
        support
        for support in dict.fromkeys(candidate_supports)
        if sum(
            entry.valid_seed_count >= minimum_seeds
            and entry.mean_gain >= materiality
            and entry.support == support
            for entry in pair_gain_evidence
        )
        >= criteria.primary_pairs_with_useful_gain_required
    )
    reasons: list[str] = []
    if (
        dense_closeness_fraction is None
        or dense_closeness_fraction < criteria.valid_unit_fraction_required
    ):
        reasons.append("sparse utility is not sufficiently close to dense")
    if not useful_supports:
        reasons.append("no registered sparse support has enough materially useful primary pairs")
    if not exact_sparse_correct:
        reasons.append("exact-sparse correctness is not valid")
    return SparseOperationalDecision(
        not reasons,
        dense_closeness_fraction,
        useful_supports,
        exact_sparse_correct,
        tuple(reasons),
    )


def evaluate_confirmation_safety(
    config: FedorbitConfig,
    pair_evidence: tuple[ConfirmationPairEvidence, ...],
) -> ConfirmationSafetyDecision:
    criteria = config.scientific.claim_criteria.confirmation_safety
    minimum_seeds = config.scientific.statistics.minimum_valid_paired_seeds
    analyzable = tuple(item for item in pair_evidence if item.valid_seed_count >= minimum_seeds)
    qualifying = tuple(
        item.directed_pair
        for item in analyzable
        if (
            item.absolute_risk_reduction >= criteria.absolute_risk_reduction_minimum
            or (
                item.relative_risk_reduction is not None
                and item.relative_risk_reduction >= criteria.relative_risk_reduction_minimum
            )
        )
        and item.coverage_loss <= criteria.qualifying_pair_coverage_loss_maximum
    )
    arr_values = tuple(item.absolute_risk_reduction for item in analyzable)
    equal_arr = sum(arr_values) / len(arr_values) if arr_values else None
    rrr_values = tuple(
        item.relative_risk_reduction
        for item in analyzable
        if item.relative_risk_reduction is not None
    )
    equal_rrr = (
        sum(rrr_values) / len(rrr_values)
        if len(rrr_values) == len(analyzable) and rrr_values
        else None
    )
    reasons: list[str] = []
    if len(qualifying) < criteria.qualifying_primary_pairs_required:
        reasons.append("too few primary pairs meet harm-reduction and coverage criteria")
    if any(
        item.harm_rate_confirm - item.harm_rate_no_confirm
        > criteria.pair_harmful_rate_worsening_maximum
        for item in analyzable
    ):
        reasons.append("a primary pair exceeds the harmful-rate worsening ceiling")
    if any(item.coverage_loss > criteria.pair_coverage_loss_maximum for item in analyzable):
        reasons.append("a primary pair exceeds the coverage-loss ceiling")
    equal_summary_passes = (
        equal_arr is not None and equal_arr >= criteria.equal_pair_absolute_risk_reduction_minimum
    ) or (
        equal_rrr is not None and equal_rrr >= criteria.equal_pair_relative_risk_reduction_minimum
    )
    if not equal_summary_passes:
        reasons.append("equal-pair harm-reduction summary does not meet its threshold")
    return ConfirmationSafetyDecision(
        not reasons,
        qualifying,
        equal_arr,
        equal_rrr,
        tuple(reasons),
    )
