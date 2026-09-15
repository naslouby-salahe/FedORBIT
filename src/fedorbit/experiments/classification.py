from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from fedorbit.analysis.metrics import CrossEntropy, relative_macro_ce_gain
from fedorbit.analysis.records import (
    ComparisonDecision,
    PairedComparisonRecord,
)
from fedorbit.config.loading import active_config
from fedorbit.experiments.catalogue import ExperimentExecutionRequest, build_catalogue
from fedorbit.experiments.synthesis import (
    completed_experiment_metric_records as _completed_experiment_metric_records,
)
from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
)
from fedorbit.infrastructure.manifests import (
    ReusableArtifactManifest,
)
from fedorbit.infrastructure.runtime import (
    execution_logger,
)
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
)
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactName,
    ArtifactState,
    CellUnavailabilityReason,
    ConfigurationSection,
    DirectedPairName,
    DomainModel,
    Estimate,
    EvidenceAdjudication,
    EvidenceCompleteness,
    EvidenceHypothesis,
    EvidenceStatus,
    ExecutionEventName,
    ExperimentName,
    ExperimentSeed,
    FieldDescription,
    Floor,
    ImplementationIdentity,
    Index,
    MetricId,
    MultiplicityFamily,
    PairCount,
    RelativeGain,
    ReportArtifactName,
    SignificanceLevel,
    SimplificationRuleName,
    SimplificationRuleState,
    StableJsonPayload,
    TransferMethod,
)


@dataclass(frozen=True, slots=True)
class RuleVerdict:
    state: SimplificationRuleState
    reason: FieldDescription


def _latest_synthesis_manifest(store: ArtifactStore) -> ReusableArtifactManifest | None:
    return store.current_completed_manifest(ExperimentName.STATISTICAL_SYNTHESIS.value)


def _status(
    status: EvidenceStatus,
    materiality: str,
    statistical: str,
    completeness: EvidenceCompleteness,
) -> EvidenceAdjudication:
    return EvidenceAdjudication(
        status,
        FieldDescription(materiality),
        FieldDescription(statistical),
        completeness,
    )


def _not_tested(
    reason: str,
) -> EvidenceAdjudication:
    return _status(
        EvidenceStatus.NOT_TESTED,
        "not evaluated",
        reason,
        EvidenceCompleteness.INCOMPLETE,
    )


def _supported(
    materiality: str,
    statistical: str,
) -> EvidenceAdjudication:
    return _status(
        EvidenceStatus.SUPPORTED, materiality, statistical, EvidenceCompleteness.COMPLETE
    )


def _not_supported(
    materiality: str,
    statistical: str,
) -> EvidenceAdjudication:
    return _status(
        EvidenceStatus.NOT_SUPPORTED, materiality, statistical, EvidenceCompleteness.COMPLETE
    )


def _partial(
    materiality: str,
    statistical: str,
) -> EvidenceAdjudication:
    return _status(
        EvidenceStatus.PARTIALLY_SUPPORTED, materiality, statistical, EvidenceCompleteness.COMPLETE
    )


def _mechanism_only(
    materiality: str,
    statistical: str,
) -> EvidenceAdjudication:
    return _status(
        EvidenceStatus.MECHANISM_ONLY, materiality, statistical, EvidenceCompleteness.COMPLETE
    )


def _conditional(
    materiality: str,
    statistical: str,
) -> EvidenceAdjudication:
    return _status(
        EvidenceStatus.CONDITIONAL, materiality, statistical, EvidenceCompleteness.COMPLETE
    )


def _null_result(
    materiality: str,
    statistical: str,
) -> EvidenceAdjudication:
    return _status(
        EvidenceStatus.NULL_RESULT, materiality, statistical, EvidenceCompleteness.COMPLETE
    )


def _cell_bool_field(
    cell: Mapping[str, int | float | str | list[int]],
    key: str,
) -> bool:
    return bool(cell.get(key, False))


def _theorem_cells(
    store: ArtifactStore,
) -> tuple[Mapping[str, int | float | str | list[int]], ...]:
    cells: list[Mapping[str, int | float | str | list[int]]] = []
    for manifest in store.all_manifests():
        if (
            ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION.value
            not in manifest.semantic_producer_coordinates
        ):
            continue
        try:
            resolved = store.resolve(manifest.artifact_id)
        except ValueError:
            continue
        if resolved.state != ArtifactState.COMPLETED or not resolved.payload_paths:
            continue
        payload = json.loads(Path(resolved.payload_paths[0]).read_text(encoding="utf-8"))
        cell = payload.get("cell")
        if isinstance(cell, dict):
            cells.append(cast(Mapping[str, int | float | str | list[int]], cell))
    return tuple(cells)


def _metric_values(
    store: ArtifactStore, experiment: ExperimentName, metric_name: MetricId
) -> tuple[Estimate, ...]:
    from fedorbit.experiments.synthesis import completed_experiment_metric_records

    return tuple(
        record.metric_value
        for record in completed_experiment_metric_records(store, experiment)
        if record.metric_name == metric_name and record.valid and record.metric_value is not None
    )


def _diagnostic_metric_values(
    store: ArtifactStore, experiment: ExperimentName, metric_name: MetricId
) -> tuple[Estimate, ...]:
    from fedorbit.experiments.synthesis import completed_experiment_diagnostic_metric_records

    return tuple(
        record.metric_value
        for record in completed_experiment_diagnostic_metric_records(store, experiment)
        if record.metric_name == metric_name and record.valid and record.metric_value is not None
    )


def _pair_records(
    comparisons: tuple[PairedComparisonRecord, ...],
    method_a: TransferMethod,
    method_b: TransferMethod,
) -> tuple[PairedComparisonRecord, ...]:
    return tuple(
        record
        for record in comparisons
        if record.method_a == method_a and record.method_b == method_b
    )


def _successful_pairs(
    records: tuple[PairedComparisonRecord, ...],
    holm_maximum: SignificanceLevel,
    bca_floor: Floor,
) -> tuple[PairedComparisonRecord, ...]:
    return tuple(
        record
        for record in records
        if record.decision == ComparisonDecision.SUPERIOR
        and record.holm_p is not None
        and record.holm_p <= holm_maximum
        and record.bca_ci_low is not None
        and record.bca_ci_low > bca_floor
        and record.mean_difference is not None
        and record.materiality_threshold is not None
        and record.mean_difference >= record.materiality_threshold
    )


def _harmful_pairs(
    records: tuple[PairedComparisonRecord, ...],
) -> tuple[PairedComparisonRecord, ...]:
    threshold = active_config().scientific.materiality.harmful_transfer_relative_macro_ce_gain
    return tuple(
        record
        for record in records
        if record.mean_difference is not None and record.mean_difference <= threshold
    )


def _local_reference_dominant_pairs(
    records: tuple[PairedComparisonRecord, ...],
    holm_maximum: SignificanceLevel,
) -> frozenset[DirectedPairName]:
    threshold = active_config().scientific.materiality.realized_relative_macro_ce
    dominant: set[DirectedPairName] = set()
    for record in records:
        if record.decision == ComparisonDecision.EQUIVALENT:
            dominant.add(record.pair)
            continue
        if (
            record.mean_difference is not None
            and record.holm_p is not None
            and record.bca_ci_high is not None
            and record.mean_difference <= -threshold
            and record.holm_p <= holm_maximum
            and record.bca_ci_high < 0.0
        ):
            dominant.add(record.pair)
    return frozenset(dominant)


def scientific_algorithmic_failure_seeds(store: ArtifactStore) -> Mapping[DirectedPairName, Index]:
    from fedorbit.experiments.synthesis import completed_experiment_metric_records

    records = completed_experiment_metric_records(
        store, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER
    )
    marker = ArtifactIdentifier(CellUnavailabilityReason.SCIENTIFIC_ALGORITHMIC_FAILURE.value)
    counts: OrderedDict[DirectedPairName, Index] = OrderedDict()
    for record in records:
        if record.metric_name is not MetricId.ABSTENTION_INDICATOR:
            continue
        if record.valid or record.invalid_reason is None:
            continue
        if marker not in record.input_artifact_ids:
            continue
        counts[record.pair] = counts.get(record.pair, 0) + 1
    return counts


def pairs_blocked_by_scientific_failure(store: ArtifactStore) -> frozenset[DirectedPairName]:
    allowed = active_config().runtime.failure_handling.solver_failures_per_pair_allowed
    return frozenset(
        pair
        for pair, count in scientific_algorithmic_failure_seeds(store).items()
        if count > allowed
    )


def utility_family_status(
    comparisons: tuple[PairedComparisonRecord, ...],
    method_a: TransferMethod,
    method_b: TransferMethod,
    holm_maximum: float,
    bca_floor: Floor,
    required_pairs: int,
    kill_local_reference: bool,
    blocked_pairs: frozenset[DirectedPairName] = frozenset(),
) -> EvidenceAdjudication:
    all_records = _pair_records(comparisons, method_a, method_b)
    records = tuple(record for record in all_records if record.pair not in blocked_pairs)
    if not records:
        if all_records:
            return _not_supported(
                "excluded by the scientific-algorithmic-failure rule",
                "more than one failed confirmatory seed in every eligible pair",
            )
        return _not_tested("no contrasts")
    harmful = _harmful_pairs(records)
    if harmful:
        return _not_supported("material harm on a primary pair", "failure rule")
    successful = _successful_pairs(records, holm_maximum, bca_floor)
    if len(successful) >= required_pairs:
        return _supported("material", "holm and BCa satisfied")
    if kill_local_reference and not successful:
        dominant = _local_reference_dominant_pairs(records, holm_maximum)
        if len(dominant) >= required_pairs:
            return _not_supported(
                "local reference sufficient", "equivalence or superiority kill fired"
            )
    analyzable = len({record.pair for record in records})
    if (
        analyzable == 3
        and len(successful) == 3
        and len(active_config().scientific.datasets.primary_directed_pairs) == 6
    ):
        return _conditional("three eligible pairs", "pre-outcome scope reduction")
    if successful:
        return _partial("subset of pairs material", "full pair threshold unmet")
    return _null_result("no material pair", "no harm")


def _classify_exactness(store: ArtifactStore) -> EvidenceAdjudication:
    cells = _theorem_cells(store)
    if not cells:
        return _not_tested("no theorem cells")
    wrong = sum(1 for cell in cells if not _cell_bool_field(cell, "exact_minima"))
    invalid = sum(1 for cell in cells if not _cell_bool_field(cell, "valid_certificate"))
    if wrong != 0 or invalid != 0:
        return _not_supported("wrong minima or invalid certificates", "exactness failed")
    required = (
        build_catalogue()
        .definition(ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION)
        .derived_planned_cells
    )
    if len(cells) < required:
        return _not_tested(f"{len(cells)} of {required} required truth-available cells evaluated")
    return _supported("separator exact on registered cells", "certificate verified")


def _ablation_pair_method_means(
    store: ArtifactStore, method: TransferMethod
) -> Mapping[DirectedPairName, Estimate]:
    from fedorbit.experiments.synthesis import completed_experiment_metric_records

    by_pair: OrderedDict[DirectedPairName, list[Estimate]] = OrderedDict()
    for record in completed_experiment_metric_records(store, ExperimentName.MECHANISM_ABLATIONS):
        if (
            record.method != method
            or record.metric_name != MetricId.RELATIVE_MACRO_CE_GAIN
            or not record.valid
            or record.metric_value is None
        ):
            continue
        by_pair.setdefault(record.pair, []).append(record.metric_value)
    means: OrderedDict[DirectedPairName, Estimate] = OrderedDict()
    for pair, values in by_pair.items():
        means[pair] = sum(values) / len(values)
    return means


def _mechanism_retention_pairs(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> PairCount:
    criteria = active_config().scientific.evaluation_criteria.coupling_mechanism
    full_means = _ablation_pair_method_means(store, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER)
    destroyed_means = _ablation_pair_method_means(store, TransferMethod.COUPLING_DESTROYED_FEDORBIT)
    equivalent_pairs = {
        record.pair
        for record in comparisons
        if record.family == MultiplicityFamily.MECHANISM_ABLATIONS
        and record.decision == ComparisonDecision.EQUIVALENT
    }
    retention_pairs = 0
    for pair, full_mean in full_means.items():
        if pair not in equivalent_pairs or full_mean <= 0.0:
            continue
        destroyed_mean = destroyed_means.get(pair)
        if destroyed_mean is None:
            continue
        retention = destroyed_mean / full_mean
        if retention >= criteria.destruction_positive_gain_retention_minimum:
            retention_pairs += 1
    return retention_pairs


def _material_coupling_pairs(
    comparisons: tuple[PairedComparisonRecord, ...],
) -> PairCount:
    return sum(
        1
        for record in comparisons
        if record.family == MultiplicityFamily.COUPLING_MECHANISM
        and record.decision == ComparisonDecision.SUPERIOR
    )


def _classify_joint_correspondence(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> EvidenceAdjudication:
    criteria = active_config().scientific.evaluation_criteria.coupling_mechanism
    material = active_config().scientific.materiality.coupling_objective_units
    synthetic = _metric_values(
        store,
        ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION,
        MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP,
    )
    if not synthetic:
        return _not_tested("no coupling gaps")
    retention_pairs = _mechanism_retention_pairs(store, comparisons)
    retention_present = retention_pairs >= criteria.primary_pairs_with_material_mean_gap_required
    if retention_present:
        return _not_supported("coupling destruction retains gain", "mechanism attribution fails")
    synthetic_fraction = sum(1 for gap in synthetic if gap > material) / len(synthetic)
    accuracy = criteria.theorem_zero_strict_classification_accuracy_required
    synthetic_pass = synthetic_fraction >= accuracy or any(gap > material for gap in synthetic)
    if not synthetic_pass:
        return _not_supported("synthetic mechanism criterion failed", "gap fraction unmet")
    real_gaps = _metric_values(
        store,
        ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION,
        MetricId.ROBUST_COUPLING_VALUE_GAP,
    )
    if not real_gaps:
        return _mechanism_only("synthetic mechanism complete", "real-packet criterion unavailable")
    real_fraction = sum(1 for gap in real_gaps if gap > material) / len(real_gaps)
    material_pairs = _material_coupling_pairs(comparisons)
    real_pass = (
        real_fraction >= criteria.real_packet_fraction_with_material_gap_minimum
        and material_pairs >= criteria.primary_pairs_with_material_mean_gap_required
    )
    if real_pass:
        return _supported("synthetic and real-packet gaps material", "coupling criteria passed")
    return _mechanism_only("synthetic mechanism complete", "real-packet materiality not reached")


def _classify_action_certification(
    store: ArtifactStore,
) -> EvidenceAdjudication:
    tolerance = active_config().scientific.materiality.coupling_objective_units
    common = _metric_values(
        store,
        ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP,
        MetricId.EXACT_MAP_ACTION_VALUE,
    )
    robust = _metric_values(
        store,
        ExperimentName.ROBUST_COMPROMISE_UNDER_UNIDENTIFIED_MAP,
        MetricId.EXACT_MAP_ACTION_VALUE,
    )
    bounds = _metric_values(
        store,
        ExperimentName.EXACT_MAP_VALUE_BOUND_VALIDATION,
        MetricId.ORBIT_RADIUS_MAP_BOUND,
    )
    values = _metric_values(
        store,
        ExperimentName.EXACT_MAP_VALUE_BOUND_VALIDATION,
        MetricId.EXACT_MAP_ACTION_VALUE,
    )
    if not common and not robust:
        return _not_tested("no unresolved-map fixtures")
    bound_valid = True
    if bounds and values:
        count = min(len(values), len(bounds))
        bound_valid = all(values[index] <= bounds[index] + tolerance for index in range(count))
    if not bound_valid:
        return _not_supported("orbit-radius bound violated", "map-value bound failure")
    common_ok = bool(common) and all(abs(value) <= tolerance for value in common)
    robust_ok = bool(robust) and all(abs(value) <= tolerance or value > 0.0 for value in robust)
    if common_ok and robust_ok:
        return _supported("common-action and robust-compromise constructed", "map bound valid")
    if common_ok or robust_ok:
        return _partial("one controlled family constructed", "map bound valid")
    return _not_supported("exact map recovery required", "neither controlled family holds")


def _classify_sparse_operational(
    store: ArtifactStore,
) -> EvidenceAdjudication:
    required = active_config().scientific.evaluation_criteria.sparse_operational_relevance
    useful_floor = active_config().scientific.materiality.useful_transfer_relative_macro_ce_gain
    sparse = _metric_values(
        store, ExperimentName.SPARSITY_AND_DENSE_FALLBACK, MetricId.RELATIVE_MACRO_CE_GAIN
    )
    if not sparse:
        return _not_tested("no sparsity metrics")
    kill = _sparse_irrelevance_applied(store)
    if kill:
        return _not_supported("dense dominates sparse supports", "sparse-irrelevance kill")
    useful = sum(1 for gain in sparse if gain > useful_floor)
    if useful >= required.primary_pairs_with_useful_gain_required:
        return _supported("sparse support retains useful gain", "pair threshold met")
    if useful:
        return _partial("at least one sparse support useful", "full operational rule unmet")
    return _null_result("no useful sparse support", "irrelevance kill not fired")


def _classify_confirmation(
    comparisons: tuple[PairedComparisonRecord, ...],
    blocked_pairs: frozenset[DirectedPairName] = frozenset(),
) -> EvidenceAdjudication:
    required = active_config().scientific.evaluation_criteria.confirmation_safety
    rows = tuple(
        record
        for record in comparisons
        if record.family == MultiplicityFamily.CONFIRMATION_SAFETY
        and record.pair not in blocked_pairs
    )
    if not rows:
        return _not_tested("no confirmation contrasts")
    worsening = tuple(
        record
        for record in rows
        if record.mean_difference is not None
        and record.mean_difference < -required.pair_harmful_rate_worsening_maximum
    )
    if worsening:
        return _not_supported("harmful-rate worsening", "safety failure")
    successful = sum(1 for record in rows if record.decision == ComparisonDecision.SUPERIOR)
    if successful >= required.qualifying_primary_pairs_required:
        return _supported("confirmation reduces harmful rate", "ARR criterion met")
    if successful:
        return _partial("subset of pairs meet ARR/RRR", "qualifying-pair threshold unmet")
    return _null_result("no pair meets harm reduction", "no worsening")


def _classify_work_structure(store: ArtifactStore) -> EvidenceAdjudication:
    values = _metric_values(
        store, ExperimentName.SCALABILITY_AND_EFFICIENCY, MetricId.WORK_STRUCTURE_SPEARMAN
    )
    certificates = _metric_values(
        store,
        ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK,
        MetricId.CORRESPONDENCE_CERTIFICATE_VALIDITY,
    )
    if not values and not certificates:
        return _not_tested("no work-structure Spearman")
    if certificates and any(value < 1.0 for value in certificates):
        return _not_supported("counter or certificate mismatch", "approximation required")
    if not values:
        return _partial("certificates match", "runtime-trend evidence missing")
    if values[0] > 0.0:
        return _supported("runtime tracks predicted work", "Spearman positive")
    return _partial("certificates match", "non-positive Spearman")


def _applied_rules(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> Mapping[SimplificationRuleName, SimplificationRuleState]:
    states: OrderedDict[SimplificationRuleName, SimplificationRuleState] = OrderedDict()
    for payload in simplification_rule_states(store, comparisons):
        fields = cast(Mapping[str, StableJsonPayload], payload)
        states[SimplificationRuleName(str(fields["rule"]))] = SimplificationRuleState(
            str(fields["state"])
        )
    return states


def _rule_state(
    states: Mapping[SimplificationRuleName, SimplificationRuleState],
    rule: SimplificationRuleName,
) -> SimplificationRuleState:
    return states.get(rule, SimplificationRuleState.NOT_TESTED)


def _exactness_gate(store: ArtifactStore) -> EvidenceAdjudication | None:
    exactness = _classify_exactness(store)
    if exactness.status == EvidenceStatus.NOT_SUPPORTED:
        return _not_supported(
            "exact-sparse separator exactness failed",
            "exactness-failure kill rule",
        )
    return None


def exact_sparse_separator_exactness_evidence(store: ArtifactStore) -> EvidenceAdjudication:
    return _classify_exactness(store)


def joint_correspondence_evidence(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> EvidenceAdjudication:
    gated = _exactness_gate(store)
    if gated is not None:
        return gated
    rules = _applied_rules(store, comparisons)
    if (
        _rule_state(rules, SimplificationRuleName.THEORY_CLASSIFICATION_FAILURE)
        is SimplificationRuleState.APPLIED
    ):
        return _not_supported(
            "designed classification disagrees with exhaustive truth",
            "theory-classification-failure kill rule",
        )
    if (
        _rule_state(rules, SimplificationRuleName.COUPLING_DESTRUCTION_RETAINS_GAIN)
        is SimplificationRuleState.APPLIED
    ):
        return _mechanism_only(
            "coupling gain retained under destruction",
            "causal coupling attribution abandoned",
        )
    return _classify_joint_correspondence(store, comparisons)


def action_certification_evidence(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> EvidenceAdjudication:
    rules = _applied_rules(store, comparisons)
    if (
        _rule_state(rules, SimplificationRuleName.THEORY_CLASSIFICATION_FAILURE)
        is SimplificationRuleState.APPLIED
    ):
        return _not_supported(
            "designed classification disagrees with exhaustive truth",
            "theory-classification-failure kill rule",
        )
    if (
        _rule_state(rules, SimplificationRuleName.UNRESOLVED_MAP_REGIME_LACKS_PRACTICAL_MOTIVATION)
        is SimplificationRuleState.APPLIED
    ):
        return _conditional(
            "map trivial to reconstruct under the motivating resources",
            "conditional algorithmic result only",
        )
    return _classify_action_certification(store)


def strict_cross_telemetry_transfer_utility_evidence(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> EvidenceAdjudication:
    gated = _exactness_gate(store)
    if gated is not None:
        return gated
    required = active_config().scientific.evaluation_criteria.strict_cross_telemetry_utility
    blocked_pairs = pairs_blocked_by_scientific_failure(store)
    return utility_family_status(
        comparisons,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_ONLY,
        required.holm_adjusted_p_maximum,
        required.bca_lower_bound_strictly_greater_than,
        required.successful_primary_pairs_required,
        False,
        blocked_pairs,
    )


def external_procedural_evidence_value(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> EvidenceAdjudication:
    gated = _exactness_gate(store)
    if gated is not None:
        return gated
    rules = _applied_rules(store, comparisons)
    if (
        _rule_state(rules, SimplificationRuleName.LOCAL_SIR_IS_SUFFICIENT)
        is SimplificationRuleState.APPLIED
    ):
        return _not_supported(
            "local reference equivalent or superior on the required pairs",
            "local-SIR-sufficiency kill rule",
        )
    required = active_config().scientific.evaluation_criteria.external_source_value_vs_local_sir
    blocked_pairs = pairs_blocked_by_scientific_failure(store)
    return utility_family_status(
        comparisons,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_SIR,
        required.holm_adjusted_p_maximum,
        required.bca_lower_bound_strictly_greater_than,
        required.successful_primary_pairs_required,
        True,
        blocked_pairs,
    )


def sparse_support_operational_relevance_evidence(store: ArtifactStore) -> EvidenceAdjudication:
    gated = _exactness_gate(store)
    if gated is not None:
        return gated
    return _classify_sparse_operational(store)


def target_confirmation_safety_evidence(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> EvidenceAdjudication:
    gated = _exactness_gate(store)
    if gated is not None:
        return gated
    rules = _applied_rules(store, comparisons)
    if (
        _rule_state(rules, SimplificationRuleName.CONFIRMATION_HAS_NO_SAFETY_VALUE)
        is SimplificationRuleState.APPLIED
    ):
        return _not_supported(
            "confirmation harm reduction and coverage loss outside thresholds",
            "confirmation-safety kill rule",
        )
    blocked_pairs = pairs_blocked_by_scientific_failure(store)
    return _classify_confirmation(comparisons, blocked_pairs)


def sparse_solver_work_structure_agreement_evidence(store: ArtifactStore) -> EvidenceAdjudication:
    gated = _exactness_gate(store)
    if gated is not None:
        return gated
    return _classify_work_structure(store)


def _rule_payload(
    rule: SimplificationRuleName,
    verdict: RuleVerdict,
) -> StableJsonPayload:
    return cast(
        StableJsonPayload, OrderedDict(rule=rule, state=verdict.state, reason=verdict.reason)
    )


def _median(
    values: tuple[Estimate, ...],
) -> Estimate | None:
    if not values:
        return None
    ordered = tuple(sorted(values))
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _sparse_irrelevance_applied(store: ArtifactStore) -> bool:
    gains = _metric_values(
        store, ExperimentName.SPARSITY_AND_DENSE_FALLBACK, MetricId.RELATIVE_MACRO_CE_GAIN
    )
    if not gains:
        return False
    scientific = active_config().scientific
    rule = scientific.simplification_rules.sparse_support_is_operationally_irrelevant
    useful_floor = active_config().scientific.materiality.useful_transfer_relative_macro_ce_gain
    useful_fraction = sum(1 for gain in gains if gain > useful_floor) / len(gains)
    return useful_fraction < rule.valid_primary_unit_fraction_minimum


def _generic_qap_rule(
    store: ArtifactStore,
) -> RuleVerdict:
    from fedorbit.experiments.synthesis import completed_experiment_metric_records

    records = completed_experiment_metric_records(
        store, ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK
    )
    sparse = tuple(
        record.metric_value
        for record in records
        if record.method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
        and record.metric_name == MetricId.WALL_TIME
        and record.valid
        and record.metric_value is not None
    )
    qap = tuple(
        record.metric_value
        for record in records
        if record.method == TransferMethod.GENERIC_EXACT_QAP
        and record.metric_name == MetricId.WALL_TIME
        and record.valid
        and record.metric_value is not None
    )
    if not sparse or not qap:
        return RuleVerdict(
            SimplificationRuleState.NOT_TESTED,
            FieldDescription("solver runtime ratios not jointly populated"),
        )
    sparse_median = _median(sparse)
    qap_median = _median(qap)
    if sparse_median is None or qap_median is None or sparse_median <= 0.0:
        return RuleVerdict(
            SimplificationRuleState.NOT_TESTED,
            FieldDescription("non-positive exact-sparse runtime"),
        )
    rule = active_config().scientific.simplification_rules.generic_qap_dominates
    ratio = qap_median / sparse_median
    if ratio <= rule.median_runtime_ratio_to_exact_sparse_maximum:
        return RuleVerdict(
            SimplificationRuleState.APPLIED,
            FieldDescription("QAP median runtime at or below exact-sparse"),
        )
    return RuleVerdict(
        SimplificationRuleState.NOT_APPLIED,
        FieldDescription("QAP median runtime exceeds exact-sparse"),
    )


def _strict_interface_rule(
    comparisons: tuple[PairedComparisonRecord, ...],
) -> RuleVerdict:
    rule = active_config().scientific.simplification_rules.strict_interface_removes_gain
    fedorbit = _pair_records(
        comparisons, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, TransferMethod.LOCAL_ONLY
    )
    oracle = _pair_records(comparisons, TransferMethod.EXACT_MAP_ORACLE, TransferMethod.LOCAL_ONLY)
    if not fedorbit or not oracle:
        return RuleVerdict(
            SimplificationRuleState.NOT_TESTED,
            FieldDescription("FedORBIT and oracle contrasts not jointly populated"),
        )
    oracle_by_pair = OrderedDict((record.pair, record) for record in oracle)
    hits = 0
    for record in fedorbit:
        oracle_row = oracle_by_pair.get(record.pair)
        if oracle_row is None or record.mean_difference is None or record.bca_ci_high is None:
            continue
        if record.mean_difference > rule.point_gain_maximum:
            continue
        if record.bca_ci_high >= rule.bca_upper_bound_maximum:
            continue
        if oracle_row.decision != ComparisonDecision.SUPERIOR:
            continue
        hits += 1
    if hits >= rule.primary_pair_majority_required:
        return RuleVerdict(
            SimplificationRuleState.APPLIED,
            FieldDescription("strict interface removes gain while oracle succeeds"),
        )
    return RuleVerdict(
        SimplificationRuleState.NOT_APPLIED,
        FieldDescription("strict-interface majority not reached"),
    )


def _source_response_rule(store: ArtifactStore) -> RuleVerdict:
    failures = _diagnostic_metric_values(
        store,
        ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION,
        MetricId.RESOURCE_LIMIT_INDICATOR,
    )
    if not failures:
        return RuleVerdict(
            SimplificationRuleState.NOT_TESTED,
            FieldDescription("principal source-packet failure fraction not persisted"),
        )
    rule = active_config().scientific.simplification_rules.source_response_is_too_unstable
    fraction = sum(1 for value in failures if value > 0.0) / len(failures)
    if fraction > rule.principal_source_packet_failure_fraction_strictly_greater_than:
        return RuleVerdict(
            SimplificationRuleState.APPLIED,
            FieldDescription("source-packet failure fraction exceeds threshold"),
        )
    return RuleVerdict(
        SimplificationRuleState.NOT_APPLIED,
        FieldDescription("source-packet failure fraction below threshold"),
    )


def _exactness_failure_rule(
    store: ArtifactStore,
) -> RuleVerdict:
    errors = _metric_values(
        store,
        ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION,
        MetricId.RELATIVE_OBJECTIVE_ERROR,
    )
    certificates = _metric_values(
        store,
        ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION,
        MetricId.CORRESPONDENCE_CERTIFICATE_VALIDITY,
    )
    if not errors and not certificates:
        return RuleVerdict(
            SimplificationRuleState.NOT_TESTED,
            FieldDescription("no exactness or certificate evidence"),
        )
    tolerance = active_config().solvers.exact_sparse.exact_validation_absolute_tolerance
    if any(error > tolerance for error in errors) or any(
        certificate < 1.0 for certificate in certificates
    ):
        return RuleVerdict(
            SimplificationRuleState.APPLIED,
            FieldDescription("objective error above tolerance or invalid certificate"),
        )
    return RuleVerdict(
        SimplificationRuleState.NOT_APPLIED,
        FieldDescription("every recorded cell is exact with a valid certificate"),
    )


def _theory_classification_rule(
    store: ArtifactStore,
) -> RuleVerdict:
    bounds = _metric_values(
        store,
        ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION,
        MetricId.ORBIT_RADIUS_MAP_BOUND,
    )
    gaps = _metric_values(
        store,
        ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION,
        MetricId.ROBUST_COUPLING_VALUE_GAP,
    )
    if not bounds and not gaps:
        return RuleVerdict(
            SimplificationRuleState.NOT_TESTED,
            FieldDescription("no map-bound or coupling-gap evidence"),
        )
    tolerance = active_config().solvers.exact_sparse.exact_validation_absolute_tolerance
    materiality = active_config().scientific.materiality.coupling_objective_units
    if any(gap < -tolerance for gap in gaps):
        return RuleVerdict(
            SimplificationRuleState.APPLIED,
            FieldDescription("robust coupling gap below zero beyond exactness tolerance"),
        )
    if any(bound < -tolerance for bound in bounds):
        return RuleVerdict(
            SimplificationRuleState.APPLIED,
            FieldDescription("orbit-radius bound slack negative beyond exactness tolerance"),
        )
    if bounds and all(bound >= -tolerance for bound in bounds) and gaps:
        del materiality
    return RuleVerdict(
        SimplificationRuleState.NOT_APPLIED,
        FieldDescription("designed classification and map bound agree with exhaustive truth"),
    )


def _local_sir_sufficiency_rule(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
    blocked_pairs: frozenset[DirectedPairName],
) -> RuleVerdict:
    del store
    criteria = active_config().scientific.evaluation_criteria
    required = criteria.external_source_value_vs_local_sir
    support = utility_family_status(
        comparisons,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_SIR,
        required.holm_adjusted_p_maximum,
        required.bca_lower_bound_strictly_greater_than,
        required.successful_primary_pairs_required,
        True,
        blocked_pairs,
    )
    if support.status is EvidenceStatus.NOT_TESTED:
        return RuleVerdict(
            SimplificationRuleState.NOT_TESTED,
            FieldDescription("no FedORBIT versus Local-SIR contrasts"),
        )
    if support.status is EvidenceStatus.SUPPORTED:
        return RuleVerdict(
            SimplificationRuleState.NOT_APPLIED,
            FieldDescription("external-procedural-evidence support rule holds"),
        )
    records = _pair_records(
        comparisons, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, TransferMethod.LOCAL_SIR
    )
    comparable = tuple(record for record in records if record.pair not in blocked_pairs)
    dominant = _local_reference_dominant_pairs(comparable, required.holm_adjusted_p_maximum)
    if len(dominant) >= required.successful_primary_pairs_required:
        return RuleVerdict(
            SimplificationRuleState.APPLIED,
            FieldDescription("local reference equivalent or superior on the required pairs"),
        )
    return RuleVerdict(
        SimplificationRuleState.NOT_APPLIED,
        FieldDescription("local reference sufficiency not reached"),
    )


def _coupling_destruction_rule(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> RuleVerdict:
    means = _ablation_pair_method_means(store, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER)
    if not means:
        return RuleVerdict(
            SimplificationRuleState.NOT_TESTED,
            FieldDescription("no full-method ablation pair means"),
        )
    retention = _mechanism_retention_pairs(store, comparisons)
    required = (
        active_config().scientific.evaluation_criteria.coupling_mechanism
    ).primary_pairs_with_material_mean_gap_required
    if retention >= required:
        return RuleVerdict(
            SimplificationRuleState.APPLIED,
            FieldDescription("registered mechanism-retention condition present"),
        )
    return RuleVerdict(
        SimplificationRuleState.NOT_APPLIED,
        FieldDescription("mechanism-retention condition absent"),
    )


def _confirmation_safety_rule(
    store: ArtifactStore,
) -> RuleVerdict:
    criteria = active_config().scientific.evaluation_criteria.confirmation_safety
    reductions = _metric_values(
        store,
        ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY,
        MetricId.ABSOLUTE_RISK_REDUCTION,
    )
    relative = _metric_values(
        store,
        ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY,
        MetricId.RELATIVE_RISK_REDUCTION,
    )
    losses = _metric_values(
        store,
        ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY,
        MetricId.COVERAGE_LOSS,
    )
    if not reductions and not relative and not losses:
        return RuleVerdict(
            SimplificationRuleState.NOT_TESTED,
            FieldDescription("no confirmation safety evidence"),
        )
    qualifying = sum(
        1 for reduction in reductions if reduction >= criteria.absolute_risk_reduction_minimum
    )
    if not reductions:
        qualifying = sum(
            1 for value in relative if value >= criteria.relative_risk_reduction_minimum
        )
    exceeds_ceiling = any(loss > criteria.pair_coverage_loss_maximum for loss in losses)
    if qualifying < criteria.qualifying_primary_pairs_required or exceeds_ceiling:
        return RuleVerdict(
            SimplificationRuleState.APPLIED,
            FieldDescription("harm reduction below both thresholds or coverage loss exceeded"),
        )
    return RuleVerdict(
        SimplificationRuleState.NOT_APPLIED,
        FieldDescription("confirmation harm reduction and coverage loss within thresholds"),
    )


def _unresolved_map_rule(
    store: ArtifactStore,
) -> RuleVerdict:
    indicators = _metric_values(
        store,
        ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT,
        MetricId.TRIVIAL_MAP_RECONSTRUCTION_INDICATOR,
    )
    if not indicators:
        return RuleVerdict(
            SimplificationRuleState.NOT_TESTED,
            FieldDescription("no map-availability applicability outcomes"),
        )
    if all(indicator >= 1.0 for indicator in indicators):
        return RuleVerdict(
            SimplificationRuleState.APPLIED,
            FieldDescription("every eligible primary pair is trivial to reconstruct"),
        )
    return RuleVerdict(
        SimplificationRuleState.NOT_APPLIED,
        FieldDescription("at least one eligible primary pair is not trivial to reconstruct"),
    )


def _point_matching_rule(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> RuleVerdict:
    rule = active_config().scientific.simplification_rules.point_matching_is_sufficient
    materiality = active_config().scientific.materiality
    records = _pair_records(
        comparisons,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
    )
    if not records:
        return RuleVerdict(
            SimplificationRuleState.NOT_TESTED,
            FieldDescription("no registered FedORBIT versus point-correspondence contrasts"),
        )
    minimum = active_config().scientific.statistics.minimum_valid_paired_seeds
    fedorbit_gains = _primary_pair_gains(store, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER)
    commitment_gains = _primary_pair_gains(store, TransferMethod.POINT_CORRESPONDENCE_COMMITMENT)
    evaluated = 0
    sufficient = 0
    for record in records:
        pair_gains = commitment_gains.get(record.pair)
        reference_gains = fedorbit_gains.get(record.pair)
        if pair_gains is None or reference_gains is None:
            continue
        if len(pair_gains) < minimum or len(reference_gains) < minimum:
            continue
        evaluated += 1
        commitment_harmful = sum(
            1 for gain in pair_gains if gain <= materiality.harmful_transfer_relative_macro_ce_gain
        ) / len(pair_gains)
        fedorbit_harmful = sum(
            1
            for gain in reference_gains
            if gain <= materiality.harmful_transfer_relative_macro_ce_gain
        ) / len(reference_gains)
        not_worse = commitment_harmful - fedorbit_harmful <= rule.harmful_rate_worsening_maximum
        advantage = (sum(pair_gains) / len(pair_gains)) - (
            sum(reference_gains) / len(reference_gains)
        )
        if not_worse and advantage >= rule.utility_advantage_over_fedorbit_minimum:
            sufficient += 1
    if evaluated == 0:
        return RuleVerdict(
            SimplificationRuleState.NOT_TESTED,
            FieldDescription("no pair reaches the registered minimum valid paired seeds"),
        )
    if sufficient == evaluated:
        return RuleVerdict(
            SimplificationRuleState.APPLIED,
            FieldDescription(
                "point matching is not worse and materially more useful on every pair"
            ),
        )
    return RuleVerdict(
        SimplificationRuleState.NOT_APPLIED,
        FieldDescription("point matching does not satisfy the sufficiency condition"),
    )


def _primary_pair_gains(
    store: ArtifactStore,
    method: TransferMethod,
) -> Mapping[DirectedPairName, tuple[RelativeGain, ...]]:
    records = _completed_experiment_metric_records(
        store, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER
    )
    local_only: OrderedDict[tuple[DirectedPairName, int], CrossEntropy] = OrderedDict()
    for record in records:
        if (
            record.method is TransferMethod.LOCAL_ONLY
            and record.metric_name is MetricId.MACRO_CROSS_ENTROPY
            and record.valid
            and record.metric_value is not None
        ):
            stored_value: Estimate = record.metric_value
            local_only[(DirectedPairName(record.pair), record.seed)] = CrossEntropy(stored_value)
    collected: OrderedDict[DirectedPairName, list[RelativeGain]] = OrderedDict()
    for record in records:
        if record.method is not method or record.metric_name is not MetricId.MACRO_CROSS_ENTROPY:
            continue
        if not record.valid or record.metric_value is None:
            continue
        stored_value: Estimate = record.metric_value
        reference = local_only.get((DirectedPairName(record.pair), record.seed))
        if reference is None:
            continue
        gain = relative_macro_ce_gain(reference, CrossEntropy(stored_value))
        relative: RelativeGain | None = gain.relative
        if relative is None:
            continue
        pair_key = DirectedPairName(record.pair)
        pair_values: list[RelativeGain] = collected.setdefault(pair_key, [])
        pair_values.append(relative)
    resolved: OrderedDict[DirectedPairName, tuple[RelativeGain, ...]] = OrderedDict()
    for pair, values in collected.items():
        resolved[pair] = tuple(values)
    return resolved


def simplification_rule_states(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> tuple[StableJsonPayload, ...]:
    rules = active_config().scientific.simplification_rules
    material = active_config().scientific.materiality
    gaps = _metric_values(
        store,
        ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION,
        MetricId.ROBUST_COUPLING_VALUE_GAP,
    )
    if gaps:
        below = sum(1 for gap in gaps if gap < material.coupling_objective_units) / len(gaps)
        rectangular_rule = rules.rectangularization_is_sufficient
        minimum = rectangular_rule.valid_real_packet_fraction_below_coupling_materiality_minimum
        rectangular = RuleVerdict(
            SimplificationRuleState.APPLIED
            if below >= minimum
            else SimplificationRuleState.NOT_APPLIED,
            FieldDescription("real-packet coupling below materiality"),
        )
    else:
        rectangular = RuleVerdict(
            SimplificationRuleState.NOT_TESTED, FieldDescription("no real-packet coupling gaps")
        )
    sparse_gains = _metric_values(
        store, ExperimentName.SPARSITY_AND_DENSE_FALLBACK, MetricId.RELATIVE_MACRO_CE_GAIN
    )
    if sparse_gains:
        sparse = RuleVerdict(
            SimplificationRuleState.APPLIED
            if _sparse_irrelevance_applied(store)
            else SimplificationRuleState.NOT_APPLIED,
            FieldDescription("sparse useful-unit fraction"),
        )
    else:
        sparse = RuleVerdict(
            SimplificationRuleState.NOT_TESTED, FieldDescription("no sparsity metrics")
        )
    point = _point_matching_rule(store, comparisons)
    qap = _generic_qap_rule(store)
    interface = _strict_interface_rule(comparisons)
    source = _source_response_rule(store)
    blocked_pairs = pairs_blocked_by_scientific_failure(store)
    exactness = _exactness_failure_rule(store)
    theory = _theory_classification_rule(store)
    local_sir = _local_sir_sufficiency_rule(store, comparisons, blocked_pairs)
    destruction = _coupling_destruction_rule(store, comparisons)
    confirmation = _confirmation_safety_rule(store)
    map_motivation = _unresolved_map_rule(store)
    return (
        _rule_payload(SimplificationRuleName.EXACTNESS_FAILURE, exactness),
        _rule_payload(SimplificationRuleName.RECTANGULARIZATION_IS_SUFFICIENT, rectangular),
        _rule_payload(SimplificationRuleName.THEORY_CLASSIFICATION_FAILURE, theory),
        _rule_payload(SimplificationRuleName.GENERIC_QAP_DOMINATES, qap),
        _rule_payload(SimplificationRuleName.SPARSE_SUPPORT_IS_OPERATIONALLY_IRRELEVANT, sparse),
        _rule_payload(SimplificationRuleName.LOCAL_SIR_IS_SUFFICIENT, local_sir),
        _rule_payload(SimplificationRuleName.POINT_MATCHING_IS_SUFFICIENT, point),
        _rule_payload(SimplificationRuleName.COUPLING_DESTRUCTION_RETAINS_GAIN, destruction),
        _rule_payload(SimplificationRuleName.STRICT_INTERFACE_REMOVES_GAIN, interface),
        _rule_payload(SimplificationRuleName.CONFIRMATION_HAS_NO_SAFETY_VALUE, confirmation),
        _rule_payload(SimplificationRuleName.SOURCE_RESPONSE_IS_TOO_UNSTABLE, source),
        _rule_payload(
            SimplificationRuleName.UNRESOLVED_MAP_REGIME_LACKS_PRACTICAL_MOTIVATION, map_motivation
        ),
    )


def execute_evidence_classification(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> ReusableArtifactManifest:
    from fedorbit.experiments.synthesis import completed_primary_transfer_comparison_records
    from fedorbit.experiments.validation import persist_synthetic_experiment_payload

    synthesis = _latest_synthesis_manifest(store)
    comparisons = completed_primary_transfer_comparison_records(store)
    logger = execution_logger()
    logger.event(
        ExecutionEventName.EVIDENCE_CLASSIFICATION_START,
        synthesis_present=synthesis is not None,
        comparison_rows=len(comparisons),
    )
    evaluated: tuple[tuple[EvidenceHypothesis, EvidenceAdjudication], ...] = (
        (
            EvidenceHypothesis.EXACT_SPARSE_SEPARATOR_EXACTNESS,
            exact_sparse_separator_exactness_evidence(store),
        ),
        (
            EvidenceHypothesis.JOINT_CORRESPONDENCE_AVOIDS_RECTANGULAR_PESSIMISM,
            joint_correspondence_evidence(store, comparisons),
        ),
        (
            EvidenceHypothesis.ACTION_CERTIFICATION_WITHOUT_FINE_MAP_IDENTIFICATION,
            action_certification_evidence(store, comparisons),
        ),
        (
            EvidenceHypothesis.STRICT_CROSS_TELEMETRY_TRANSFER_UTILITY,
            strict_cross_telemetry_transfer_utility_evidence(store, comparisons),
        ),
        (
            EvidenceHypothesis.VALUE_OF_EXTERNAL_PROCEDURAL_EVIDENCE,
            external_procedural_evidence_value(store, comparisons),
        ),
        (
            EvidenceHypothesis.OPERATIONAL_RELEVANCE_OF_SPARSE_SUPPORT,
            sparse_support_operational_relevance_evidence(store),
        ),
        (
            EvidenceHypothesis.TARGET_CONFIRMATION_SAFETY,
            target_confirmation_safety_evidence(store, comparisons),
        ),
        (
            EvidenceHypothesis.SPARSE_SOLVER_WORK_STRUCTURE_AGREEMENT,
            sparse_solver_work_structure_agreement_evidence(store),
        ),
    )
    rows: list[StableJsonPayload] = []
    for hypothesis, adjudication in evaluated:
        rows.append(
            cast(
                StableJsonPayload,
                OrderedDict(
                    question=hypothesis.value,
                    final_state=adjudication.status.value,
                    materiality_result=adjudication.materiality,
                    statistical_result=adjudication.statistical,
                    evidence_completeness=adjudication.completeness,
                    scope="registered primary evidence",
                    supporting_table=ReportArtifactName.EVIDENCE_STATUS,
                    supporting_figure=ReportArtifactName.REAL_TRANSFER_GAIN_FOREST_PLOT,
                    forbidden_wording="",
                ),
            )
        )
    simplification = simplification_rule_states(store, comparisons)
    seed = ExperimentSeed(0)
    return persist_synthetic_experiment_payload(
        store,
        layout,
        request,
        seed,
        lambda fingerprint: cast(
            StableJsonPayload,
            OrderedDict(
                experiment=request.experiment.value,
                dependency_fingerprint_sha256=fingerprint,
                statuses=tuple(rows),
                simplification_rules=simplification,
                synthesis_artifact_id=None if synthesis is None else synthesis.artifact_id.value,
            ),
        ),
        frozenset({ConfigurationSection.METRICS}),
        ImplementationIdentity.CLASSIFICATION_V1,
        ArtifactName("evidence-classification"),
    )


class EvidenceStatusRow(DomainModel):
    question: EvidenceHypothesis
    final_state: EvidenceStatus
    materiality_result: FieldDescription
    statistical_result: FieldDescription
    evidence_completeness: EvidenceCompleteness
    scope: FieldDescription
    supporting_table: ReportArtifactName
    supporting_figure: ReportArtifactName
    forbidden_wording: FieldDescription


def completed_evidence_status_rows(store: ArtifactStore) -> tuple[EvidenceStatusRow, ...]:
    current = store.current_completed_manifest(ExperimentName.EVIDENCE_CLASSIFICATION.value)
    if current is None:
        return ()
    payload = json.loads(Path(current.payload_paths[0]).read_text(encoding="utf-8"))
    rows = payload.get("statuses", ())
    return tuple(EvidenceStatusRow.model_validate(row) for row in rows if isinstance(row, dict))
