from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from fedorbit.analysis.records import (
    ComparisonDecision,
    PairedComparisonRecord,
)
from fedorbit.config.loading import active_config
from fedorbit.experiments.protocol import ExperimentExecutionRequest
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
    ArtifactState,
    ConfigurationSection,
    DirectedPairName,
    DomainModel,
    EvidenceAdjudication,
    EvidenceStatus,
    ExperimentName,
    ExperimentSeed,
    FieldDescription,
    Floor,
    MetricId,
    MultiplicityFamily,
    ProducerModuleName,
    ReportArtifactName,
    ResearchQuestion,
    SimplificationRuleState,
    StableJsonPayload,
    TransferMethod,
)

_MODULE_NAME = ProducerModuleName("fedorbit.experiments.classification")


def _latest_synthesis_manifest(store: ArtifactStore) -> ReusableArtifactManifest | None:
    candidates: list[ReusableArtifactManifest] = []
    for manifest in store.all_manifests():
        if ExperimentName.STATISTICAL_SYNTHESIS.value not in manifest.semantic_producer_coordinates:
            continue
        try:
            resolved = store.resolve(manifest.artifact_id)
        except ValueError:
            continue
        if resolved.state == ArtifactState.COMPLETED:
            candidates.append(resolved)
    if not candidates:
        return None
    return max(candidates, key=lambda manifest: manifest.payload_paths[0])


def _status(
    status: EvidenceStatus,
    materiality: str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    statistical: str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    completeness: str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> EvidenceAdjudication:
    return EvidenceAdjudication(status, materiality, statistical, completeness)


def _not_tested(reason: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                ) -> EvidenceAdjudication:
    return _status(EvidenceStatus.NOT_TESTED, "not evaluated", reason, "incomplete") #TODO: should be enum. Not hardcoded string


def _supported(materiality: str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
               statistical: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
               ) -> EvidenceAdjudication:
    return _status(EvidenceStatus.SUPPORTED, materiality, statistical, "complete") #TODO: should be enum. Not hardcoded string


def _not_supported(materiality: str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                   statistical: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                   ) -> EvidenceAdjudication:
    return _status(EvidenceStatus.NOT_SUPPORTED, materiality, statistical, "complete") #TODO: should be enum. Not hardcoded string


def _partial(materiality: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
             , statistical: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
             ) -> EvidenceAdjudication:
    return _status(EvidenceStatus.PARTIALLY_SUPPORTED, materiality, statistical, "complete") #TODO: should be enum. Not hardcoded string


def _mechanism_only(materiality: str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                    statistical: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                    ) -> EvidenceAdjudication:
    return _status(EvidenceStatus.MECHANISM_ONLY, materiality, statistical, "complete") #TODO: should be enum. Not hardcoded string


def _conditional(materiality: str,  #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                 statistical: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                 ) -> EvidenceAdjudication:
    return _status(EvidenceStatus.CONDITIONAL, materiality, statistical, "complete") #TODO: should be enum. Not hardcoded string


def _null_result(materiality: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                 , statistical: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                 ) -> EvidenceAdjudication:
    return _status(EvidenceStatus.NULL_RESULT, materiality, statistical, "complete") #TODO: should be enum. Not hardcoded string


def _cell_bool_field(cell: Mapping[str, int | float | str | list[int]], #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                     key: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                     ) -> bool:
    return bool(cell.get(key, False))


def _theorem_cells(
    store: ArtifactStore,
) -> tuple[Mapping[str, int | float | str | list[int]], ...]: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    cells: list[Mapping[str, int | float | str | list[int]]] = [] #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
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
) -> tuple[float, ...]: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    from fedorbit.experiments.synthesis import completed_experiment_metric_records

    return tuple(
        float(record.metric_value)
        for record in completed_experiment_metric_records(store, experiment)
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
    holm_maximum: float, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
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
    holm_maximum: float, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> frozenset[str]: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    threshold = active_config().scientific.materiality.realized_relative_macro_ce
    dominant: set[str] = set()
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


def utility_family_status(
    comparisons: tuple[PairedComparisonRecord, ...],
    method_a: TransferMethod,
    method_b: TransferMethod,
    holm_maximum: float, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    bca_floor: Floor, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    required_pairs: int, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    kill_local_reference: bool,
) -> EvidenceAdjudication:
    records = _pair_records(comparisons, method_a, method_b)
    if not records:
        return _not_tested("no contrasts") #TODO: should be enum. Not hardcoded string
    harmful = _harmful_pairs(records)
    if harmful:
        return _not_supported("material harm on a primary pair", "failure rule") #TODO: should be enum. Not hardcoded string
    successful = _successful_pairs(records, holm_maximum, bca_floor)
    if len(successful) >= required_pairs:
        return _supported("material", "holm and BCa satisfied") #TODO: should be enum. Not hardcoded string
    if kill_local_reference and not successful:
        dominant = _local_reference_dominant_pairs(records, holm_maximum)
        if len(dominant) >= required_pairs:
            return _not_supported(
                "local reference sufficient", "equivalence or superiority kill fired"
            ) #TODO: should be enum. Not hardcoded string
    analyzable = len({record.pair for record in records})
    if (
        analyzable == 3
        and len(successful) == 3
        and len(active_config().scientific.datasets.primary_directed_pairs) == 6
    ):
        return _conditional("three eligible pairs", "pre-outcome scope reduction") #TODO: should be enum. Not hardcoded string
    if successful:
        return _partial("subset of pairs material", "full pair threshold unmet") #TODO: should be enum. Not hardcoded string
    return _null_result("no material pair", "no harm") #TODO: should be enum. Not hardcoded string


def _classify_exactness(store: ArtifactStore) -> EvidenceAdjudication:
    cells = _theorem_cells(store)
    if not cells:
        return _not_tested("no theorem cells") #TODO: should be enum. Not hardcoded string
    wrong = sum(1 for cell in cells if not _cell_bool_field(cell, "exact_minima")) #TODO: should be enum, not hardcoded string
    invalid = sum(1 for cell in cells if not _cell_bool_field(cell, "valid_certificate")) #TODO: should be enum, not hardcoded string
    if wrong == 0 and invalid == 0:
        return _supported("separator exact on registered cells", "certificate verified") #TODO: should be enum. Not hardcoded string
    return _not_supported("wrong minima or invalid certificates", "exactness failed") #TODO: should be enum. Not hardcoded string


def _ablation_pair_method_means(
    store: ArtifactStore, method: TransferMethod
) -> Mapping[DirectedPairName, float]: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    from fedorbit.experiments.synthesis import completed_experiment_metric_records

    by_pair: OrderedDict[DirectedPairName, list[float]] = OrderedDict()
    for record in completed_experiment_metric_records(store, ExperimentName.MECHANISM_ABLATIONS):
        if (
            record.method != method
            or record.metric_name != MetricId.RELATIVE_MACRO_CE_GAIN
            or not record.valid
            or record.metric_value is None
        ):
            continue
        by_pair.setdefault(record.pair, []).append(float(record.metric_value))
    means: OrderedDict[DirectedPairName, float] = OrderedDict()
    for pair, values in by_pair.items():
        means[pair] = sum(values) / len(values)
    return means


def _mechanism_retention_pairs(
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> int: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
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


def _material_coupling_pairs(comparisons: tuple[PairedComparisonRecord, ...]) -> int: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
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
        return _not_tested("no coupling gaps") #TODO: should be enum, not hardcoded string
    retention_pairs = _mechanism_retention_pairs(store, comparisons)
    retention_present = retention_pairs >= criteria.primary_pairs_with_material_mean_gap_required
    if retention_present:
        return _not_supported("coupling destruction retains gain", "mechanism attribution fails") #TODO: should be enum, not hardcoded string
    synthetic_fraction = sum(1 for gap in synthetic if gap > material) / len(synthetic)
    accuracy = criteria.theorem_zero_strict_classification_accuracy_required
    synthetic_pass = synthetic_fraction >= accuracy or any(gap > material for gap in synthetic)
    if not synthetic_pass:
        return _not_supported("synthetic mechanism criterion failed", "gap fraction unmet") #TODO: should be enum, not hardcoded string
    real_gaps = _metric_values(
        store,
        ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION,
        MetricId.ROBUST_COUPLING_VALUE_GAP,
    )
    if not real_gaps:
        return _mechanism_only("synthetic mechanism complete", "real-packet criterion unavailable") #TODO: should be enum, not hardcoded string
    real_fraction = sum(1 for gap in real_gaps if gap > material) / len(real_gaps)
    material_pairs = _material_coupling_pairs(comparisons)
    real_pass = (
        real_fraction >= criteria.real_packet_fraction_with_material_gap_minimum
        and material_pairs >= criteria.primary_pairs_with_material_mean_gap_required
    )
    if real_pass:
        return _supported("synthetic and real-packet gaps material", "coupling criteria passed") #TODO: should be enum, not hardcoded string
    return _mechanism_only("synthetic mechanism complete", "real-packet materiality not reached") #TODO: should be enum, not hardcoded string


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
        return _not_tested("no unresolved-map fixtures") #TODO: should be enum, not hardcoded string
    bound_valid = True
    if bounds and values:
        count = min(len(values), len(bounds))
        bound_valid = all(values[index] <= bounds[index] + tolerance for index in range(count))
    if not bound_valid:
        return _not_supported("orbit-radius bound violated", "map-value bound failure") #TODO: should be enum, not hardcoded string
    common_ok = bool(common) and all(abs(value) <= tolerance for value in common)
    robust_ok = bool(robust) and all(abs(value) <= tolerance or value > 0.0 for value in robust)
    if common_ok and robust_ok:
        return _supported("common-action and robust-compromise constructed", "map bound valid") #TODO: should be enum, not hardcoded string
    if common_ok or robust_ok:
        return _partial("one controlled family constructed", "map bound valid") #TODO: should be enum, not hardcoded string
    return _not_supported("exact map recovery required", "neither controlled family holds") #TODO: should be enum, not hardcoded string


def _classify_sparse_operational(
    store: ArtifactStore,
) -> EvidenceAdjudication:
    required = active_config().scientific.evaluation_criteria.sparse_operational_relevance
    useful_floor = active_config().scientific.materiality.useful_transfer_relative_macro_ce_gain
    sparse = _metric_values(
        store, ExperimentName.SPARSITY_AND_DENSE_FALLBACK, MetricId.RELATIVE_MACRO_CE_GAIN
    )
    if not sparse:
        return _not_tested("no sparsity metrics") #TODO: should be enum, not hardcoded string
    kill = _sparse_irrelevance_applied(store)
    if kill:
        return _not_supported("dense dominates sparse supports", "sparse-irrelevance kill") #TODO: should be enum, not hardcoded string
    useful = sum(1 for gain in sparse if gain > useful_floor)
    if useful >= required.primary_pairs_with_useful_gain_required:
        return _supported("sparse support retains useful gain", "pair threshold met") #TODO: should be enum, not hardcoded string
    if useful:
        return _partial("at least one sparse support useful", "full operational rule unmet") #TODO: should be enum, not hardcoded string
    return _null_result("no useful sparse support", "irrelevance kill not fired") #TODO: should be enum, not hardcoded string


def _classify_confirmation(
    comparisons: tuple[PairedComparisonRecord, ...],
) -> EvidenceAdjudication:
    required = active_config().scientific.evaluation_criteria.confirmation_safety
    rows = tuple(
        record for record in comparisons if record.family == MultiplicityFamily.CONFIRMATION_SAFETY
    )
    if not rows:
        return _not_tested("no confirmation contrasts") #TODO: should be enum, not hardcoded string
    worsening = tuple(
        record
        for record in rows
        if record.mean_difference is not None
        and record.mean_difference < -required.pair_harmful_rate_worsening_maximum
    )
    if worsening:
        return _not_supported("harmful-rate worsening", "safety failure") #TODO: should be enum, not hardcoded string
    successful = sum(1 for record in rows if record.decision == ComparisonDecision.SUPERIOR)
    if successful >= required.qualifying_primary_pairs_required:
        return _supported("confirmation reduces harmful rate", "ARR criterion met") #TODO: should be enum, not hardcoded string
    if successful:
        return _partial("subset of pairs meet ARR/RRR", "qualifying-pair threshold unmet") #TODO: should be enum, not hardcoded string
    return _null_result("no pair meets harm reduction", "no worsening") #TODO: should be enum, not hardcoded string


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
        return _not_tested("no work-structure Spearman") #TODO: should be enum, not hardcoded string
    if certificates and any(value < 1.0 for value in certificates):
        return _not_supported("counter or certificate mismatch", "approximation required") #TODO: should be enum, not hardcoded string
    if not values:
        return _partial("certificates match", "runtime-trend evidence missing") #TODO: should be enum, not hardcoded string
    if values[0] > 0.0:
        return _supported("runtime tracks predicted work", "Spearman positive") #TODO: should be enum, not hardcoded string
    return _partial("certificates match", "non-positive Spearman") #TODO: should be enum, not hardcoded string


_EXACT_SPARSE_DEPENDENT_QUESTIONS = frozenset(
    {
        ResearchQuestion.JOINT_CORRESPONDENCE_AVOIDS_RECTANGULAR_PESSIMISM,
        ResearchQuestion.STRICT_CROSS_TELEMETRY_TRANSFER_UTILITY,
        ResearchQuestion.VALUE_OF_EXTERNAL_PROCEDURAL_EVIDENCE,
        ResearchQuestion.OPERATIONAL_RELEVANCE_OF_SPARSE_SUPPORT,
        ResearchQuestion.TARGET_CONFIRMATION_SAFETY,
        ResearchQuestion.SPARSE_SOLVER_WORK_STRUCTURE_AGREEMENT,
    }
)


def _classify_question(
    question: ResearchQuestion,
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> EvidenceAdjudication:
    criteria = active_config().scientific.evaluation_criteria
    if question == ResearchQuestion.EXACT_SPARSE_SEPARATOR_EXACTNESS:
        return _classify_exactness(store)
    if question in _EXACT_SPARSE_DEPENDENT_QUESTIONS:
        exactness = _classify_exactness(store)
        if exactness.status == EvidenceStatus.NOT_SUPPORTED:
            return _not_supported(
                "exact-sparse separator exactness failed", "exactness-failure kill rule" #TODO: should be enum, not hardcoded string
            )
    if question == ResearchQuestion.JOINT_CORRESPONDENCE_AVOIDS_RECTANGULAR_PESSIMISM:
        return _classify_joint_correspondence(store, comparisons)
    if question == ResearchQuestion.ACTION_CERTIFICATION_WITHOUT_FINE_MAP_IDENTIFICATION:
        return _classify_action_certification(store)
    if question == ResearchQuestion.STRICT_CROSS_TELEMETRY_TRANSFER_UTILITY:
        required = criteria.strict_cross_telemetry_utility
        return utility_family_status(
            comparisons,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            TransferMethod.LOCAL_ONLY,
            required.holm_adjusted_p_maximum,
            required.bca_lower_bound_strictly_greater_than,
            required.successful_primary_pairs_required,
            False,
        )
    if question == ResearchQuestion.VALUE_OF_EXTERNAL_PROCEDURAL_EVIDENCE:
        required = criteria.external_source_value_vs_local_sir
        return utility_family_status(
            comparisons,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            TransferMethod.LOCAL_SIR,
            required.holm_adjusted_p_maximum,
            required.bca_lower_bound_strictly_greater_than,
            required.successful_primary_pairs_required,
            True,
        )
    if question == ResearchQuestion.OPERATIONAL_RELEVANCE_OF_SPARSE_SUPPORT:
        return _classify_sparse_operational(store)
    if question == ResearchQuestion.TARGET_CONFIRMATION_SAFETY:
        return _classify_confirmation(comparisons)
    if question == ResearchQuestion.SPARSE_SOLVER_WORK_STRUCTURE_AGREEMENT:
        return _classify_work_structure(store)
    return _not_tested("no dedicated contrast family")


def _rule_payload(rule: str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                  state: SimplificationRuleState, 
                  reason: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                  ) -> StableJsonPayload:
    return cast(StableJsonPayload, OrderedDict(rule=rule, state=state.value, reason=reason))


def _median(values: tuple[float, ...] #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            ) -> float | None: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
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


def _generic_qap_rule(store: ArtifactStore) -> tuple[SimplificationRuleState,
                                                     str]: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    from fedorbit.experiments.synthesis import completed_experiment_metric_records

    records = completed_experiment_metric_records(
        store, ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK
    )
    sparse = tuple(
        float(record.metric_value)
        for record in records
        if record.method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
        and record.metric_name == MetricId.WALL_TIME
        and record.valid
        and record.metric_value is not None
    )
    qap = tuple(
        float(record.metric_value)
        for record in records
        if record.method == TransferMethod.GENERIC_EXACT_QAP
        and record.metric_name == MetricId.WALL_TIME
        and record.valid
        and record.metric_value is not None
    )
    if not sparse or not qap:
        return SimplificationRuleState.NOT_TESTED, "solver runtime ratios not jointly populated" #TODO: should be enum, not hardcoded string
    sparse_median = _median(sparse)
    qap_median = _median(qap)
    if sparse_median is None or qap_median is None or sparse_median <= 0.0:
        return SimplificationRuleState.NOT_TESTED, "non-positive exact-sparse runtime" #TODO: should be enum, not hardcoded string
    rule = active_config().scientific.simplification_rules.generic_qap_dominates
    ratio = qap_median / sparse_median
    if ratio <= rule.median_runtime_ratio_to_exact_sparse_maximum:
        return SimplificationRuleState.APPLIED, "QAP median runtime at or below exact-sparse" #TODO: should be enum, not hardcoded string
    return SimplificationRuleState.NOT_APPLIED, "QAP median runtime exceeds exact-sparse" #TODO: should be enum, not hardcoded string


def _strict_interface_rule(
    comparisons: tuple[PairedComparisonRecord, ...],
) -> tuple[SimplificationRuleState, 
           str]: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    rule = active_config().scientific.simplification_rules.strict_interface_removes_gain
    fedorbit = _pair_records(
        comparisons, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, TransferMethod.LOCAL_ONLY
    )
    oracle = _pair_records(comparisons, TransferMethod.EXACT_MAP_ORACLE, TransferMethod.LOCAL_ONLY)
    if not fedorbit or not oracle:
        return (
            SimplificationRuleState.NOT_TESTED,
            "FedORBIT and oracle contrasts not jointly populated", #TODO: should be enum, not hardcoded string
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
        return (
            SimplificationRuleState.APPLIED,
            "strict interface removes gain while oracle succeeds", #TODO: should be enum, not hardcoded string
        )
    return SimplificationRuleState.NOT_APPLIED, "strict-interface majority not reached" #TODO: should be enum, not hardcoded string


def _source_response_rule(store: ArtifactStore) -> tuple[SimplificationRuleState, str]:
    failures = _metric_values(
        store,
        ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION,
        MetricId.RESOURCE_LIMIT_INDICATOR,
    )
    if not failures:
        return (
            SimplificationRuleState.NOT_TESTED,
            "principal source-packet failure fraction not persisted", #TODO: should be enum, not hardcoded string
        )
    rule = active_config().scientific.simplification_rules.source_response_is_too_unstable
    fraction = sum(1 for value in failures if value > 0.0) / len(failures)
    if fraction > rule.principal_source_packet_failure_fraction_strictly_greater_than:
        return SimplificationRuleState.APPLIED, "source-packet failure fraction exceeds threshold" #TODO: should be enum, not hardcoded string
    return SimplificationRuleState.NOT_APPLIED, "source-packet failure fraction below threshold" #TODO: should be enum, not hardcoded string


def _simplification_rule_states(
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
        rectangular = (
            SimplificationRuleState.APPLIED
            if below >= minimum
            else SimplificationRuleState.NOT_APPLIED
        )
        rectangular_reason = "real-packet coupling below materiality" #TODO: should be enum, not hardcoded string
    else:
        rectangular = SimplificationRuleState.NOT_TESTED
        rectangular_reason = "no real-packet coupling gaps" #TODO: should be enum, not hardcoded string
    sparse_gains = _metric_values(
        store, ExperimentName.SPARSITY_AND_DENSE_FALLBACK, MetricId.RELATIVE_MACRO_CE_GAIN
    )
    if sparse_gains:
        sparse_state = (
            SimplificationRuleState.APPLIED
            if _sparse_irrelevance_applied(store)
            else SimplificationRuleState.NOT_APPLIED
        )
        sparse_reason = "sparse useful-unit fraction" #TODO: should be enum, not hardcoded string
    else:
        sparse_state = SimplificationRuleState.NOT_TESTED
        sparse_reason = "no sparsity metrics" #TODO: should be enum, not hardcoded string
    point_rows = _pair_records(
        comparisons,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
    )
    if point_rows:
        advantage = sum(
            1
            for record in point_rows
            if record.mean_difference is not None
            and record.mean_difference
            >= rules.point_matching_is_sufficient.utility_advantage_over_fedorbit_minimum
        )
        point_state = (
            SimplificationRuleState.APPLIED if advantage else SimplificationRuleState.NOT_APPLIED
        )
        point_reason = "point-matching vs exact-sparse contrast" #TODO: should be enum, not hardcoded string
    else:
        point_state = SimplificationRuleState.NOT_TESTED
        point_reason = "no point-matching contrasts" #TODO: should be enum, not hardcoded string
    qap_state, qap_reason = _generic_qap_rule(store)
    interface_state, interface_reason = _strict_interface_rule(comparisons)
    source_state, source_reason = _source_response_rule(store)
    return (
        _rule_payload("rectangularization_is_sufficient", rectangular, rectangular_reason), #TODO: should be enum, not hardcoded string
        _rule_payload("generic_qap_dominates", qap_state, qap_reason), #TODO: should be enum, not hardcoded string
        _rule_payload("sparse_support_is_operationally_irrelevant", sparse_state, sparse_reason), #TODO: should be enum, not hardcoded string
        _rule_payload("point_matching_is_sufficient", point_state, point_reason), #TODO: should be enum, not hardcoded string
        _rule_payload("strict_interface_removes_gain", interface_state, interface_reason), #TODO: should be enum, not hardcoded string
        _rule_payload("source_response_is_too_unstable", source_state, source_reason), #TODO: should be enum, not hardcoded string
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
        "evidence_classification_start",
        synthesis_present=synthesis is not None,
        comparison_rows=len(comparisons),
    )
    rows: list[StableJsonPayload] = []
    for question in ResearchQuestion:
        adjudication = _classify_question(question, store, comparisons)
        status = adjudication.status
        materiality = adjudication.materiality
        statistical = adjudication.statistical
        completeness = adjudication.completeness
        rows.append(
            cast(
                StableJsonPayload,
                OrderedDict(
                    question=question.value,
                    final_state=status.value,
                    materiality_result=materiality,
                    statistical_result=statistical,
                    evidence_completeness=completeness,
                    scope="registered primary evidence", #TODO: should be enum, not hardcoded strings
                    supporting_table="evidence-status", #TODO: should be enum, not hardcoded strings
                    supporting_figure="real-transfer-gain-forest-plot", #TODO: should be enum, not hardcoded strings
                    forbidden_wording="",
                ),
            )
        )
    simplification = _simplification_rule_states(store, comparisons)
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
        _MODULE_NAME,
        "evidence-classification", #TODO: should be enum, not hardcoded strings
    )


class EvidenceStatusRow(DomainModel):
    question: ResearchQuestion
    final_state: EvidenceStatus
    materiality_result: FieldDescription
    statistical_result: FieldDescription
    evidence_completeness: FieldDescription
    scope: FieldDescription
    supporting_table: ReportArtifactName
    supporting_figure: ReportArtifactName
    forbidden_wording: FieldDescription


def completed_evidence_status_rows(store: ArtifactStore) -> tuple[EvidenceStatusRow, ...]:
    candidates: list[ReusableArtifactManifest] = []
    for item in store.all_manifests():
        if ExperimentName.EVIDENCE_CLASSIFICATION.value not in item.semantic_producer_coordinates:
            continue
        try:
            resolved = store.resolve(item.artifact_id)
        except ValueError:
            continue
        if resolved.state == ArtifactState.COMPLETED:
            candidates.append(resolved)
    if not candidates:
        return ()
    latest = max(candidates, key=lambda item: item.payload_paths[0])
    payload = json.loads(Path(latest.payload_paths[0]).read_text(encoding="utf-8"))
    rows = payload.get("statuses", ())
    return tuple(EvidenceStatusRow.model_validate(row) for row in rows if isinstance(row, dict))
