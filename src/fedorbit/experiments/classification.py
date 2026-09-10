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
    DomainModel,
    EvidenceStatus,
    ExperimentName,
    ExperimentSeed,
    FieldDescription,
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


def _pair_count(
    comparisons: tuple[PairedComparisonRecord, ...],
    method_a: TransferMethod,
    method_b: TransferMethod,
    decision: ComparisonDecision,
    holm_maximum: float,
    bca_floor: float,
) -> int:
    count = 0
    for record in comparisons:
        if record.method_a != method_a or record.method_b != method_b:
            continue
        if record.decision != decision:
            continue
        if record.holm_p is None or record.holm_p > holm_maximum:
            continue
        if record.bca_ci_low is None or record.bca_ci_low <= bca_floor:
            continue
        count += 1
    return count


def _not_tested(reason: str) -> tuple[EvidenceStatus, str, str, str]:
    return (EvidenceStatus.NOT_TESTED, "not evaluated", reason, "incomplete")


def _supported(materiality: str, statistical: str) -> tuple[EvidenceStatus, str, str, str]:
    return (EvidenceStatus.SUPPORTED, materiality, statistical, "complete")


def _not_supported(materiality: str, statistical: str) -> tuple[EvidenceStatus, str, str, str]:
    return (EvidenceStatus.NOT_SUPPORTED, materiality, statistical, "complete")


def _count_field(cell: Mapping[str, int | float | str | list[int]], key: str) -> int:
    value = cell.get(key, 1)
    return int(value) if isinstance(value, int | float) else 1


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
        for nested in payload.get("cells", ()):
            if isinstance(nested, dict):
                cells.append(cast(Mapping[str, int | float | str | list[int]], nested))
    return tuple(cells)


def _metric_values(
    store: ArtifactStore, experiment: ExperimentName, metric_name: MetricId
) -> tuple[float, ...]:
    from fedorbit.experiments.synthesis import completed_experiment_metric_records

    return tuple(
        float(record.metric_value)
        for record in completed_experiment_metric_records(store, experiment)
        if record.metric_name == metric_name and record.valid and record.metric_value is not None
    )


def _pair_contrast_status(
    comparisons: tuple[PairedComparisonRecord, ...],
    method_a: TransferMethod,
    method_b: TransferMethod,
    holm_maximum: float,
    bca_floor: float,
    required_pairs: int,
) -> tuple[EvidenceStatus, str, str, str]:
    successful = _pair_count(
        comparisons,
        method_a,
        method_b,
        ComparisonDecision.SUPERIOR,
        holm_maximum,
        bca_floor,
    )
    if successful >= required_pairs:
        return _supported("material", "holm and BCa satisfied")
    relevant = tuple(
        record
        for record in comparisons
        if record.method_a == method_a and record.method_b == method_b
    )
    if relevant:
        return _not_supported("below pair threshold", "available contrasts insufficient")
    return _not_tested("no contrasts")


def _classify_question(
    question: ResearchQuestion,
    store: ArtifactStore,
    comparisons: tuple[PairedComparisonRecord, ...],
) -> tuple[EvidenceStatus, str, str, str]:
    criteria = active_config().scientific.evaluation_criteria
    if question == ResearchQuestion.EXACT_SPARSE_SEPARATOR_EXACTNESS:
        cells = _theorem_cells(store)
        if not cells:
            return _not_tested("no theorem cells")
        wrong = sum(_count_field(cell, "wrong_minima_count") for cell in cells)
        invalid = sum(_count_field(cell, "invalid_certificate_count") for cell in cells)
        if wrong == 0 and invalid == 0:
            return _supported("separator exact on registered cells", "certificate verified")
        return _not_supported("wrong minima or invalid certificates", "exactness failed")
    if question == ResearchQuestion.JOINT_CORRESPONDENCE_AVOIDS_RECTANGULAR_PESSIMISM:
        gaps = _metric_values(
            store,
            ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION,
            MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP,
        )
        if not gaps:
            return _not_tested("no coupling gaps")
        material = active_config().scientific.materiality.coupling_objective_units
        fraction = sum(1 for gap in gaps if gap > material) / len(gaps)
        if fraction >= criteria.coupling_mechanism.real_packet_fraction_with_material_gap_minimum:
            return _supported("material rectangularization gap", "gap fraction met")
        return _not_supported("gap fraction below criterion", "rectangularization not material")
    if question == ResearchQuestion.ACTION_CERTIFICATION_WITHOUT_FINE_MAP_IDENTIFICATION:
        values = _metric_values(
            store,
            ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP,
            MetricId.EXACT_MAP_ACTION_VALUE,
        )
        if not values:
            return _not_tested("no unresolved-map fixtures")
        tolerance = active_config().scientific.materiality.coupling_objective_units
        if all(abs(value) <= tolerance for value in values):
            return _supported("common action independent of fine map", "map value near zero")
        return _not_supported("nonzero exact-map action value", "fine-map dependence remains")
    if question == ResearchQuestion.STRICT_CROSS_TELEMETRY_TRANSFER_UTILITY:
        required = criteria.strict_cross_telemetry_utility
        return _pair_contrast_status(
            comparisons,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            TransferMethod.LOCAL_ONLY,
            required.holm_adjusted_p_maximum,
            required.bca_lower_bound_strictly_greater_than,
            required.successful_primary_pairs_required,
        )
    if question == ResearchQuestion.VALUE_OF_EXTERNAL_PROCEDURAL_EVIDENCE:
        required = criteria.external_source_value_vs_local_sir
        return _pair_contrast_status(
            comparisons,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            TransferMethod.LOCAL_SIR,
            required.holm_adjusted_p_maximum,
            required.bca_lower_bound_strictly_greater_than,
            required.successful_primary_pairs_required,
        )
    if question == ResearchQuestion.OPERATIONAL_RELEVANCE_OF_SPARSE_SUPPORT:
        required = criteria.sparse_operational_relevance
        sparse = _metric_values(
            store, ExperimentName.SPARSITY_AND_DENSE_FALLBACK, MetricId.RELATIVE_MACRO_CE_GAIN
        )
        if not sparse:
            return _not_tested("no sparsity metrics")
        useful = sum(
            1
            for gain in sparse
            if gain > active_config().scientific.materiality.useful_transfer_relative_macro_ce_gain
        )
        if useful >= required.primary_pairs_with_useful_gain_required:
            return _supported("sparse support retains useful gain", "pair threshold met")
        return _not_supported("useful sparse units below criterion", "operational relevance unmet")
    if question == ResearchQuestion.TARGET_CONFIRMATION_SAFETY:
        required = criteria.confirmation_safety
        confirmation_rows = tuple(
            record
            for record in comparisons
            if record.family == MultiplicityFamily.CONFIRMATION_SAFETY
        )
        successful = sum(
            1 for record in confirmation_rows if record.decision == ComparisonDecision.SUPERIOR
        )
        if successful >= required.qualifying_primary_pairs_required:
            return _supported("confirmation reduces harmful rate", "ARR criterion met")
        if confirmation_rows:
            return _not_supported("qualifying confirmation pairs below criterion", "safety unmet")
        return _not_tested("no confirmation contrasts")
    if question == ResearchQuestion.SPARSE_SOLVER_WORK_STRUCTURE_AGREEMENT:
        values = _metric_values(
            store, ExperimentName.SCALABILITY_AND_EFFICIENCY, MetricId.WORK_STRUCTURE_SPEARMAN
        )
        if not values:
            return _not_tested("no work-structure Spearman")
        if values[0] > 0.0:
            return _supported("runtime tracks predicted work", "Spearman positive")
        return _not_supported("non-positive Spearman", "work-structure disagreement")
    return _not_tested("no dedicated contrast family")


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
        rectangular_reason = "real-packet coupling below materiality"
    else:
        rectangular = SimplificationRuleState.NOT_TESTED
        rectangular_reason = "no real-packet coupling gaps"
    sparse_gains = _metric_values(
        store, ExperimentName.SPARSITY_AND_DENSE_FALLBACK, MetricId.RELATIVE_MACRO_CE_GAIN
    )
    if sparse_gains:
        useful = sum(
            1 for gain in sparse_gains if gain > material.useful_transfer_relative_macro_ce_gain
        ) / len(sparse_gains)
        sparse_state = (
            SimplificationRuleState.APPLIED
            if useful
            < rules.sparse_support_is_operationally_irrelevant.valid_primary_unit_fraction_minimum
            else SimplificationRuleState.NOT_APPLIED
        )
        sparse_reason = "sparse useful-unit fraction"
    else:
        sparse_state = SimplificationRuleState.NOT_TESTED
        sparse_reason = "no sparsity metrics"
    point_rows = tuple(
        record
        for record in comparisons
        if record.method_a == TransferMethod.POINT_CORRESPONDENCE_COMMITMENT
        and record.method_b == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
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
        point_reason = "point-matching vs exact-sparse contrast"
    else:
        point_state = SimplificationRuleState.NOT_TESTED
        point_reason = "no point-matching contrasts"
    return (
        cast(
            StableJsonPayload,
            OrderedDict(
                rule="rectangularization_is_sufficient",
                state=rectangular.value,
                reason=rectangular_reason,
            ),
        ),
        cast(
            StableJsonPayload,
            OrderedDict(
                rule="generic_qap_dominates",
                state=SimplificationRuleState.NOT_TESTED.value,
                reason="solver runtime ratios not jointly populated",
            ),
        ),
        cast(
            StableJsonPayload,
            OrderedDict(
                rule="sparse_support_is_operationally_irrelevant",
                state=sparse_state.value,
                reason=sparse_reason,
            ),
        ),
        cast(
            StableJsonPayload,
            OrderedDict(
                rule="point_matching_is_sufficient",
                state=point_state.value,
                reason=point_reason,
            ),
        ),
        cast(
            StableJsonPayload,
            OrderedDict(
                rule="strict_interface_removes_gain",
                state=SimplificationRuleState.NOT_TESTED.value,
                reason="point-gain BCa family not jointly populated",
            ),
        ),
        cast(
            StableJsonPayload,
            OrderedDict(
                rule="source_response_is_too_unstable",
                state=SimplificationRuleState.NOT_TESTED.value,
                reason="principal source-packet failure fraction not persisted",
            ),
        ),
    )


def execute_evidence_classification(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> ReusableArtifactManifest:
    from fedorbit.experiments.synthesis import completed_primary_transfer_comparison_records
    from fedorbit.experiments.validation import _persist_synthetic_experiment_payload

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
        status, materiality, statistical, completeness = _classify_question(
            question, store, comparisons
        )
        rows.append(
            cast(
                StableJsonPayload,
                OrderedDict(
                    question=question.value,
                    final_state=status.value,
                    materiality_result=materiality,
                    statistical_result=statistical,
                    evidence_completeness=completeness,
                    scope="registered primary evidence",
                    supporting_table="evidence-status",
                    supporting_figure="real-transfer-gain-forest-plot",
                    forbidden_wording="",
                ),
            )
        )
    simplification = _simplification_rule_states(store, comparisons)
    seed = ExperimentSeed(0)
    return _persist_synthetic_experiment_payload(
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
        "evidence-classification",
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
