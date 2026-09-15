from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

import fedorbit.experiments.classification as classification
import fedorbit.experiments.synthesis as synthesis
from fedorbit.analysis.records import (
    ComparisonDecision,
    MetricDirection,
    MetricRecord,
    PairedComparisonRecord,
)
from fedorbit.config.loading import active_config
from fedorbit.experiments.classification import (
    action_certification_evidence,
    external_procedural_evidence_value,
    simplification_rule_states,
)
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.types import (
    PRINCIPAL_EVALUATION_CONDITION,
    ArtifactIdentifier,
    ContrastName,
    DirectedPairName,
    Estimate,
    EvidenceStatus,
    ExperimentName,
    MetricId,
    MetricUnit,
    MultiplicityFamily,
    Sha256Digest,
    SimplificationRuleName,
    SimplificationRuleState,
    TransferMethod,
)

_PRIMARY = "ton_iot_windows10_host -> ton_iot_linux_process_host"


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "outputs", tmp_path / "outputs" / "cache" / "staging")


def _rules(
    tmp_path: Path,
    comparisons: tuple[PairedComparisonRecord, ...] = (),
) -> dict[SimplificationRuleName, SimplificationRuleState]:
    store = _store(tmp_path)
    return {
        SimplificationRuleName(str(record["rule"])): SimplificationRuleState(str(record["state"]))
        for payload in simplification_rule_states(store, comparisons)
        for record in (cast(Mapping[str, object], payload),)
    }


def _point_record(pair: str, mean_difference: float) -> PairedComparisonRecord:
    return PairedComparisonRecord(
        contrast_name=ContrastName("fedorbit-vs-point-correspondence"),
        family=MultiplicityFamily.POINT_CORRESPONDENCE_SAFETY,
        pair=DirectedPairName(pair),
        method_a=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        method_b=TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
        metric=MetricId.MACRO_CROSS_ENTROPY,
        paired_seed_count=8,
        mean_difference=mean_difference,
        median_difference=mean_difference,
        bca_ci_low=mean_difference,
        bca_ci_high=mean_difference + 0.01,
        raw_p=0.01,
        holm_p=0.02,
        materiality_threshold=0.01,
        equivalence_margin_low=None,
        equivalence_margin_high=None,
        input_metric_artifact_ids=(ArtifactIdentifier("metric"),),
        dependency_fingerprint_sha256=Sha256Digest("b" * 64),
        decision=ComparisonDecision.NOT_SUPPORTED,
    )


def _metric(
    experiment: ExperimentName,
    pair: str,
    method: TransferMethod,
    metric_name: MetricId,
    value: Estimate,
    seed: int = 1,
    unit: MetricUnit = MetricUnit.SCORE,
    direction: MetricDirection = MetricDirection.LOWER_IS_BETTER,
) -> MetricRecord:
    return MetricRecord(
        experiment=experiment,
        pair=DirectedPairName(pair),
        method=method,
        condition=PRINCIPAL_EVALUATION_CONDITION.name,
        seed=seed,
        metric_name=metric_name,
        metric_value=value,
        metric_unit=unit,
        direction=direction,
        evaluation_class_set_sha256=Sha256Digest("e" * 64),
        input_artifact_ids=(ArtifactIdentifier("score-artifact"),),
        dependency_fingerprint_sha256=Sha256Digest("d" * 64),
        valid=True,
        invalid_reason=None,
    )


def _metric_record(method: TransferMethod, value: Estimate, seed: int = 1) -> MetricRecord:
    return _metric(
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        _PRIMARY,
        method,
        MetricId.MACRO_CROSS_ENTROPY,
        value,
        seed,
    )


def _macro_ce_records(fedorbit: float, commitment: float) -> tuple[MetricRecord, ...]:
    seeds = active_config().scientific.randomness.confirmatory_seeds
    return tuple(
        record
        for seed in seeds
        for record in (
            _metric_record(TransferMethod.LOCAL_ONLY, 4.0, seed),
            _metric_record(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, fedorbit, seed),
            _metric_record(TransferMethod.POINT_CORRESPONDENCE_COMMITMENT, commitment, seed),
        )
    )


def test_point_matching_rule_fires_when_every_pair_is_not_worse_and_more_useful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _macro_ce_records(3.8, 3.6)

    def metric_records(
        _store: ArtifactStore, _experiment: ExperimentName
    ) -> tuple[MetricRecord, ...]:
        return records

    monkeypatch.setattr(classification, "_completed_experiment_metric_records", metric_records)
    states = _rules(tmp_path, (_point_record(_PRIMARY, 0.05),))
    assert (
        states[SimplificationRuleName.POINT_MATCHING_IS_SUFFICIENT]
        is SimplificationRuleState.APPLIED
    )


def test_point_matching_rule_stays_not_applied_without_the_registered_advantage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _macro_ce_records(3.8, 3.79)

    def metric_records(
        _store: ArtifactStore, _experiment: ExperimentName
    ) -> tuple[MetricRecord, ...]:
        return records

    monkeypatch.setattr(classification, "_completed_experiment_metric_records", metric_records)
    states = _rules(tmp_path, (_point_record(_PRIMARY, 0.05),))
    assert (
        states[SimplificationRuleName.POINT_MATCHING_IS_SUFFICIENT]
        is SimplificationRuleState.NOT_APPLIED
    )


def test_exactness_failure_rule_reads_persisted_objective_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def metric_values(
        _store: ArtifactStore, experiment: ExperimentName, metric_name: MetricId
    ) -> tuple[Estimate, ...]:
        if experiment is not ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION:
            return ()
        if metric_name is MetricId.RELATIVE_OBJECTIVE_ERROR:
            return (0.5,)
        if metric_name is MetricId.CORRESPONDENCE_CERTIFICATE_VALIDITY:
            return (1.0,)
        return ()

    monkeypatch.setattr(classification, "_metric_values", metric_values)
    states = _rules(tmp_path)
    assert states[SimplificationRuleName.EXACTNESS_FAILURE] is SimplificationRuleState.APPLIED


def test_rectangularization_rule_fires_when_gaps_are_mostly_below_materiality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    materiality = active_config().scientific.materiality.coupling_objective_units

    def metric_values(
        _store: ArtifactStore, experiment: ExperimentName, metric_name: MetricId
    ) -> tuple[Estimate, ...]:
        if (
            experiment is ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION
            and metric_name is MetricId.ROBUST_COUPLING_VALUE_GAP
        ):
            return (materiality / 2.0,) * 9 + (materiality * 2.0,)
        return ()

    monkeypatch.setattr(classification, "_metric_values", metric_values)
    states = _rules(tmp_path)
    assert (
        states[SimplificationRuleName.RECTANGULARIZATION_IS_SUFFICIENT]
        is SimplificationRuleState.APPLIED
    )


def test_theory_classification_failure_rule_fires_on_negative_coupling_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def metric_values(
        _store: ArtifactStore, experiment: ExperimentName, metric_name: MetricId
    ) -> tuple[Estimate, ...]:
        if (
            experiment is ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION
            and metric_name is MetricId.ROBUST_COUPLING_VALUE_GAP
        ):
            return (-1.0,)
        return ()

    monkeypatch.setattr(classification, "_metric_values", metric_values)
    states = _rules(tmp_path)
    assert (
        states[SimplificationRuleName.THEORY_CLASSIFICATION_FAILURE]
        is SimplificationRuleState.APPLIED
    )


def test_sparse_support_irrelevance_rule_fires_below_useful_fraction_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    useful_floor = active_config().scientific.materiality.useful_transfer_relative_macro_ce_gain

    def metric_values(
        _store: ArtifactStore, experiment: ExperimentName, metric_name: MetricId
    ) -> tuple[Estimate, ...]:
        if (
            experiment is ExperimentName.SPARSITY_AND_DENSE_FALLBACK
            and metric_name is MetricId.RELATIVE_MACRO_CE_GAIN
        ):
            return (useful_floor - 0.01,) * 10
        return ()

    monkeypatch.setattr(classification, "_metric_values", metric_values)
    states = _rules(tmp_path)
    assert (
        states[SimplificationRuleName.SPARSE_SUPPORT_IS_OPERATIONALLY_IRRELEVANT]
        is SimplificationRuleState.APPLIED
    )


def test_local_sir_sufficiency_rule_removes_the_external_source_claim(tmp_path: Path) -> None:
    pairs = active_config().scientific.datasets.primary_directed_pairs
    comparisons = tuple(
        PairedComparisonRecord(
            contrast_name=ContrastName("fedorbit-vs-local-sir"),
            family=MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR,
            pair=spec.direction,
            method_a=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            method_b=TransferMethod.LOCAL_SIR,
            metric=MetricId.RELATIVE_MACRO_CE_GAIN,
            paired_seed_count=8,
            mean_difference=-0.05,
            median_difference=-0.05,
            bca_ci_low=-0.09,
            bca_ci_high=-0.01,
            raw_p=0.01,
            holm_p=0.02,
            materiality_threshold=0.01,
            equivalence_margin_low=None,
            equivalence_margin_high=None,
            input_metric_artifact_ids=(ArtifactIdentifier("metric"),),
            dependency_fingerprint_sha256=Sha256Digest("c" * 64),
            decision=ComparisonDecision.EQUIVALENT,
        )
        for spec in pairs
    )
    states = _rules(tmp_path, comparisons)
    assert states[SimplificationRuleName.LOCAL_SIR_IS_SUFFICIENT] is SimplificationRuleState.APPLIED
    adjudication = external_procedural_evidence_value(_store(tmp_path), comparisons)
    assert adjudication.status is EvidenceStatus.NOT_SUPPORTED


def test_unresolved_map_rule_marks_action_certification_conditional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def metric_values(
        _store: ArtifactStore, experiment: ExperimentName, metric_name: MetricId
    ) -> tuple[Estimate, ...]:
        if (
            experiment is ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT
            and metric_name is MetricId.TRIVIAL_MAP_RECONSTRUCTION_INDICATOR
        ):
            return (1.0, 1.0)
        return ()

    monkeypatch.setattr(classification, "_metric_values", metric_values)
    states = _rules(tmp_path)
    assert (
        states[SimplificationRuleName.UNRESOLVED_MAP_REGIME_LACKS_PRACTICAL_MOTIVATION]
        is SimplificationRuleState.APPLIED
    )
    adjudication = action_certification_evidence(_store(tmp_path), ())
    assert adjudication.status is EvidenceStatus.CONDITIONAL


def test_generic_qap_rule_fires_when_qap_runtime_at_or_below_sparse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threshold = (
        active_config().scientific.simplification_rules.generic_qap_dominates
    ).median_runtime_ratio_to_exact_sparse_maximum
    records = tuple(
        _metric(
            ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK,
            _PRIMARY,
            method,
            MetricId.WALL_TIME,
            value,
            seed,
        )
        for seed in (1, 2, 3)
        for method, value in (
            (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, 10.0),
            (TransferMethod.GENERIC_EXACT_QAP, 10.0 * threshold),
        )
    )

    def metric_records(
        _store: ArtifactStore, experiment: ExperimentName
    ) -> tuple[MetricRecord, ...]:
        return records if experiment is ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK else ()

    monkeypatch.setattr(synthesis, "completed_experiment_metric_records", metric_records)
    states = _rules(tmp_path)
    assert states[SimplificationRuleName.GENERIC_QAP_DOMINATES] is SimplificationRuleState.APPLIED


def test_strict_interface_rule_fires_when_solver_flat_and_oracle_succeeds(
    tmp_path: Path,
) -> None:
    rule = active_config().scientific.simplification_rules.strict_interface_removes_gain
    pairs = [spec.direction for spec in active_config().scientific.datasets.primary_directed_pairs]
    comparisons = tuple(
        PairedComparisonRecord(
            contrast_name=ContrastName(f"fedorbit-vs-local-only-{index}"),
            family=MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
            pair=DirectedPairName(pair),
            method_a=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            method_b=TransferMethod.LOCAL_ONLY,
            metric=MetricId.MACRO_CROSS_ENTROPY,
            paired_seed_count=8,
            mean_difference=rule.point_gain_maximum,
            median_difference=rule.point_gain_maximum,
            bca_ci_low=rule.point_gain_maximum,
            bca_ci_high=rule.bca_upper_bound_maximum / 2.0,
            raw_p=0.5,
            holm_p=0.5,
            materiality_threshold=0.01,
            equivalence_margin_low=None,
            equivalence_margin_high=None,
            input_metric_artifact_ids=(ArtifactIdentifier("metric"),),
            dependency_fingerprint_sha256=Sha256Digest("f" * 64),
            decision=ComparisonDecision.EQUIVALENT,
        )
        for index, pair in enumerate(pairs[: rule.primary_pair_majority_required])
    ) + tuple(
        PairedComparisonRecord(
            contrast_name=ContrastName(f"oracle-vs-local-only-{index}"),
            family=MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
            pair=DirectedPairName(pair),
            method_a=TransferMethod.EXACT_MAP_ORACLE,
            method_b=TransferMethod.LOCAL_ONLY,
            metric=MetricId.MACRO_CROSS_ENTROPY,
            paired_seed_count=8,
            mean_difference=-0.2,
            median_difference=-0.2,
            bca_ci_low=-0.25,
            bca_ci_high=-0.15,
            raw_p=0.01,
            holm_p=0.02,
            materiality_threshold=0.01,
            equivalence_margin_low=None,
            equivalence_margin_high=None,
            input_metric_artifact_ids=(ArtifactIdentifier("metric"),),
            dependency_fingerprint_sha256=Sha256Digest("a" * 64),
            decision=ComparisonDecision.SUPERIOR,
        )
        for index, pair in enumerate(pairs[: rule.primary_pair_majority_required])
    )
    states = _rules(tmp_path, comparisons)
    assert (
        states[SimplificationRuleName.STRICT_INTERFACE_REMOVES_GAIN]
        is SimplificationRuleState.APPLIED
    )


def test_coupling_destruction_rule_fires_when_destroyed_mechanism_retains_gain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    criteria = active_config().scientific.evaluation_criteria.coupling_mechanism
    pairs = [spec.direction for spec in active_config().scientific.datasets.primary_directed_pairs][
        : criteria.primary_pairs_with_material_mean_gap_required
    ]
    full_gain = 0.2
    destroyed_gain = full_gain * criteria.destruction_positive_gain_retention_minimum
    records = tuple(
        _metric(
            ExperimentName.MECHANISM_ABLATIONS,
            pair,
            method,
            MetricId.RELATIVE_MACRO_CE_GAIN,
            value,
            seed,
        )
        for pair in pairs
        for seed in (1, 2)
        for method, value in (
            (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, full_gain),
            (TransferMethod.COUPLING_DESTROYED_FEDORBIT, destroyed_gain),
        )
    )

    def metric_records(
        _store: ArtifactStore, experiment: ExperimentName
    ) -> tuple[MetricRecord, ...]:
        return records if experiment is ExperimentName.MECHANISM_ABLATIONS else ()

    monkeypatch.setattr(synthesis, "completed_experiment_metric_records", metric_records)
    comparisons = tuple(
        PairedComparisonRecord(
            contrast_name=ContrastName(f"full-vs-destroyed-{index}"),
            family=MultiplicityFamily.MECHANISM_ABLATIONS,
            pair=DirectedPairName(pair),
            method_a=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            method_b=TransferMethod.COUPLING_DESTROYED_FEDORBIT,
            metric=MetricId.RELATIVE_MACRO_CE_GAIN,
            paired_seed_count=8,
            mean_difference=0.0,
            median_difference=0.0,
            bca_ci_low=-0.01,
            bca_ci_high=0.01,
            raw_p=0.9,
            holm_p=0.9,
            materiality_threshold=0.01,
            equivalence_margin_low=-0.02,
            equivalence_margin_high=0.02,
            input_metric_artifact_ids=(ArtifactIdentifier("metric"),),
            dependency_fingerprint_sha256=Sha256Digest("b" * 64),
            decision=ComparisonDecision.EQUIVALENT,
        )
        for index, pair in enumerate(pairs)
    )
    states = _rules(tmp_path, comparisons)
    assert (
        states[SimplificationRuleName.COUPLING_DESTRUCTION_RETAINS_GAIN]
        is SimplificationRuleState.APPLIED
    )


def test_confirmation_safety_rule_fires_below_qualifying_pair_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    criteria = active_config().scientific.evaluation_criteria.confirmation_safety

    def metric_values(
        _store: ArtifactStore, experiment: ExperimentName, metric_name: MetricId
    ) -> tuple[Estimate, ...]:
        if experiment is not ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY:
            return ()
        if metric_name is MetricId.ABSOLUTE_RISK_REDUCTION:
            return (criteria.absolute_risk_reduction_minimum / 2.0,) * 6
        if metric_name is MetricId.COVERAGE_LOSS:
            return (criteria.pair_coverage_loss_maximum / 2.0,) * 6
        return ()

    monkeypatch.setattr(classification, "_metric_values", metric_values)
    states = _rules(tmp_path)
    assert (
        states[SimplificationRuleName.CONFIRMATION_HAS_NO_SAFETY_VALUE]
        is SimplificationRuleState.APPLIED
    )


def test_source_response_rule_fires_above_principal_failure_fraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rule = active_config().scientific.simplification_rules.source_response_is_too_unstable
    failing = int(rule.principal_source_packet_failure_fraction_strictly_greater_than * 10) + 1

    def diagnostic_values(
        _store: ArtifactStore, experiment: ExperimentName, metric_name: MetricId
    ) -> tuple[Estimate, ...]:
        if (
            experiment is not ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION
            or metric_name is not MetricId.RESOURCE_LIMIT_INDICATOR
        ):
            return ()
        return (1.0,) * failing + (0.0,) * (10 - failing)

    monkeypatch.setattr(classification, "_diagnostic_metric_values", diagnostic_values)
    states = _rules(tmp_path)
    assert (
        states[SimplificationRuleName.SOURCE_RESPONSE_IS_TOO_UNSTABLE]
        is SimplificationRuleState.APPLIED
    )
