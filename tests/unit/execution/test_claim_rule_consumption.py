from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

import fedorbit.experiments.classification as classification
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


def _metric_record(method: TransferMethod, value: Estimate, seed: int = 1) -> MetricRecord:
    return MetricRecord(
        experiment=ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        pair=DirectedPairName(_PRIMARY),
        method=method,
        condition=PRINCIPAL_EVALUATION_CONDITION.name,
        seed=seed,
        metric_name=MetricId.MACRO_CROSS_ENTROPY,
        metric_value=value,
        metric_unit=MetricUnit.SCORE,
        direction=MetricDirection.LOWER_IS_BETTER,
        evaluation_class_set_sha256=Sha256Digest("e" * 64),
        input_artifact_ids=(ArtifactIdentifier("score-artifact"),),
        dependency_fingerprint_sha256=Sha256Digest("d" * 64),
        valid=True,
        invalid_reason=None,
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
