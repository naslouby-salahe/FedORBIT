from __future__ import annotations

import json
from pathlib import Path

from fedorbit.analysis.records import ComparisonDecision, PairedComparisonRecord
from fedorbit.config.loading import active_config
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.classification import (
    execute_evidence_classification,
    utility_family_status,
)
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactState,
    ContrastName,
    DirectedPairName,
    EvidenceStatus,
    ExperimentName,
    MetricId,
    MultiplicityFamily,
    OverwritePolicy,
    Sha256Digest,
    TransferMethod,
)


def test_evidence_classification_records_not_tested_without_synthesis(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    catalogue = build_catalogue()
    request = ExperimentExecutionRequest(
        experiment=ExperimentName.EVIDENCE_CLASSIFICATION,
        definition=catalogue.definition(ExperimentName.EVIDENCE_CLASSIFICATION),
        overwrite_policy=OverwritePolicy.REUSE,
    )
    manifest = execute_evidence_classification(store, layout, request)
    resolved = store.resolve(manifest.artifact_id)
    assert resolved.state == ArtifactState.COMPLETED
    payload = json.loads(Path(manifest.payload_paths[0]).read_text(encoding="utf-8"))
    assert payload["synthesis_artifact_id"] is None
    statuses = payload["statuses"]
    assert statuses
    assert {row["final_state"] for row in statuses} == {EvidenceStatus.NOT_TESTED.value}


def _contrast(
    pair: str,
    decision: ComparisonDecision,
    mean_difference: float,
    holm_p: float,
    bca_low: float,
) -> PairedComparisonRecord:
    digest = Sha256Digest("a" * 64)
    return PairedComparisonRecord(
        contrast_name=ContrastName("primary"),
        family=MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
        pair=DirectedPairName(pair),
        method_a=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        method_b=TransferMethod.LOCAL_ONLY,
        metric=MetricId.RELATIVE_MACRO_CE_GAIN,
        paired_seed_count=8,
        mean_difference=mean_difference,
        median_difference=mean_difference,
        bca_ci_low=bca_low,
        bca_ci_high=mean_difference + 0.02,
        raw_p=holm_p,
        holm_p=holm_p,
        materiality_threshold=0.01,
        equivalence_margin_low=None,
        equivalence_margin_high=None,
        input_metric_artifact_ids=(ArtifactIdentifier("metric"),),
        dependency_fingerprint_sha256=digest,
        decision=decision,
    )


def test_utility_family_supported_partial_null_and_harm() -> None:
    required = 4
    holm = 0.05
    bca = 0.0
    empty = utility_family_status(
        (),
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_ONLY,
        holm,
        bca,
        required,
        False,
    )
    assert empty.status == EvidenceStatus.NOT_TESTED
    superior = tuple(
        _contrast(f"pair-{index}", ComparisonDecision.SUPERIOR, 0.05, 0.01, 0.02)
        for index in range(required)
    )
    supported = utility_family_status(
        superior,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_ONLY,
        holm,
        bca,
        required,
        False,
    )
    assert supported.status == EvidenceStatus.SUPPORTED
    partial = utility_family_status(
        superior[:1],
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_ONLY,
        holm,
        bca,
        required,
        False,
    )
    assert partial.status == EvidenceStatus.PARTIALLY_SUPPORTED
    null = utility_family_status(
        (_contrast("pair-0", ComparisonDecision.NOT_SUPPORTED, 0.0, 0.4, -0.01),),
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_ONLY,
        holm,
        bca,
        required,
        False,
    )
    assert null.status == EvidenceStatus.NULL_RESULT
    harmful = utility_family_status(
        (_contrast("pair-0", ComparisonDecision.NOT_SUPPORTED, -0.05, 0.4, -0.08),),
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_ONLY,
        holm,
        bca,
        required,
        False,
    )
    assert harmful.status == EvidenceStatus.NOT_SUPPORTED


def _local_sir_contrast(
    pair: str,
    decision: ComparisonDecision,
    mean_difference: float,
    holm_p: float,
    bca_low: float,
    bca_high: float,
) -> PairedComparisonRecord:
    digest = Sha256Digest("a" * 64)
    return PairedComparisonRecord(
        contrast_name=ContrastName("external-source-vs-local-sir"),
        family=MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR,
        pair=DirectedPairName(pair),
        method_a=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        method_b=TransferMethod.LOCAL_SIR,
        metric=MetricId.RELATIVE_MACRO_CE_GAIN,
        paired_seed_count=8,
        mean_difference=mean_difference,
        median_difference=mean_difference,
        bca_ci_low=bca_low,
        bca_ci_high=bca_high,
        raw_p=holm_p,
        holm_p=holm_p,
        materiality_threshold=0.01,
        equivalence_margin_low=None,
        equivalence_margin_high=None,
        input_metric_artifact_ids=(ArtifactIdentifier("metric"),),
        dependency_fingerprint_sha256=digest,
        decision=decision,
    )


def test_utility_family_local_sir_equivalence_kill_fires_without_advantage() -> None:
    required = 4
    holm = 0.05
    bca = 0.0
    equivalent = tuple(
        _local_sir_contrast(f"pair-{index}", ComparisonDecision.EQUIVALENT, 0.0, 0.4, -0.01, 0.01)
        for index in range(required)
    )
    result = utility_family_status(
        equivalent,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_SIR,
        holm,
        bca,
        required,
        True,
    )
    assert result.status == EvidenceStatus.NOT_SUPPORTED


def test_utility_family_local_sir_superior_kill_fires() -> None:
    required = 4
    holm = 0.05
    bca = 0.0
    threshold = active_config().scientific.materiality.realized_relative_macro_ce
    dominant = tuple(
        _local_sir_contrast(
            f"pair-{index}",
            ComparisonDecision.NOT_SUPPORTED,
            -threshold - 0.01,
            0.01,
            -0.05,
            -0.01,
        )
        for index in range(required)
    )
    result = utility_family_status(
        dominant,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_SIR,
        holm,
        bca,
        required,
        True,
    )
    assert result.status == EvidenceStatus.NOT_SUPPORTED


def test_utility_family_local_sir_kill_does_not_fire_with_fedorbit_advantage() -> None:
    required = 4
    holm = 0.05
    bca = 0.0
    records = (
        *(
            _local_sir_contrast(
                f"pair-{index}", ComparisonDecision.EQUIVALENT, 0.0, 0.4, -0.01, 0.01
            )
            for index in range(required - 1)
        ),
        _local_sir_contrast("pair-advantage", ComparisonDecision.SUPERIOR, 0.05, 0.01, 0.02, 0.08),
    )
    result = utility_family_status(
        records,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_SIR,
        holm,
        bca,
        required,
        True,
    )
    assert result.status != EvidenceStatus.NOT_SUPPORTED
