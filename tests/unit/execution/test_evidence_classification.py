from __future__ import annotations

import json
from pathlib import Path

from fedorbit.analysis.records import ComparisonDecision, PairedComparisonRecord
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.classification import (
    _utility_family_status,
    execute_evidence_classification,
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
    empty = _utility_family_status(
        (),
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_ONLY,
        holm,
        bca,
        required,
        False,
    )
    assert empty[0] == EvidenceStatus.NOT_TESTED
    superior = tuple(
        _contrast(f"pair-{index}", ComparisonDecision.SUPERIOR, 0.05, 0.01, 0.02)
        for index in range(required)
    )
    supported = _utility_family_status(
        superior,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_ONLY,
        holm,
        bca,
        required,
        False,
    )
    assert supported[0] == EvidenceStatus.SUPPORTED
    partial = _utility_family_status(
        superior[:1],
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_ONLY,
        holm,
        bca,
        required,
        False,
    )
    assert partial[0] == EvidenceStatus.PARTIALLY_SUPPORTED
    null = _utility_family_status(
        (_contrast("pair-0", ComparisonDecision.NOT_SUPPORTED, 0.0, 0.4, -0.01),),
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_ONLY,
        holm,
        bca,
        required,
        False,
    )
    assert null[0] == EvidenceStatus.NULL_RESULT
    harmful = _utility_family_status(
        (_contrast("pair-0", ComparisonDecision.NOT_SUPPORTED, -0.05, 0.4, -0.08),),
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_ONLY,
        holm,
        bca,
        required,
        False,
    )
    assert harmful[0] == EvidenceStatus.NOT_SUPPORTED
