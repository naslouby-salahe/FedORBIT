from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import cast

from fedorbit.analysis.records import ComparisonDecision, PairedComparisonRecord
from fedorbit.config.loading import active_config
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.classification import (
    execute_evidence_classification,
    utility_family_status,
)
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.experiments.validation import persist_synthetic_experiment_payload
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactState,
    ConfigurationSection,
    ContrastName,
    DirectedPairName,
    EvidenceStatus,
    ExperimentCondition,
    ExperimentName,
    ExperimentSeed,
    MetricId,
    MultiplicityFamily,
    OverwritePolicy,
    ProducerModuleName,
    ResearchQuestion,
    Sha256Digest,
    StableJsonPayload,
    SupportSize,
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


def _theorem_instance_payload(
    exact_minima: bool, valid_certificate: bool
) -> Callable[[Sha256Digest], StableJsonPayload]:
    def build(fingerprint: Sha256Digest) -> StableJsonPayload:
        cell: StableJsonPayload = cast(
            StableJsonPayload,
            OrderedDict(
                block_pattern=[2],
                support=1,
                seed=1103,
                instance_index=0,
                absolute_objective_error=0.0,
                exact_minima=exact_minima,
                valid_certificate=valid_certificate,
            ),
        )
        return cast(
            StableJsonPayload,
            OrderedDict(
                experiment=ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION.value,
                dependency_fingerprint_sha256=fingerprint,
                cell=cell,
            ),
        )

    return build


def test_evidence_classification_exactness_reflects_theorem_instances(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    catalogue = build_catalogue()
    theorem_request = ExperimentExecutionRequest(
        experiment=ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION,
        definition=catalogue.definition(ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION),
        overwrite_policy=OverwritePolicy.REUSE,
    )
    persist_synthetic_experiment_payload(
        store,
        layout,
        theorem_request,
        ExperimentSeed(theorem_request.definition.seeds[0]),
        _theorem_instance_payload(True, True),
        frozenset({ConfigurationSection.ACTION}),
        ProducerModuleName("fedorbit.experiments.validation"),
        "theorem-exhaustive.test-pass",
        ExperimentCondition("pattern-2-seed1103-instance0"),
        SupportSize(1),
    )
    request = ExperimentExecutionRequest(
        experiment=ExperimentName.EVIDENCE_CLASSIFICATION,
        definition=catalogue.definition(ExperimentName.EVIDENCE_CLASSIFICATION),
        overwrite_policy=OverwritePolicy.REUSE,
    )
    manifest = execute_evidence_classification(store, layout, request)
    payload = json.loads(Path(manifest.payload_paths[0]).read_text(encoding="utf-8"))
    rows = {row["question"]: row for row in payload["statuses"]}
    exactness_row = rows[ResearchQuestion.EXACT_SPARSE_SEPARATOR_EXACTNESS.value]
    assert exactness_row["final_state"] == EvidenceStatus.SUPPORTED.value


def test_evidence_classification_exactness_fails_on_wrong_minima(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    catalogue = build_catalogue()
    theorem_request = ExperimentExecutionRequest(
        experiment=ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION,
        definition=catalogue.definition(ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION),
        overwrite_policy=OverwritePolicy.REUSE,
    )
    persist_synthetic_experiment_payload(
        store,
        layout,
        theorem_request,
        ExperimentSeed(theorem_request.definition.seeds[0]),
        _theorem_instance_payload(False, True),
        frozenset({ConfigurationSection.ACTION}),
        ProducerModuleName("fedorbit.experiments.validation"),
        "theorem-exhaustive.test-fail",
        ExperimentCondition("pattern-2-seed1103-instance1"),
        SupportSize(1),
    )
    request = ExperimentExecutionRequest(
        experiment=ExperimentName.EVIDENCE_CLASSIFICATION,
        definition=catalogue.definition(ExperimentName.EVIDENCE_CLASSIFICATION),
        overwrite_policy=OverwritePolicy.REUSE,
    )
    manifest = execute_evidence_classification(store, layout, request)
    payload = json.loads(Path(manifest.payload_paths[0]).read_text(encoding="utf-8"))
    rows = {row["question"]: row for row in payload["statuses"]}
    exactness_row = rows[ResearchQuestion.EXACT_SPARSE_SEPARATOR_EXACTNESS.value]
    assert exactness_row["final_state"] == EvidenceStatus.NOT_SUPPORTED.value
