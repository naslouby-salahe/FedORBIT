from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.analysis.records import MetricDirection
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.classification import (
    pairs_blocked_by_scientific_failure,
    scientific_algorithmic_failure_seeds,
    utility_family_status,
)
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.experiments.scoring import persist_primary_transfer_metric
from fedorbit.experiments.synthesis import (
    StatisticsError,
    persist_family_definition_lock,
    registered_family_contrast_counts,
)
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import WorkspaceLayout, build_layout
from fedorbit.types import (
    ArtifactIdentifier,
    CellUnavailabilityReason,
    ContrastName,
    DatasetId,
    DirectedPairName,
    EvidenceStatus,
    ExperimentName,
    Index,
    InvalidReason,
    MetricId,
    MetricUnit,
    MultiplicityFamily,
    OverwritePolicy,
    Sha256Digest,
    TransferMethod,
)

_PAIR = DirectedPairName("ton_iot_linux_process_host -> ton_iot_windows10_host")


def _persist_abstention(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    seed: int,
    category: CellUnavailabilityReason,
) -> None:
    persist_primary_transfer_metric(
        store,
        layout,
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        _PAIR,
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        DatasetId.TON_IOT_WINDOWS10_HOST,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        seed,
        MetricId.ABSTENTION_INDICATOR,
        None,
        MetricUnit.BOOLEAN,
        MetricDirection.DESCRIPTIVE,
        (ArtifactIdentifier(category.value),),
        OverwritePolicy.REPLACE,
        valid=False,
        invalid_reason=InvalidReason(f"ScientificAlgorithmicFailureError: seed {seed}"),
    )


def test_scientific_algorithmic_failure_seeds_counts_per_pair(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    _persist_abstention(
        store, layout, 1103, CellUnavailabilityReason.SCIENTIFIC_ALGORITHMIC_FAILURE
    )
    _persist_abstention(
        store, layout, 2207, CellUnavailabilityReason.SCIENTIFIC_ALGORITHMIC_FAILURE
    )
    _persist_abstention(store, layout, 3319, CellUnavailabilityReason.INVALID_EVALUATION_DATA)

    counts = scientific_algorithmic_failure_seeds(store)
    assert counts[_PAIR] == 2


def test_pairs_blocked_by_scientific_failure_requires_more_than_allowed(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    _persist_abstention(
        store, layout, 1103, CellUnavailabilityReason.SCIENTIFIC_ALGORITHMIC_FAILURE
    )
    assert pairs_blocked_by_scientific_failure(store) == frozenset()

    _persist_abstention(
        store, layout, 2207, CellUnavailabilityReason.SCIENTIFIC_ALGORITHMIC_FAILURE
    )
    assert pairs_blocked_by_scientific_failure(store) == frozenset({_PAIR})


def test_utility_family_status_excludes_blocked_pairs() -> None:
    from fedorbit.analysis.records import ComparisonDecision, PairedComparisonRecord

    digest = Sha256Digest("a" * 64)
    superior = (
        PairedComparisonRecord(
            contrast_name=ContrastName("primary"),
            family=MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
            pair=DirectedPairName("pair-0"),
            method_a=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            method_b=TransferMethod.LOCAL_ONLY,
            metric=MetricId.RELATIVE_MACRO_CE_GAIN,
            paired_seed_count=8,
            mean_difference=0.05,
            median_difference=0.05,
            bca_ci_low=0.02,
            bca_ci_high=0.07,
            raw_p=0.01,
            holm_p=0.01,
            materiality_threshold=0.01,
            equivalence_margin_low=None,
            equivalence_margin_high=None,
            input_metric_artifact_ids=(ArtifactIdentifier("metric"),),
            dependency_fingerprint_sha256=digest,
            decision=ComparisonDecision.SUPERIOR,
        ),
    )
    blocked = utility_family_status(
        superior,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.LOCAL_ONLY,
        0.05,
        0.0,
        1,
        False,
        blocked_pairs=frozenset({DirectedPairName("pair-0")}),
    )
    assert blocked.status == EvidenceStatus.NOT_SUPPORTED


def test_family_definition_lock_persists_and_rejects_registered_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    catalogue = build_catalogue()
    request = ExperimentExecutionRequest(
        experiment=ExperimentName.STATISTICAL_SYNTHESIS,
        definition=catalogue.definition(ExperimentName.STATISTICAL_SYNTHESIS),
        overwrite_policy=OverwritePolicy.REUSE,
    )

    persist_family_definition_lock(store, layout, request)
    persist_family_definition_lock(store, layout, request)

    import fedorbit.experiments.synthesis as synthesis_module

    original = registered_family_contrast_counts()
    changed_family = next(iter(original))
    mutated: dict[MultiplicityFamily, Index] = {
        **original,
        changed_family: original[changed_family] + 1,
    }

    def _mutated_counts() -> dict[MultiplicityFamily, Index]:
        return mutated

    monkeypatch.setattr(synthesis_module, "registered_family_contrast_counts", _mutated_counts)
    with pytest.raises(StatisticsError):
        persist_family_definition_lock(store, layout, request)
