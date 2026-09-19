from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from fedorbit.experiments.assembly import persist_or_reuse_target_importance
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.manifests import (
    FINE_CONCEPT_FIELD,
    NATIVE_CLASS_IDS_FIELD,
    MethodReadableTransferEligibility,
    SemanticCellManifest,
)
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.methods.target import TransferNodeRisk
from fedorbit.types import (
    AnonymousNodeDisplayId,
    ArtifactStage,
    ArtifactState,
    CoarseGroup,
    DatasetId,
    ExperimentName,
    ExperimentSeed,
    FieldDescription,
    GitRevision,
    OverwritePolicy,
    Sha256Digest,
    TransferMethod,
)


def test_method_readable_schema_forbids_native_ids_and_fine_concepts() -> None:
    row = MethodReadableTransferEligibility(
        client=DatasetId.TON_IOT_NETWORK,
        seed=ExperimentSeed(1103),
        coarse_group=CoarseGroup.DISRUPTION,
        anonymous_node_id=AnonymousNodeDisplayId("node-0001"),
        present=True,
        train_count=1,
        meta_count=1,
        confirm_count=1,
        test_count=1,
        source_eligible=False,
        target_eligible=False,
    )
    payload = row.model_dump(mode="json")
    assert NATIVE_CLASS_IDS_FIELD not in payload
    assert FINE_CONCEPT_FIELD not in payload
    assert row.native_local_class_ids is None
    assert row.fine_concept is None


def test_semantic_cell_manifest_contains_section_22_3_fields() -> None:
    digest = Sha256Digest("a" * 64)
    manifest = SemanticCellManifest(
        experiment=ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        dataset=DatasetId.TON_IOT_NETWORK,
        source_client=DatasetId.EDGE_IIOTSET_NETWORK,
        target_client=DatasetId.TON_IOT_NETWORK,
        directed_pair=None,
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        condition=None,
        support=None,
        seed=ExperimentSeed(1103),
        scientific_configuration_sha256=digest,
        dependency_fingerprint_sha256=digest,
        producer_stage=ArtifactStage.EVALUATION,
        upstream_artifact_ids=(),
        relevant_code_sha256=digest,
        material_runtime_sha256=digest,
        git_commit=GitRevision("deadbeef"),
        environment_sha256=digest,
        state=ArtifactState.COMPLETED,
        state_reason=FieldDescription("completed"),
    )
    payload = manifest.model_dump(mode="json")
    for field in (
        "experiment",
        "dataset",
        "source_client",
        "target_client",
        "directed_pair",
        "method",
        "condition",
        "support",
        "seed",
        "scientific_configuration_sha256",
        "dependency_fingerprint_sha256",
        "producer_stage",
        "upstream_artifact_ids",
        "dataset_manifest_sha256",
        "split_sha256",
        "preprocessing_sha256",
        "source_checkpoint_sha256",
        "response_packet_sha256",
        "target_checkpoint_sha256",
        "importance_vector_sha256",
        "resource_manifest_sha256",
        "relevant_code_sha256",
        "material_runtime_sha256",
        "git_commit",
        "environment_sha256",
        "state",
        "state_reason",
    ):
        assert field in payload


def test_target_importance_is_reused_by_semantic_identity(tmp_path: Path) -> None:
    layout = build_layout(tmp_path)
    store = ArtifactStore(layout.execution_root)
    risks = (
        TransferNodeRisk(node_index=0, is_actionable=True, meta_class_risk=0.4),
        TransferNodeRisk(node_index=1, is_actionable=True, meta_class_risk=0.6),
    )
    first = persist_or_reuse_target_importance(
        store,
        layout,
        DatasetId.TON_IOT_NETWORK,
        1103,
        risks,
        OverwritePolicy.REPLACE,
        (),
    )
    second = persist_or_reuse_target_importance(
        store,
        layout,
        DatasetId.TON_IOT_NETWORK,
        1103,
        risks,
        OverwritePolicy.REUSE,
        (),
    )
    assert first is not None and second is not None
    assert OrderedDict(first.weights_by_node_index) == OrderedDict(second.weights_by_node_index)
    packets = tuple(
        manifest
        for manifest in store.all_manifests()
        if manifest.artifact_type.value == "target_importance"
    )
    assert len(packets) == 1
