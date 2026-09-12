from __future__ import annotations

import json
import statistics
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from fedorbit.analysis.metrics import (
    harm_indicator,
)
from fedorbit.analysis.records import (
    ComparisonDecision,
    MetricRecord,
    PairedComparisonRecord,
    StatisticalAlternative,
    StatisticalExactness,
    StatisticalMetadataRecord,
    validate_comparison_metadata,
)
from fedorbit.analysis.statistics import (
    NamedPValue,
    PValueSet,
    exact_sign_flip_test,
    holm_step_down,
    paired_bca_interval,
    statistical_bootstrap_seed,
    tost_equivalence,
)
from fedorbit.config.loading import active_config
from fedorbit.datasets.common import (
    file_sha256,
)
from fedorbit.datasets.materialization import (
    MaterializedClient,
)
from fedorbit.datasets.ontology import TRANSFER_ONTOLOGY, transfer_concept_for, transfer_eligibility
from fedorbit.experiments.cells import experiment_relevance
from fedorbit.experiments.protocol import ExperimentExecutionRequest
from fedorbit.experiments.scoring import build_completion_manifest, latest_completed_manifest
from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
)
from fedorbit.infrastructure.environment import environment_snapshot
from fedorbit.infrastructure.evidence import TableScalar
from fedorbit.infrastructure.manifests import (
    ReusableArtifactManifest,
    artifact_id,
)
from fedorbit.infrastructure.provenance import (
    configuration_subset_digest,
    implementation_fingerprint,
    runtime_fingerprint,
    stage_dependency_fingerprint,
)
from fedorbit.infrastructure.runtime import (
    RandomSeed,
    current_code_revision,
)
from fedorbit.infrastructure.storage import atomic_write_json
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
    experiment_workspace,
    safe_slug,
)
from fedorbit.types import (
    ArtifactFingerprint,
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactPath,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    ArtifactTypeName,
    BootstrapPurpose,
    ComparisonContrastSuffix,
    ComparisonStatistic,
    ConfigurationSection,
    ContrastName,
    DatasetId,
    DirectedPair,
    DirectedPairName,
    EvaluationConditionName,
    ExperimentCondition,
    ExperimentLocalMethod,
    ExperimentName,
    FineLabel,
    Index,
    MethodName,
    MetricId,
    MultiplicityFamily,
    OracleTransferConcept,
    OverwritePolicy,
    ProducerModuleName,
    PValueName,
    RelativeGain,
    ResampleCount,
    SampleCount,
    SemanticCell,
    SemanticCoordinateText,
    Sha256Digest,
    Split,
    StableJsonPayload,
    StatisticalTestName,
    SupportCount,
    TransferMethod,
)

_MODULE_NAME = ProducerModuleName("fedorbit.experiments.synthesis")
_PRINCIPAL_CONDITION = EvaluationConditionName("principal") #TODO: should be enum instead of hardcoded string


def _concept_split_support(
    client: MaterializedClient, concept: OracleTransferConcept, split: Split
) -> Index:
    total: Index = sum(
        client.class_row_counts[label][split]
        for label in client.class_manifest.class_names
        if transfer_concept_for(client.dataset, FineLabel(label)) == concept
    )
    return total


def transfer_ontology_null_padding_rows(
    pair: DirectedPairName,
    source_client: MaterializedClient,
    target_client: MaterializedClient,
) -> tuple[StableJsonPayload, ...]:
    rows: list[StableJsonPayload] = []
    for concept in OracleTransferConcept:
        coarse_group, _, _ = TRANSFER_ONTOLOGY[concept]
        source_train = _concept_split_support(source_client, concept, Split.TRAIN)
        source_meta = _concept_split_support(source_client, concept, Split.META)
        target_meta = _concept_split_support(target_client, concept, Split.META)
        target_confirm = _concept_split_support(target_client, concept, Split.CONFIRM)
        target_test = _concept_split_support(target_client, concept, Split.TEST)
        source_real = (source_train + source_meta) > 0
        target_real = (target_meta + target_confirm + target_test) > 0
        eligibility = transfer_eligibility(
            source_train, source_meta, target_meta, target_confirm, target_test
        )
        action_eligible = eligibility.source_eligible and eligibility.target_eligible
        null_reason: str | None = None #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        if not source_real:
            null_reason = f"{concept.value} absent from source dataset"
        elif not target_real:
            null_reason = f"{concept.value} absent from target dataset"
        elif not action_eligible:
            null_reason = f"{concept.value} present but below configured support minimum"
        rows.append(
            cast(
                StableJsonPayload,
                OrderedDict(
                    candidate_concept=concept.value,
                    pair=pair,
                    coarse_group=coarse_group.value,
                    source_real=source_real,
                    target_real=target_real,
                    support_counts=cast(
                        StableJsonPayload,
                        OrderedDict(
                            source_train=source_train,
                            source_meta=source_meta,
                            target_meta=target_meta,
                            target_confirm=target_confirm,
                            target_test=target_test,
                        ),
                    ),
                    action_eligibility=action_eligible,
                    null_reason=null_reason,
                ),
            )
        )
    return tuple(rows)


_STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS = frozenset({ConfigurationSection.METRICS})


@dataclass(frozen=True, slots=True)
class _SeedMetric:
    value: float #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    artifact_id: ArtifactIdentifier


def _iter_completed_json_payloads(
    store: ArtifactStore, experiment: ExperimentName,
    payload_key: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
) -> Iterator[tuple[ReusableArtifactManifest,
                    Mapping[str, StableJsonPayload]]]: #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    experiment_value = experiment.value
    for manifest in store.all_manifests():
        if experiment_value not in manifest.semantic_producer_coordinates:
            continue
        try:
            resolved = store.resolve(manifest.artifact_id)
        except ValueError:
            continue
        if resolved.state != ArtifactState.COMPLETED:
            continue
        for payload_path in resolved.payload_paths:
            path = Path(payload_path)
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            record_payload = payload.get(payload_key)
            if record_payload is not None:
                yield resolved, record_payload


def completed_experiment_metric_records(
    store: ArtifactStore, experiment: ExperimentName
) -> tuple[MetricRecord, ...]:
    return tuple(
        MetricRecord.model_validate(payload)
        for _, payload in _iter_completed_json_payloads(store, experiment, "metric_record") #TODO: should be enum instead of hardcoded string
    )


def completed_experiment_metric_records_with_support(
    store: ArtifactStore, experiment: ExperimentName
) -> tuple[tuple[MetricRecord, SupportCount | None], ...]:
    results: list[tuple[MetricRecord, SupportCount | None]] = []
    for manifest, payload in _iter_completed_json_payloads(store, experiment, "metric_record"): #TODO: should be enum instead of hardcoded string
        record = MetricRecord.model_validate(payload)
        coordinates = json.loads(manifest.semantic_producer_coordinates)
        support = coordinates.get("support") #TODO: should be enum instead of hardcoded string
        results.append((record, support))
    return tuple(results)


def transfer_ontology_and_null_padding_rows(
    store: ArtifactStore,
) -> tuple[Mapping[str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
                   TableScalar], ...]:
    manifest = latest_completed_manifest(
        store, ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION
    )
    if manifest is None or len(manifest.payload_paths) != 1:
        return ()
    payload_path = Path(manifest.payload_paths[0])
    if not payload_path.is_file():
        return ()
    payload = cast(Mapping[str, object], json.loads(payload_path.read_text(encoding="utf-8")))
    rows: list[Mapping[str, TableScalar]] = []
    for pair_entry in cast(list[Mapping[str, object]], payload.get("primary_pairs", [])): #TODO: should be enum instead of hardcoded string
        for ontology_row in cast(
            list[Mapping[str, object]], pair_entry.get("transfer_ontology", []) #TODO: should be enum instead of hardcoded string
        ):
            support = cast(
                Mapping[str, object],
                ontology_row.get("support_counts" #TODO: should be enum instead of hardcoded string
                                 , OrderedDict[str, object]()),
            )
            rows.append(
                OrderedDict(
                    candidate_concept=cast(str | None, ontology_row.get("candidate_concept")), #TODO: should be enum instead of hardcoded string
                    pair=cast(str | None, ontology_row.get("pair")), #TODO: should be enum instead of hardcoded string
                    coarse_group=cast(str | None, ontology_row.get("coarse_group")), #TODO: should be enum instead of hardcoded string
                    source_real_or_null="real" if ontology_row.get("source_real") else "null", #TODO: should be enum instead of hardcoded string
                    target_real_or_null="real" if ontology_row.get("target_real") else "null", #TODO: should be enum instead of hardcoded string
                    support_counts=(
                        f"source_train={support.get('source_train')}," #TODO: should be enum instead of hardcoded string
                        f"source_meta={support.get('source_meta')}," #TODO: should be enum instead of hardcoded string
                        f"target_meta={support.get('target_meta')}," #TODO: should be enum instead of hardcoded string
                        f"target_confirm={support.get('target_confirm')}," #TODO: should be enum instead of hardcoded string
                        f"target_test={support.get('target_test')}" #TODO: should be enum instead of hardcoded string
                    ),
                    action_eligibility=cast(bool | None, ontology_row.get("action_eligibility")), #TODO: should be enum instead of hardcoded string
                    null_reason=cast(str | None, ontology_row.get("null_reason")), #TODO: should be enum instead of hardcoded string
                )
            )
    return tuple(rows)


def completed_primary_transfer_metric_records(store: ArtifactStore) -> tuple[MetricRecord, ...]:
    return tuple(
        MetricRecord.model_validate(payload)
        for _, payload in _iter_completed_json_payloads(
            store, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER, "metric_record" #TODO: should be enum instead of hardcoded string
        )
    )


def completed_primary_transfer_comparison_records(
    store: ArtifactStore,
) -> tuple[PairedComparisonRecord, ...]:
    return tuple(
        PairedComparisonRecord.model_validate(payload)
        for _, payload in _iter_completed_json_payloads(
            store, ExperimentName.STATISTICAL_SYNTHESIS, "comparison_record" #TODO: should be enum instead of hardcoded string
        )
    )


def _completed_primary_transfer_macro_ce(
    store: ArtifactStore,
) -> Mapping[tuple[DirectedPairName, TransferMethod, RandomSeed], _SeedMetric]:
    result: OrderedDict[tuple[DirectedPairName, TransferMethod, RandomSeed], _SeedMetric] = (
        OrderedDict()
    )
    for resolved, record_payload in _iter_completed_json_payloads(
        store, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER, "metric_record" #TODO: should be enum instead of hardcoded string
    ):
        record = MetricRecord.model_validate(record_payload)
        if (
            record.metric_name != MetricId.MACRO_CROSS_ENTROPY
            or not record.valid
            or record.metric_value is None
        ):
            continue
        result[(record.pair, record.method, record.seed)] = _SeedMetric(
            float(record.metric_value), resolved.artifact_id
        )
    return result


def persist_primary_transfer_comparison(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    pair: DirectedPairName,
    method: TransferMethod,
    paired_seed_count: Index,
    mean_difference: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    median_difference: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    bca_ci_low: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    bca_ci_high: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    raw_p: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    holm_p: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    decision: ComparisonDecision,
    input_metric_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        directed_pair=DirectedPair(
            source=DatasetId(pair.split(" -> ")[0]), target=DatasetId(pair.split(" -> ")[1])
        ),
        method=method,
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.STATISTICS,
            cell,
            relevance,
            tuple(identifier.value for identifier in input_metric_artifact_ids),
            _STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    comparison = PairedComparisonRecord(
        contrast_name=ContrastName(f"{method.value} vs {TransferMethod.LOCAL_ONLY.value}: {pair}"),
        family=MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
        pair=DirectedPairName(pair),
        method_a=method,
        method_b=TransferMethod.LOCAL_ONLY,
        metric=MetricId.MACRO_CROSS_ENTROPY,
        paired_seed_count=paired_seed_count,
        mean_difference=mean_difference,
        median_difference=median_difference,
        bca_ci_low=bca_ci_low,
        bca_ci_high=bca_ci_high,
        raw_p=raw_p,
        holm_p=holm_p,
        materiality_threshold=active_config().scientific.materiality.realized_relative_macro_ce,
        equivalence_margin_low=None,
        equivalence_margin_high=None,
        input_metric_artifact_ids=tuple(input_metric_artifact_ids),
        dependency_fingerprint_sha256=fingerprint,
        decision=decision,
    )
    payload_path = (
        experiment_workspace(layout, experiment)
        / "artifacts" #TODO: should be enum instead of hardcoded string
        / "derived" #TODO: should be enum instead of hardcoded string
        / f"comparison.{pair.replace(' -> ', '-to-')}.{method.value}.json"
    )
    payload = cast(
        StableJsonPayload, OrderedDict(comparison_record=comparison.model_dump(mode="json"))
    )
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(ArtifactStage.STATISTICS).sha256)
    completion = build_completion_manifest(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=ArtifactStage.STATISTICS,
        upstream_artifact_ids=tuple(input_metric_artifact_ids),
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.OTHER.value), payload, Sha256Digest(fingerprint)
            ),
            artifact_type=ArtifactType.OTHER,
            semantic_producer_coordinates=coordinates,
            producer_stage=ArtifactStage.STATISTICS,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=tuple(input_metric_artifact_ids),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            material_runtime_sha256=runtime_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_sha256,
            schema_version="1.0", #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
            created_git_commit=current_code_revision().commit,
            created_environment_sha256=environment_snapshot().fingerprint_sha256,
            state=ArtifactState.COMPLETED,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
    )
    store.write_completed(manifest, completion)
    return manifest


def execute_statistical_synthesis(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    metrics = _completed_primary_transfer_macro_ce(store)
    pairs = sorted({pair for pair, _, _ in metrics})
    methods = sorted(
        {method for _, method, _ in metrics if method != TransferMethod.LOCAL_ONLY},
        key=lambda method: method.value,
    )
    statistics_config = active_config().scientific.statistics
    for method in methods:
        raw_p_by_pair: OrderedDict[str, float] = OrderedDict() #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        metadata_inputs: OrderedDict[str, tuple[Index, RandomSeed]] = OrderedDict() #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        contrasts: OrderedDict[
            str,
            tuple[
                Index,
                float | None,
                float | None,
                float | None,
                float | None,
                tuple[ArtifactIdentifier, ...],
            ],
        ] = OrderedDict()
        for pair in pairs:
            local_only_seeds = OrderedDict(
                (seed, entry)
                for (candidate_pair, candidate_method, seed), entry in metrics.items()
                if candidate_pair == pair and candidate_method == TransferMethod.LOCAL_ONLY
            )
            method_seeds = OrderedDict(
                (seed, entry)
                for (candidate_pair, candidate_method, seed), entry in metrics.items()
                if candidate_pair == pair and candidate_method == method
            )
            shared_seeds = sorted(set(local_only_seeds) & set(method_seeds))
            paired_seed_count: Index = len(shared_seeds)
            input_ids = tuple(
                identifier
                for seed in shared_seeds
                for identifier in (
                    local_only_seeds[seed].artifact_id,
                    method_seeds[seed].artifact_id,
                )
            )
            if len(shared_seeds) < statistics_config.minimum_valid_paired_seeds:
                contrasts[pair] = (paired_seed_count, None, None, None, None, input_ids)
                continue
            local_only_values = tuple(local_only_seeds[seed].value for seed in shared_seeds)
            method_values = tuple(method_seeds[seed].value for seed in shared_seeds)
            bootstrap_seed = statistical_bootstrap_seed(
                ContrastName(f"{method.value} vs Local-Only"),
                MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
                DirectedPairName(pair),
                MetricId.MACRO_CROSS_ENTROPY,
                BootstrapPurpose("primary-transfer-gain"),
            )
            bca = paired_bca_interval(local_only_values, method_values, bootstrap_seed)
            sign_flip = exact_sign_flip_test(local_only_values, method_values)
            raw_p_by_pair[pair] = sign_flip.p_value
            metadata_inputs[pair] = (sign_flip.nonzero_difference_count, bootstrap_seed)
            contrasts[pair] = (
                paired_seed_count,
                float(sign_flip.mean_difference),
                float(sign_flip.median_difference),
                None if bca.lower is None else float(bca.lower),
                None if bca.upper is None else float(bca.upper),
                input_ids,
            )
        holm_adjusted = holm_step_down(
            PValueSet(
                tuple(
                    NamedPValue(PValueName(pair), p_value)
                    for pair, p_value in raw_p_by_pair.items()
                )
            )
        )
        criteria = active_config().scientific.evaluation_criteria.strict_cross_telemetry_utility
        for pair in pairs:
            paired_seed_count, mean_difference, median_difference, bca_low, bca_high, input_ids = (
                contrasts[pair]
            )
            if mean_difference is None:
                decision = ComparisonDecision.INSUFFICIENT_EVIDENCE
                raw_p = None
                holm_p = None
            else:
                raw_p = raw_p_by_pair[pair]
                holm_p = holm_adjusted.value_of(PValueName(pair))
                if bca_low is None:
                    decision = ComparisonDecision.DEGENERATE
                elif (
                    holm_p is not None
                    and holm_p <= criteria.holm_adjusted_p_maximum
                    and bca_low > criteria.bca_lower_bound_strictly_greater_than
                ):
                    decision = ComparisonDecision.SUPERIOR
                else:
                    decision = ComparisonDecision.NOT_SUPPORTED
            if not input_ids:
                continue
            comparison_manifest = persist_primary_transfer_comparison(
                store,
                layout,
                request.experiment,
                pair,
                method,
                paired_seed_count,
                mean_difference,
                median_difference,
                bca_low,
                bca_high,
                raw_p,
                float(holm_p) if holm_p is not None else None,
                decision,
                input_ids,
                request.overwrite_policy,
            )
            if comparison_manifest is not None and pair in metadata_inputs:
                resolved_comparison = store.resolve(comparison_manifest.artifact_id)
                comparison_record = PairedComparisonRecord.model_validate(
                    json.loads(Path(resolved_comparison.payload_paths[0]).read_text())[
                        "comparison_record" #TODO: should be enum instead of hardcoded string
                    ]
                )
                nonzero_count, bootstrap_seed = metadata_inputs[pair]
                family_size: SampleCount = len(raw_p_by_pair)
                persist_statistical_metadata(
                    store,
                    layout,
                    request.experiment,
                    DirectedPairName(pair),
                    method,
                    comparison_record,
                    ComparisonStatistic.SIGN_FLIP_SUPERIORITY,
                    nonzero_count,
                    bootstrap_seed,
                    _holm_rank(raw_p_by_pair, pair),
                    family_size,
                    request.overwrite_policy,
                )
    gap_metrics = _completed_real_packet_coupling_gap(store)
    coupling_pairs = sorted({pair for pair, _ in gap_metrics})
    coupling_raw_p_by_pair: OrderedDict[str, float] = OrderedDict() #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    coupling_metadata_inputs: OrderedDict[str, tuple[Index, RandomSeed]] = OrderedDict() #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    coupling_contrasts: OrderedDict[
        str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        tuple[
            Index,
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            tuple[ArtifactIdentifier, ...],
        ],
    ] = OrderedDict()
    for pair in coupling_pairs:
        pair_seeds = OrderedDict(
            (seed, entry)
            for (candidate_pair, seed), entry in gap_metrics.items()
            if candidate_pair == pair
        )
        seeds = sorted(pair_seeds)
        paired_seed_count: Index = len(seeds)
        input_ids = tuple(pair_seeds[seed].artifact_id for seed in seeds)
        if len(seeds) < statistics_config.minimum_valid_paired_seeds:
            coupling_contrasts[pair] = (paired_seed_count, None, None, None, None, input_ids)
            continue
        gap_values = tuple(pair_seeds[seed].value for seed in seeds)
        zero_reference = tuple(0.0 for _ in gap_values)
        coupling_bootstrap_seed = statistical_bootstrap_seed(
            ContrastName(f"Exact correspondence orbit vs Matched-Resource Rectangular: {pair}"),
            MultiplicityFamily.COUPLING_MECHANISM,
            DirectedPairName(pair),
            MetricId.ROBUST_COUPLING_VALUE_GAP,
            BootstrapPurpose("coupling-mechanism-gap"),
        )
        bca = paired_bca_interval(gap_values, zero_reference, coupling_bootstrap_seed)
        sign_flip = exact_sign_flip_test(gap_values, zero_reference)
        coupling_raw_p_by_pair[pair] = sign_flip.p_value
        coupling_metadata_inputs[pair] = (
            sign_flip.nonzero_difference_count,
            coupling_bootstrap_seed,
        )
        coupling_contrasts[pair] = (
            paired_seed_count,
            float(sign_flip.mean_difference),
            float(sign_flip.median_difference),
            None if bca.lower is None else float(bca.lower),
            None if bca.upper is None else float(bca.upper),
            input_ids,
        )
    coupling_holm_adjusted = holm_step_down(
        PValueSet(
            tuple(
                NamedPValue(PValueName(pair), p_value)
                for pair, p_value in coupling_raw_p_by_pair.items()
            )
        )
    )
    coupling_criteria = active_config().scientific.evaluation_criteria.coupling_mechanism
    for pair in coupling_pairs:
        (
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            input_ids,
        ) = coupling_contrasts[pair]
        if mean_difference is None:
            decision = ComparisonDecision.INSUFFICIENT_EVIDENCE
            raw_p = None
            holm_p = None
        else:
            raw_p = coupling_raw_p_by_pair[pair]
            holm_p = coupling_holm_adjusted.value_of(PValueName(pair))
            if bca_low is None:
                decision = ComparisonDecision.DEGENERATE
            elif (
                holm_p is not None
                and holm_p <= coupling_criteria.holm_adjusted_p_maximum
                and bca_low > 0.0
            ):
                decision = ComparisonDecision.SUPERIOR
            else:
                decision = ComparisonDecision.NOT_SUPPORTED
        if not input_ids:
            continue
        coupling_comparison_manifest = persist_coupling_mechanism_comparison(
            store,
            layout,
            request.experiment,
            DirectedPairName(pair),
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            raw_p,
            float(holm_p) if holm_p is not None else None,
            decision,
            input_ids,
            request.overwrite_policy,
        )
        if coupling_comparison_manifest is not None and pair in coupling_metadata_inputs:
            resolved_coupling_comparison = store.resolve(coupling_comparison_manifest.artifact_id)
            coupling_comparison_record = PairedComparisonRecord.model_validate(
                json.loads(Path(resolved_coupling_comparison.payload_paths[0]).read_text())[
                    "comparison_record" #TODO: should be enum instead of hardcoded string
                ]
            )
            coupling_nonzero_count, coupling_metadata_seed = coupling_metadata_inputs[pair]
            coupling_family_size: SampleCount = len(coupling_raw_p_by_pair)
            persist_statistical_metadata(
                store,
                layout,
                request.experiment,
                DirectedPairName(pair),
                TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                coupling_comparison_record,
                ComparisonStatistic.SIGN_FLIP_AGAINST_ZERO,
                coupling_nonzero_count,
                coupling_metadata_seed,
                _holm_rank(coupling_raw_p_by_pair, pair),
                coupling_family_size,
                request.overwrite_policy,
            )
    external_source_criteria = (
        active_config().scientific.evaluation_criteria.external_source_value_vs_local_sir
    )
    equivalence_margins = active_config().scientific.materiality.equivalence_relative_macro_ce
    external_source_pairs = sorted(
        {
            pair
            for pair, method, _ in metrics
            if method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
        }
    )
    external_source_raw_p: OrderedDict[str, float] = OrderedDict() #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    external_source_contrasts: OrderedDict[ #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        tuple[ #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            Index,
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            tuple[ArtifactIdentifier, ...],
            Index,
            RandomSeed,
        ],
    ] = OrderedDict()
    for pair in external_source_pairs:
        local_sir_seeds = OrderedDict(
            (seed, entry)
            for (candidate_pair, candidate_method, seed), entry in metrics.items()
            if candidate_pair == pair and candidate_method == TransferMethod.LOCAL_SIR
        )
        fedorbit_seeds = OrderedDict(
            (seed, entry)
            for (candidate_pair, candidate_method, seed), entry in metrics.items()
            if candidate_pair == pair
            and candidate_method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
        )
        shared_seeds = sorted(set(local_sir_seeds) & set(fedorbit_seeds))
        paired_seed_count: Index = len(shared_seeds)
        input_ids = tuple(
            identifier
            for seed in shared_seeds
            for identifier in (
                local_sir_seeds[seed].artifact_id,
                fedorbit_seeds[seed].artifact_id,
            )
        )
        if len(shared_seeds) < statistics_config.minimum_valid_paired_seeds:
            placeholder_count: Index = 0
            placeholder_seed: RandomSeed = 0
            external_source_contrasts[pair] = (
                paired_seed_count,
                None,
                None,
                None,
                None,
                None,
                input_ids,
                placeholder_count,
                placeholder_seed,
            )
            continue
        local_sir_values = tuple(local_sir_seeds[seed].value for seed in shared_seeds)
        fedorbit_values = tuple(fedorbit_seeds[seed].value for seed in shared_seeds)
        bootstrap_seed = statistical_bootstrap_seed(
            ContrastName("FedORBIT Exact-Sparse Solver vs Local-SIR"),
            MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR,
            DirectedPairName(pair),
            MetricId.MACRO_CROSS_ENTROPY,
            BootstrapPurpose("external-source-vs-local-sir-gain"),
        )
        bca = paired_bca_interval(local_sir_values, fedorbit_values, bootstrap_seed)
        sign_flip = exact_sign_flip_test(local_sir_values, fedorbit_values)
        tost = tost_equivalence(fedorbit_values, local_sir_values)
        external_source_raw_p[f"{pair}|superiority"] = sign_flip.p_value
        external_source_raw_p[f"{pair}|equivalence"] = tost.p_equiv
        external_source_contrasts[pair] = (
            paired_seed_count,
            float(sign_flip.mean_difference),
            float(sign_flip.median_difference),
            None if bca.lower is None else float(bca.lower),
            None if bca.upper is None else float(bca.upper),
            float(tost.p_equiv),
            input_ids,
            sign_flip.nonzero_difference_count,
            bootstrap_seed,
        )
    external_source_holm = holm_step_down(
        PValueSet(
            tuple(
                NamedPValue(PValueName(name), p_value)
                for name, p_value in external_source_raw_p.items()
            )
        )
    )
    for pair in external_source_pairs:
        (
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            p_equiv,
            input_ids,
            nonzero_count,
            bootstrap_seed,
        ) = external_source_contrasts[pair]
        if not input_ids:
            continue
        if mean_difference is None:
            superiority_decision = ComparisonDecision.INSUFFICIENT_EVIDENCE
            equivalence_decision = ComparisonDecision.INSUFFICIENT_EVIDENCE
            raw_p = None
            holm_p = None
            equivalence_holm_p = None
        else:
            raw_p = external_source_raw_p[f"{pair}|superiority"]
            holm_p = external_source_holm.value_of(PValueName(f"{pair}|superiority"))
            equivalence_holm_p = external_source_holm.value_of(PValueName(f"{pair}|equivalence"))
            if bca_low is None:
                superiority_decision = ComparisonDecision.DEGENERATE
            elif (
                holm_p is not None
                and holm_p <= external_source_criteria.holm_adjusted_p_maximum
                and bca_low > external_source_criteria.bca_lower_bound_strictly_greater_than
            ):
                superiority_decision = ComparisonDecision.SUPERIOR
            else:
                superiority_decision = ComparisonDecision.NOT_SUPPORTED
            if (
                equivalence_holm_p is not None
                and equivalence_holm_p <= statistics_config.tost_alpha_per_one_sided_test
            ):
                equivalence_decision = ComparisonDecision.EQUIVALENT
            else:
                equivalence_decision = ComparisonDecision.NOT_SUPPORTED
        realized_relative_macro_ce = (
            active_config().scientific.materiality.realized_relative_macro_ce
        )
        superiority_manifest = persist_baseline_comparison(
            store,
            layout,
            request.experiment,
            DirectedPairName(pair),
            MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            TransferMethod.LOCAL_SIR,
            MetricId.MACRO_CROSS_ENTROPY,
            ContrastName(
                "FedORBIT Exact-Sparse Solver vs Local-SIR"
                " — TEST relative macro-CE gain superiority"
            ),
            realized_relative_macro_ce,
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            raw_p,
            float(holm_p) if holm_p is not None else None,
            superiority_decision,
            None,
            None,
            input_ids,
            request.overwrite_policy,
        )
        equivalence_manifest = persist_baseline_comparison(
            store,
            layout,
            request.experiment,
            DirectedPairName(pair),
            MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            TransferMethod.LOCAL_SIR,
            MetricId.MACRO_CROSS_ENTROPY,
            ContrastName(
                "FedORBIT Exact-Sparse Solver vs Local-SIR"
                f" — TEST relative macro-CE gain {ComparisonContrastSuffix.TOST_EQUIVALENCE.value}"
            ),
            realized_relative_macro_ce,
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            float(p_equiv) if p_equiv is not None else None,
            float(equivalence_holm_p) if equivalence_holm_p is not None else None,
            equivalence_decision,
            equivalence_margins.lower,
            equivalence_margins.upper,
            input_ids,
            request.overwrite_policy,
        )
        family_size: SampleCount = len(external_source_raw_p)
        for manifest, statistic, seed_value in (
            (superiority_manifest, ComparisonStatistic.SIGN_FLIP_SUPERIORITY, bootstrap_seed),
            (equivalence_manifest, ComparisonStatistic.TOST_EQUIVALENCE, bootstrap_seed),
        ):
            if manifest is None or mean_difference is None:
                continue
            resolved = store.resolve(manifest.artifact_id)
            comparison_record = PairedComparisonRecord.model_validate(
                json.loads(Path(resolved.payload_paths[0]).read_text())["comparison_record"] #TODO: should be enum instead of hardcoded string
            )
            key = (
                "superiority" #TODO: should be enum instead of hardcoded string
                if statistic == ComparisonStatistic.SIGN_FLIP_SUPERIORITY
                else "equivalence" #TODO: should be enum instead of hardcoded string
            )
            persist_statistical_metadata(
                store,
                layout,
                request.experiment,
                DirectedPairName(pair),
                TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                comparison_record,
                statistic,
                nonzero_count,
                seed_value,
                _holm_rank(external_source_raw_p, f"{pair}|{key}"),
                family_size,
                request.overwrite_policy,
            )
    point_correspondence_pairs = sorted(
        {
            pair
            for pair, method, _ in metrics
            if method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
        }
    )
    point_correspondence_raw_p: OrderedDict[str, float] = OrderedDict() #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    point_correspondence_contrasts: OrderedDict[ #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        tuple[ #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            Index,
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            tuple[ArtifactIdentifier, ...],
            Index,
            RandomSeed,
        ],
    ] = OrderedDict()
    for pair in point_correspondence_pairs:
        commitment_seeds = OrderedDict(
            (seed, entry)
            for (candidate_pair, candidate_method, seed), entry in metrics.items()
            if candidate_pair == pair
            and candidate_method == TransferMethod.POINT_CORRESPONDENCE_COMMITMENT
        )
        fedorbit_seeds = OrderedDict(
            (seed, entry)
            for (candidate_pair, candidate_method, seed), entry in metrics.items()
            if candidate_pair == pair
            and candidate_method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
        )
        shared_seeds = sorted(set(commitment_seeds) & set(fedorbit_seeds))
        paired_seed_count: Index = len(shared_seeds)
        input_ids = tuple(
            identifier
            for seed in shared_seeds
            for identifier in (
                commitment_seeds[seed].artifact_id,
                fedorbit_seeds[seed].artifact_id,
            )
        )
        if len(shared_seeds) < statistics_config.minimum_valid_paired_seeds:
            placeholder_count: Index = 0
            placeholder_seed: RandomSeed = 0
            point_correspondence_contrasts[pair] = (
                paired_seed_count,
                None,
                None,
                None,
                None,
                None,
                input_ids,
                placeholder_count,
                placeholder_seed,
            )
            continue
        commitment_values = tuple(commitment_seeds[seed].value for seed in shared_seeds)
        fedorbit_values = tuple(fedorbit_seeds[seed].value for seed in shared_seeds)
        bootstrap_seed = statistical_bootstrap_seed(
            ContrastName("FedORBIT Exact-Sparse Solver vs Point-Correspondence Commitment"),
            MultiplicityFamily.POINT_CORRESPONDENCE_SAFETY,
            DirectedPairName(pair),
            MetricId.MACRO_CROSS_ENTROPY,
            BootstrapPurpose("point-correspondence-safety-difference"),
        )
        bca = paired_bca_interval(commitment_values, fedorbit_values, bootstrap_seed)
        sign_flip = exact_sign_flip_test(commitment_values, fedorbit_values)
        tost = tost_equivalence(fedorbit_values, commitment_values)
        point_correspondence_raw_p[f"{pair}|difference"] = sign_flip.p_value
        point_correspondence_raw_p[f"{pair}|equivalence"] = tost.p_equiv
        point_correspondence_contrasts[pair] = (
            paired_seed_count,
            float(sign_flip.mean_difference),
            float(sign_flip.median_difference),
            None if bca.lower is None else float(bca.lower),
            None if bca.upper is None else float(bca.upper),
            float(tost.p_equiv),
            input_ids,
            sign_flip.nonzero_difference_count,
            bootstrap_seed,
        )
    point_correspondence_holm = holm_step_down(
        PValueSet(
            tuple(
                NamedPValue(PValueName(name), p_value)
                for name, p_value in point_correspondence_raw_p.items()
            )
        )
    )
    for pair in point_correspondence_pairs:
        (
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            p_equiv,
            input_ids,
            nonzero_count,
            bootstrap_seed,
        ) = point_correspondence_contrasts[pair]
        if not input_ids:
            continue
        if mean_difference is None:
            difference_decision = ComparisonDecision.INSUFFICIENT_EVIDENCE
            equivalence_decision = ComparisonDecision.INSUFFICIENT_EVIDENCE
            raw_p = None
            holm_p = None
            equivalence_holm_p = None
        else:
            raw_p = point_correspondence_raw_p[f"{pair}|difference"]
            holm_p = point_correspondence_holm.value_of(PValueName(f"{pair}|difference"))
            equivalence_holm_p = point_correspondence_holm.value_of(
                PValueName(f"{pair}|equivalence")
            )
            difference_decision = (
                ComparisonDecision.DEGENERATE
                if bca_low is None
                else ComparisonDecision.NOT_SUPPORTED
            )
            if (
                equivalence_holm_p is not None
                and equivalence_holm_p <= statistics_config.tost_alpha_per_one_sided_test
            ):
                equivalence_decision = ComparisonDecision.EQUIVALENT
            else:
                equivalence_decision = ComparisonDecision.NOT_SUPPORTED
        difference_manifest = persist_baseline_comparison(
            store,
            layout,
            request.experiment,
            DirectedPairName(pair),
            MultiplicityFamily.POINT_CORRESPONDENCE_SAFETY,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
            MetricId.MACRO_CROSS_ENTROPY,
            ContrastName(
                "FedORBIT Exact-Sparse Solver vs Point-Correspondence Commitment"
                f" — TEST relative macro-CE {ComparisonContrastSuffix.DIFFERENCE.value}"
            ),
            None,
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            raw_p,
            float(holm_p) if holm_p is not None else None,
            difference_decision,
            None,
            None,
            input_ids,
            request.overwrite_policy,
        )
        point_equivalence_manifest = persist_baseline_comparison(
            store,
            layout,
            request.experiment,
            DirectedPairName(pair),
            MultiplicityFamily.POINT_CORRESPONDENCE_SAFETY,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
            MetricId.MACRO_CROSS_ENTROPY,
            ContrastName(
                "FedORBIT Exact-Sparse Solver vs Point-Correspondence Commitment"
                f" — TEST relative macro-CE {ComparisonContrastSuffix.TOST_EQUIVALENCE.value}"
            ),
            None,
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            float(p_equiv) if p_equiv is not None else None,
            float(equivalence_holm_p) if equivalence_holm_p is not None else None,
            equivalence_decision,
            equivalence_margins.lower,
            equivalence_margins.upper,
            input_ids,
            request.overwrite_policy,
        )
        point_correspondence_family_size: SampleCount = len(point_correspondence_raw_p)
        for manifest, statistic, seed_value in (
            (
                difference_manifest,
                ComparisonStatistic.SIGN_FLIP_DIFFERENCE_COMMON_REFERENCE,
                bootstrap_seed,
            ),
            (point_equivalence_manifest, ComparisonStatistic.TOST_EQUIVALENCE, bootstrap_seed),
        ):
            if manifest is None or mean_difference is None:
                continue
            resolved = store.resolve(manifest.artifact_id)
            comparison_record = PairedComparisonRecord.model_validate(
                json.loads(Path(resolved.payload_paths[0]).read_text())["comparison_record"] #TODO: should be enum instead of hardcoded string
            )
            key = (
                "difference" #TODO: Use a proper enum instead of a raw string
                if statistic == ComparisonStatistic.SIGN_FLIP_DIFFERENCE_COMMON_REFERENCE
                else "equivalence" #TODO: Use a proper enum instead of a raw string
            )
            persist_statistical_metadata(
                store,
                layout,
                request.experiment,
                DirectedPairName(pair),
                TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                comparison_record,
                statistic,
                nonzero_count,
                seed_value,
                _holm_rank(point_correspondence_raw_p, f"{pair}|{key}"),
                point_correspondence_family_size,
                request.overwrite_policy,
            )
    _execute_ablation_and_sparsity_and_confirmation_statistical_synthesis(store, layout, request)


def _execute_ablation_and_sparsity_and_confirmation_statistical_synthesis(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    statistics_config = active_config().scientific.statistics
    equivalence_margins = active_config().scientific.materiality.equivalence_relative_macro_ce
    local_only_metrics = _completed_primary_transfer_macro_ce(store)

    ablation_metrics = _completed_condition_macro_ce(store, ExperimentName.MECHANISM_ABLATIONS)
    ablation_pairs = sorted(
        {
            pair
            for pair, method, condition, _ in ablation_metrics
            if method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
            and condition == _PRINCIPAL_CONDITION
        }
    )
    ablation_raw_p: OrderedDict[str, float] = OrderedDict() #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    ablation_contrasts: OrderedDict[ #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        tuple[ #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            Index,
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            tuple[ArtifactIdentifier, ...],
            Index,
            RandomSeed,
        ],
    ] = OrderedDict()
    for pair in ablation_pairs:
        full_seeds = OrderedDict(
            (seed, entry)
            for (candidate_pair, method, condition, seed), entry in ablation_metrics.items()
            if candidate_pair == pair
            and method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
            and condition == _PRINCIPAL_CONDITION
        )
        destroyed_seeds = OrderedDict(
            (seed, entry)
            for (candidate_pair, method, condition, seed), entry in ablation_metrics.items()
            if candidate_pair == pair
            and method == TransferMethod.COUPLING_DESTROYED_FEDORBIT
            and condition == _PRINCIPAL_CONDITION
        )
        shared_seeds = sorted(set(full_seeds) & set(destroyed_seeds))
        paired_seed_count: Index = len(shared_seeds)
        input_ids = tuple(
            identifier
            for seed in shared_seeds
            for identifier in (full_seeds[seed].artifact_id, destroyed_seeds[seed].artifact_id)
        )
        if len(shared_seeds) < statistics_config.minimum_valid_paired_seeds:
            placeholder_count: Index = 0
            placeholder_seed: RandomSeed = 0
            ablation_contrasts[pair] = (
                paired_seed_count,
                None,
                None,
                None,
                None,
                None,
                input_ids,
                placeholder_count,
                placeholder_seed,
            )
            continue
        full_values = tuple(full_seeds[seed].value for seed in shared_seeds)
        destroyed_values = tuple(destroyed_seeds[seed].value for seed in shared_seeds)
        bootstrap_seed = statistical_bootstrap_seed(
            ContrastName("FedORBIT Exact-Sparse Solver vs Coupling-Destroyed FedORBIT"),
            MultiplicityFamily.MECHANISM_ABLATIONS,
            DirectedPairName(pair),
            MetricId.MACRO_CROSS_ENTROPY,
            BootstrapPurpose("mechanism-ablations-difference"),
        )
        bca = paired_bca_interval(destroyed_values, full_values, bootstrap_seed)
        sign_flip = exact_sign_flip_test(destroyed_values, full_values)
        tost = tost_equivalence(full_values, destroyed_values)
        ablation_raw_p[f"{pair}|difference"] = sign_flip.p_value
        ablation_raw_p[f"{pair}|equivalence"] = tost.p_equiv
        ablation_contrasts[pair] = (
            paired_seed_count,
            float(sign_flip.mean_difference),
            float(sign_flip.median_difference),
            None if bca.lower is None else float(bca.lower),
            None if bca.upper is None else float(bca.upper),
            float(tost.p_equiv),
            input_ids,
            sign_flip.nonzero_difference_count,
            bootstrap_seed,
        )
    ablation_holm = holm_step_down(
        PValueSet(
            tuple(
                NamedPValue(PValueName(name), p_value) for name, p_value in ablation_raw_p.items()
            )
        )
    )
    for pair in ablation_pairs:
        (
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            p_equiv,
            input_ids,
            nonzero_count,
            bootstrap_seed,
        ) = ablation_contrasts[pair]
        if not input_ids:
            continue
        if mean_difference is None:
            difference_decision = ComparisonDecision.INSUFFICIENT_EVIDENCE
            equivalence_decision = ComparisonDecision.INSUFFICIENT_EVIDENCE
            raw_p = None
            holm_p = None
            equivalence_holm_p = None
        else:
            raw_p = ablation_raw_p[f"{pair}|difference"]
            holm_p = ablation_holm.value_of(PValueName(f"{pair}|difference"))
            equivalence_holm_p = ablation_holm.value_of(PValueName(f"{pair}|equivalence"))
            difference_decision = (
                ComparisonDecision.DEGENERATE
                if bca_low is None
                else ComparisonDecision.NOT_SUPPORTED
            )
            if (
                equivalence_holm_p is not None
                and equivalence_holm_p <= statistics_config.tost_alpha_per_one_sided_test
            ):
                equivalence_decision = ComparisonDecision.EQUIVALENT
            else:
                equivalence_decision = ComparisonDecision.NOT_SUPPORTED
        difference_manifest = persist_baseline_comparison(
            store,
            layout,
            request.experiment,
            DirectedPairName(pair),
            MultiplicityFamily.MECHANISM_ABLATIONS,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            TransferMethod.COUPLING_DESTROYED_FEDORBIT,
            MetricId.MACRO_CROSS_ENTROPY,
            ContrastName(
                "FedORBIT Exact-Sparse Solver vs Coupling-Destroyed FedORBIT"
                f" — TEST relative macro-CE {ComparisonContrastSuffix.DIFFERENCE.value}"
            ),
            None,
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            raw_p,
            float(holm_p) if holm_p is not None else None,
            difference_decision,
            None,
            None,
            input_ids,
            request.overwrite_policy,
        )
        ablation_equivalence_manifest = persist_baseline_comparison(
            store,
            layout,
            request.experiment,
            DirectedPairName(pair),
            MultiplicityFamily.MECHANISM_ABLATIONS,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            TransferMethod.COUPLING_DESTROYED_FEDORBIT,
            MetricId.MACRO_CROSS_ENTROPY,
            ContrastName(
                "FedORBIT Exact-Sparse Solver vs Coupling-Destroyed FedORBIT"
                f" — TEST relative macro-CE {ComparisonContrastSuffix.TOST_EQUIVALENCE.value}"
            ),
            None,
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            float(p_equiv) if p_equiv is not None else None,
            float(equivalence_holm_p) if equivalence_holm_p is not None else None,
            equivalence_decision,
            equivalence_margins.lower,
            equivalence_margins.upper,
            input_ids,
            request.overwrite_policy,
        )
        ablation_family_size: SampleCount = len(ablation_raw_p)
        for manifest, statistic, seed_value in (
            (
                difference_manifest,
                ComparisonStatistic.SIGN_FLIP_DIFFERENCE_COMMON_REFERENCE,
                bootstrap_seed,
            ),
            (ablation_equivalence_manifest, ComparisonStatistic.TOST_EQUIVALENCE, bootstrap_seed),
        ):
            if manifest is None or mean_difference is None:
                continue
            resolved = store.resolve(manifest.artifact_id)
            comparison_record = PairedComparisonRecord.model_validate(
                json.loads(Path(resolved.payload_paths[0]).read_text())["comparison_record"] #TODO: should be enum instead of hardcoded string
            )
            key = (
                "difference" #TODO: should be enum instead of hardcoded string
                if statistic == ComparisonStatistic.SIGN_FLIP_DIFFERENCE_COMMON_REFERENCE
                else "equivalence" #TODO: should be enum instead of hardcoded string
            )
            persist_statistical_metadata(
                store,
                layout,
                request.experiment,
                DirectedPairName(pair),
                TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                comparison_record,
                statistic,
                nonzero_count,
                seed_value,
                _holm_rank(ablation_raw_p, f"{pair}|{key}"),
                ablation_family_size,
                request.overwrite_policy,
            )

    sparsity_metrics = _completed_condition_macro_ce(
        store, ExperimentName.SPARSITY_AND_DENSE_FALLBACK
    )
    sparsity_pairs = sorted({pair for pair, _, _, _ in sparsity_metrics})
    condition_pairs: tuple[tuple[EvaluationConditionName, EvaluationConditionName], ...] = (
        (
            EvaluationConditionName("exact sparse s=1"), #TODO: should be enum not hardcoded strings
            EvaluationConditionName("exact sparse s=2"), #TODO: should be enum not hardcoded strings
        ),
        (
            EvaluationConditionName("exact sparse s=3"), #TODO: should be enum not hardcoded strings
            EvaluationConditionName("exact sparse s=2"), #TODO: should be enum not hardcoded strings
        ),
        (EvaluationConditionName("dense CCP"), #TODO: should be enum not hardcoded strings
         EvaluationConditionName("exact sparse s=2")), #TODO: should be enum not hardcoded strings
    )
    sparsity_condition_method: Mapping[EvaluationConditionName, MethodName] = OrderedDict(
        (
            (
                EvaluationConditionName("exact sparse s=1"), #TODO: should be enum not hardcoded strings
                ExperimentLocalMethod.EXACT_SPARSE_SUPPORT_ONE,
            ),
            (
                EvaluationConditionName("exact sparse s=2"), #TODO: should be enum not hardcoded strings
                ExperimentLocalMethod.EXACT_SPARSE_SUPPORT_TWO,
            ),
            (
                EvaluationConditionName("exact sparse s=3"), #TODO: should be enum not hardcoded strings
                ExperimentLocalMethod.EXACT_SPARSE_SUPPORT_THREE,
            ),
            (EvaluationConditionName("dense CCP"),  #TODO: should be enum not hardcoded strings
             TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK),
        )
    )
    sparsity_raw_p: OrderedDict[str, float] = OrderedDict() #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    sparsity_contrasts: OrderedDict[ #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        tuple[ #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            Index,
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            tuple[ArtifactIdentifier, ...],
            Index,
            RandomSeed,
        ],
    ] = OrderedDict()
    for pair in sparsity_pairs:
        local_only_seeds = OrderedDict(
            (seed, entry)
            for (candidate_pair, method, seed), entry in local_only_metrics.items()
            if candidate_pair == pair and method == TransferMethod.LOCAL_ONLY
        )
        for condition_a, condition_b in condition_pairs:
            contrast_key = f"{pair}|{condition_a}|{condition_b}"
            seeds_a = OrderedDict(
                (seed, entry)
                for (candidate_pair, _, condition, seed), entry in sparsity_metrics.items()
                if candidate_pair == pair and condition == condition_a
            )
            seeds_b = OrderedDict(
                (seed, entry)
                for (candidate_pair, _, condition, seed), entry in sparsity_metrics.items()
                if candidate_pair == pair and condition == condition_b
            )
            shared_seeds = sorted(set(local_only_seeds) & set(seeds_a) & set(seeds_b))
            paired_seed_count = len(shared_seeds)
            input_ids = tuple(
                identifier
                for seed in shared_seeds
                for identifier in (
                    local_only_seeds[seed].artifact_id,
                    seeds_a[seed].artifact_id,
                    seeds_b[seed].artifact_id,
                )
            )
            if len(shared_seeds) < statistics_config.minimum_valid_paired_seeds:
                sparsity_placeholder_count: Index = 0
                sparsity_placeholder_seed: RandomSeed = 0
                sparsity_contrasts[contrast_key] = (
                    paired_seed_count,
                    None,
                    None,
                    None,
                    None,
                    input_ids,
                    sparsity_placeholder_count,
                    sparsity_placeholder_seed,
                )
                continue
            gain_a = tuple(
                (local_only_seeds[seed].value - seeds_a[seed].value) / local_only_seeds[seed].value
                for seed in shared_seeds
            )
            gain_b = tuple(
                (local_only_seeds[seed].value - seeds_b[seed].value) / local_only_seeds[seed].value
                for seed in shared_seeds
            )
            bootstrap_seed = statistical_bootstrap_seed(
                ContrastName(f"{condition_a} vs {condition_b}"),
                MultiplicityFamily.SPARSITY_SENSITIVITY,
                DirectedPairName(pair),
                MetricId.RELATIVE_MACRO_CE_GAIN,
                BootstrapPurpose("sparsity-sensitivity-gain-difference"),
            )
            bca = paired_bca_interval(gain_a, gain_b, bootstrap_seed)
            sign_flip = exact_sign_flip_test(gain_a, gain_b)
            sparsity_raw_p[contrast_key] = sign_flip.p_value
            sparsity_contrasts[contrast_key] = (
                paired_seed_count,
                float(sign_flip.mean_difference),
                float(sign_flip.median_difference),
                None if bca.lower is None else float(bca.lower),
                None if bca.upper is None else float(bca.upper),
                input_ids,
                sign_flip.nonzero_difference_count,
                bootstrap_seed,
            )
    sparsity_holm = holm_step_down(
        PValueSet(
            tuple(
                NamedPValue(PValueName(name), p_value) for name, p_value in sparsity_raw_p.items()
            )
        )
    )
    sparsity_family_size: SampleCount = len(sparsity_raw_p)
    for pair in sparsity_pairs:
        for condition_a, condition_b in condition_pairs:
            contrast_key = f"{pair}|{condition_a}|{condition_b}"
            if contrast_key not in sparsity_contrasts:
                continue
            (
                paired_seed_count,
                mean_difference,
                median_difference,
                bca_low,
                bca_high,
                input_ids,
                sparsity_nonzero_count,
                sparsity_bootstrap_seed,
            ) = sparsity_contrasts[contrast_key]
            if not input_ids:
                continue
            if mean_difference is None:
                decision = ComparisonDecision.INSUFFICIENT_EVIDENCE
                raw_p = None
                holm_p = None
            else:
                raw_p = sparsity_raw_p[contrast_key]
                holm_p = sparsity_holm.value_of(PValueName(contrast_key))
                decision = (
                    ComparisonDecision.DEGENERATE
                    if bca_low is None
                    else ComparisonDecision.NOT_SUPPORTED
                )
            manifest = persist_baseline_comparison(
                store,
                layout,
                request.experiment,
                DirectedPairName(pair),
                MultiplicityFamily.SPARSITY_SENSITIVITY,
                sparsity_condition_method[condition_a],
                sparsity_condition_method[condition_b],
                MetricId.RELATIVE_MACRO_CE_GAIN,
                ContrastName(f"{condition_a} vs {condition_b} — TEST relative macro-CE difference"),
                None,
                paired_seed_count,
                mean_difference,
                median_difference,
                bca_low,
                bca_high,
                raw_p,
                float(holm_p) if holm_p is not None else None,
                decision,
                None,
                None,
                input_ids,
                request.overwrite_policy,
            )
            if manifest is None or mean_difference is None:
                continue
            resolved = store.resolve(manifest.artifact_id)
            comparison_record = PairedComparisonRecord.model_validate(
                json.loads(Path(resolved.payload_paths[0]).read_text())["comparison_record"] #TODO: should be enum instead of hardcoded string
            )
            persist_statistical_metadata(
                store,
                layout,
                request.experiment,
                DirectedPairName(pair),
                TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                comparison_record,
                ComparisonStatistic.SIGN_FLIP_DIFFERENCE_COMMON_REFERENCE,
                sparsity_nonzero_count,
                sparsity_bootstrap_seed,
                _holm_rank(sparsity_raw_p, contrast_key),
                sparsity_family_size,
                request.overwrite_policy,
            )

    confirmation_criteria = active_config().scientific.evaluation_criteria.confirmation_safety
    harmful_threshold = (
        active_config().scientific.materiality.harmful_transfer_relative_macro_ce_gain
    )
    confirmation_metrics = _completed_condition_macro_ce(
        store, ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY
    )
    confirmation_pairs = sorted(
        {
            pair
            for pair, method, condition, _ in confirmation_metrics
            if method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
            and condition == _PRINCIPAL_CONDITION
        }
    )
    confirmation_raw_p: OrderedDict[str, float] = OrderedDict() #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    confirmation_contrasts: OrderedDict[ #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        str, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        tuple[ #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            Index,
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            tuple[ArtifactIdentifier, ...],
            float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
            Index,
            RandomSeed,
        ],
    ] = OrderedDict()
    for pair in confirmation_pairs:
        local_only_seeds = OrderedDict(
            (seed, entry)
            for (candidate_pair, method, seed), entry in local_only_metrics.items()
            if candidate_pair == pair and method == TransferMethod.LOCAL_ONLY
        )
        with_confirm_seeds = OrderedDict(
            (seed, entry)
            for (candidate_pair, method, condition, seed), entry in confirmation_metrics.items()
            if candidate_pair == pair
            and method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
            and condition == _PRINCIPAL_CONDITION
        )
        without_confirm_seeds = OrderedDict(
            (seed, entry)
            for (candidate_pair, method, condition, seed), entry in confirmation_metrics.items()
            if candidate_pair == pair
            and method == TransferMethod.FEDORBIT_WITHOUT_CONFIRMATION
            and condition == _PRINCIPAL_CONDITION
        )
        shared_seeds = sorted(
            set(local_only_seeds) & set(with_confirm_seeds) & set(without_confirm_seeds)
        )
        paired_seed_count = len(shared_seeds)
        input_ids = tuple(
            identifier
            for seed in shared_seeds
            for identifier in (
                local_only_seeds[seed].artifact_id,
                with_confirm_seeds[seed].artifact_id,
                without_confirm_seeds[seed].artifact_id,
            )
        )
        if len(shared_seeds) < statistics_config.minimum_valid_paired_seeds:
            confirmation_placeholder_count: Index = 0
            confirmation_placeholder_seed: RandomSeed = 0
            confirmation_contrasts[pair] = (
                paired_seed_count,
                None,
                None,
                None,
                None,
                input_ids,
                None,
                confirmation_placeholder_count,
                confirmation_placeholder_seed,
            )
            continue
        harmful_with_values: list[float] = [] #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        for seed in shared_seeds:
            gain: RelativeGain = (
                local_only_seeds[seed].value - with_confirm_seeds[seed].value
            ) / local_only_seeds[seed].value
            harmful_with_values.append(1.0 if harm_indicator(gain, harmful_threshold) else 0.0)
        harmful_with = tuple(harmful_with_values)
        harmful_without_values: list[float] = [] #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
        for seed in shared_seeds:
            gain: RelativeGain = (
                local_only_seeds[seed].value - without_confirm_seeds[seed].value
            ) / local_only_seeds[seed].value
            harmful_without_values.append(1.0 if harm_indicator(gain, harmful_threshold) else 0.0)
        harmful_without = tuple(harmful_without_values)
        harm_rate_no_confirm = statistics.fmean(harmful_without)
        bootstrap_seed = statistical_bootstrap_seed(
            ContrastName(
                "FedORBIT Without Confirmation vs FedORBIT Exact-Sparse Solver with confirmation"
            ),
            MultiplicityFamily.CONFIRMATION_SAFETY,
            DirectedPairName(pair),
            MetricId.ABSOLUTE_RISK_REDUCTION,
            BootstrapPurpose("confirmation-safety-harm-rate-difference"),
        )
        bca = paired_bca_interval(harmful_without, harmful_with, bootstrap_seed)
        sign_flip = exact_sign_flip_test(harmful_without, harmful_with)
        confirmation_raw_p[pair] = sign_flip.p_value
        confirmation_contrasts[pair] = (
            paired_seed_count,
            float(sign_flip.mean_difference),
            float(sign_flip.median_difference),
            None if bca.lower is None else float(bca.lower),
            None if bca.upper is None else float(bca.upper),
            input_ids,
            harm_rate_no_confirm,
            sign_flip.nonzero_difference_count,
            bootstrap_seed,
        )
    confirmation_holm = holm_step_down(
        PValueSet(
            tuple(
                NamedPValue(PValueName(pair), p_value)
                for pair, p_value in confirmation_raw_p.items()
            )
        )
    )
    confirmation_family_size: SampleCount = len(confirmation_raw_p)
    for pair in confirmation_pairs:
        (
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            input_ids,
            harm_rate_no_confirm,
            confirmation_nonzero_count,
            confirmation_bootstrap_seed,
        ) = confirmation_contrasts[pair]
        if not input_ids:
            continue
        if mean_difference is None:
            decision = ComparisonDecision.INSUFFICIENT_EVIDENCE
            raw_p = None
            holm_p = None
        else:
            raw_p = confirmation_raw_p[pair]
            holm_p = confirmation_holm.value_of(PValueName(pair))
            if bca_low is None:
                decision = ComparisonDecision.DEGENERATE
            else:
                arr = mean_difference
                rrr = (
                    arr / harm_rate_no_confirm
                    if harm_rate_no_confirm is not None and harm_rate_no_confirm > 0.0
                    else None
                )
                if arr >= confirmation_criteria.absolute_risk_reduction_minimum or (
                    rrr is not None and rrr >= confirmation_criteria.relative_risk_reduction_minimum
                ):
                    decision = ComparisonDecision.SUPERIOR
                else:
                    decision = ComparisonDecision.NOT_SUPPORTED
        manifest = persist_baseline_comparison(
            store,
            layout,
            request.experiment,
            DirectedPairName(pair),
            MultiplicityFamily.CONFIRMATION_SAFETY,
            TransferMethod.FEDORBIT_WITHOUT_CONFIRMATION,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            MetricId.ABSOLUTE_RISK_REDUCTION,
            ContrastName(
                "FedORBIT Without Confirmation vs FedORBIT Exact-Sparse Solver with confirmation"
                " — harmful-transfer rate difference"
            ),
            confirmation_criteria.absolute_risk_reduction_minimum,
            paired_seed_count,
            mean_difference,
            median_difference,
            bca_low,
            bca_high,
            raw_p,
            float(holm_p) if holm_p is not None else None,
            decision,
            None,
            None,
            input_ids,
            request.overwrite_policy,
        )
        if manifest is None or mean_difference is None:
            continue
        resolved = store.resolve(manifest.artifact_id)
        comparison_record = PairedComparisonRecord.model_validate(
            json.loads(Path(resolved.payload_paths[0]).read_text())["comparison_record"] #TODO: should be enum instead of hardcoded string
        )
        persist_statistical_metadata(
            store,
            layout,
            request.experiment,
            DirectedPairName(pair),
            TransferMethod.FEDORBIT_WITHOUT_CONFIRMATION,
            comparison_record,
            ComparisonStatistic.SEED_LEVEL_RATE_DIFFERENCE_SIGN_FLIP,
            confirmation_nonzero_count,
            confirmation_bootstrap_seed,
            _holm_rank(confirmation_raw_p, pair),
            confirmation_family_size,
            request.overwrite_policy,
        )


def _completed_condition_macro_ce(
    store: ArtifactStore,
    experiment: ExperimentName,
) -> Mapping[
    tuple[DirectedPairName, TransferMethod, EvaluationConditionName, RandomSeed], _SeedMetric
]:
    result: OrderedDict[
        tuple[DirectedPairName, TransferMethod, EvaluationConditionName, RandomSeed], _SeedMetric
    ] = OrderedDict()
    for resolved, record_payload in _iter_completed_json_payloads(
        store, experiment, "metric_record" #TODO: should be enum instead of hardcoded string
    ):
        record = MetricRecord.model_validate(record_payload)
        if (
            record.metric_name != MetricId.MACRO_CROSS_ENTROPY
            or not record.valid
            or record.metric_value is None
        ):
            continue
        result[(record.pair, record.method, record.condition, record.seed)] = _SeedMetric(
            float(record.metric_value), resolved.artifact_id
        )
    return result


def _completed_real_packet_coupling_gap(
    store: ArtifactStore,
) -> Mapping[tuple[DirectedPairName, RandomSeed], _SeedMetric]:
    result: OrderedDict[tuple[DirectedPairName, RandomSeed], _SeedMetric] = OrderedDict()
    for resolved, record_payload in _iter_completed_json_payloads(
        store, ExperimentName.REAL_PACKET_COUPLING_MECHANISM_VALIDATION, "metric_record" #TODO: should be enum instead of hardcoded string
    ):
        record = MetricRecord.model_validate(record_payload)
        if (
            record.metric_name != MetricId.ROBUST_COUPLING_VALUE_GAP
            or record.method != TransferMethod.MATCHED_RESOURCE_RECTANGULAR
            or not record.valid
            or record.metric_value is None
        ):
            continue
        result[(record.pair, record.seed)] = _SeedMetric(
            float(record.metric_value), resolved.artifact_id
        )
    return result


def persist_coupling_mechanism_comparison(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    pair: DirectedPairName,
    paired_seed_count: Index,
    mean_difference: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    median_difference: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    bca_ci_low: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    bca_ci_high: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    raw_p: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    holm_p: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    decision: ComparisonDecision,
    input_metric_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        directed_pair=DirectedPair(
            source=DatasetId(pair.split(" -> ")[0]), target=DatasetId(pair.split(" -> ")[1])
        ),
        method=TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.STATISTICS,
            cell,
            relevance,
            tuple(identifier.value for identifier in input_metric_artifact_ids),
            _STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    comparison = PairedComparisonRecord(
        contrast_name=ContrastName(
            f"Exact correspondence orbit vs Matched-Resource Rectangular: {pair}"
        ),
        family=MultiplicityFamily.COUPLING_MECHANISM,
        pair=DirectedPairName(pair),
        method_a=ExperimentLocalMethod.EXACT_ORBIT,
        method_b=TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
        metric=MetricId.ROBUST_COUPLING_VALUE_GAP,
        paired_seed_count=paired_seed_count,
        mean_difference=mean_difference,
        median_difference=median_difference,
        bca_ci_low=bca_ci_low,
        bca_ci_high=bca_ci_high,
        raw_p=raw_p,
        holm_p=holm_p,
        materiality_threshold=active_config().scientific.materiality.coupling_objective_units,
        equivalence_margin_low=None,
        equivalence_margin_high=None,
        input_metric_artifact_ids=tuple(input_metric_artifact_ids),
        dependency_fingerprint_sha256=fingerprint,
        decision=decision,
    )
    payload_path = (
        experiment_workspace(layout, experiment)
        / "artifacts" #TODO: should be enums not hardcoded strings
        / "derived" #TODO: should be enums not hardcoded strings
        / f"coupling-comparison.{pair.replace(' -> ', '-to-')}.json"
    )
    payload = cast(
        StableJsonPayload, OrderedDict(comparison_record=comparison.model_dump(mode="json"))
    )
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(ArtifactStage.STATISTICS).sha256)
    completion = build_completion_manifest(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=ArtifactStage.STATISTICS,
        upstream_artifact_ids=tuple(input_metric_artifact_ids),
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.OTHER.value), payload, Sha256Digest(fingerprint)
            ),
            artifact_type=ArtifactType.OTHER,
            semantic_producer_coordinates=coordinates,
            producer_stage=ArtifactStage.STATISTICS,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=tuple(input_metric_artifact_ids),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            material_runtime_sha256=runtime_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_sha256,
            schema_version="1.0", #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
            created_git_commit=current_code_revision().commit,
            created_environment_sha256=environment_snapshot().fingerprint_sha256,
            state=ArtifactState.COMPLETED,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
    )
    store.write_completed(manifest, completion)
    return manifest


def persist_baseline_comparison(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    pair: DirectedPairName,
    family: MultiplicityFamily,
    method_a: MethodName,
    method_b: MethodName,
    metric: MetricId,
    contrast_name: ContrastName,
    materiality_threshold: RelativeGain | None,
    paired_seed_count: Index,
    mean_difference: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    median_difference: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    bca_ci_low: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    bca_ci_high: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    raw_p: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    holm_p: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    decision: ComparisonDecision,
    equivalence_margin_low: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    equivalence_margin_high: float | None, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    input_metric_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        directed_pair=DirectedPair(
            source=DatasetId(pair.split(" -> ")[0]), target=DatasetId(pair.split(" -> ")[1])
        ),
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        condition=ExperimentCondition(contrast_name),
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.STATISTICS,
            cell,
            relevance,
            tuple(identifier.value for identifier in input_metric_artifact_ids),
            _STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    comparison = PairedComparisonRecord(
        contrast_name=contrast_name,
        family=family,
        pair=pair,
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        paired_seed_count=paired_seed_count,
        mean_difference=mean_difference,
        median_difference=median_difference,
        bca_ci_low=bca_ci_low,
        bca_ci_high=bca_ci_high,
        raw_p=raw_p,
        holm_p=holm_p,
        materiality_threshold=materiality_threshold,
        equivalence_margin_low=equivalence_margin_low,
        equivalence_margin_high=equivalence_margin_high,
        input_metric_artifact_ids=tuple(input_metric_artifact_ids),
        dependency_fingerprint_sha256=fingerprint,
        decision=decision,
    )
    payload_path = (
        experiment_workspace(layout, experiment)
        / "artifacts" #TODO: should be enums not hardcoded strings
        / "derived" #TODO: should be enums not hardcoded strings
        / (f"baseline-comparison.{pair.replace(' -> ', '-to-')}.{safe_slug(contrast_name)}.json")
    )
    payload = cast(
        StableJsonPayload, OrderedDict(comparison_record=comparison.model_dump(mode="json"))
    )
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(ArtifactStage.STATISTICS).sha256)
    completion = build_completion_manifest(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=ArtifactStage.STATISTICS,
        upstream_artifact_ids=tuple(input_metric_artifact_ids),
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.OTHER.value), payload, Sha256Digest(fingerprint)
            ),
            artifact_type=ArtifactType.OTHER,
            semantic_producer_coordinates=coordinates,
            producer_stage=ArtifactStage.STATISTICS,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=tuple(input_metric_artifact_ids),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            material_runtime_sha256=runtime_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_sha256,
            schema_version="1.0", #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
            created_git_commit=current_code_revision().commit,
            created_environment_sha256=environment_snapshot().fingerprint_sha256,
            state=ArtifactState.COMPLETED,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
    )
    store.write_completed(manifest, completion)
    return manifest


def _holm_rank(raw_p_by_pair: Mapping[str, float] #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
               , pair: str #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
               ) -> Index:
    ordered = sorted(raw_p_by_pair.items(), key=lambda item: (item[1], item[0]))
    rank: Index = next(index for index, (name, _) in enumerate(ordered, start=1) if name == pair)
    return rank


def persist_statistical_metadata(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    pair: DirectedPairName,
    method: TransferMethod,
    comparison: PairedComparisonRecord,
    comparison_statistic: ComparisonStatistic,
    nonzero_difference_count: Index,
    bootstrap_seed: RandomSeed,
    holm_rank: Index,
    family_size: SampleCount,
    overwrite_policy: OverwritePolicy,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        directed_pair=DirectedPair(
            source=DatasetId(pair.split(" -> ")[0]), target=DatasetId(pair.split(" -> ")[1])
        ),
        method=method,
        condition=ExperimentCondition(comparison_statistic.value),
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    input_artifact_ids = comparison.input_metric_artifact_ids
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.STATISTICS,
            cell,
            relevance,
            tuple(identifier.value for identifier in input_artifact_ids),
            _STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    statistics_config = active_config().scientific.statistics
    zero_difference_count: Index = comparison.paired_seed_count - nonzero_difference_count
    resamples: ResampleCount = statistics_config.ci_bootstrap_repetitions
    metadata = StatisticalMetadataRecord(
        test_name=StatisticalTestName(comparison_statistic.value),
        exact_or_asymptotic=StatisticalExactness.EXACT,
        alternative=StatisticalAlternative.TWO_SIDED,
        zero_difference_count=zero_difference_count,
        bootstrap_resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        holm_rank=holm_rank,
        family_size=family_size,
        statistical_code_sha256=implementation_fingerprint(_MODULE_NAME),
    )
    validate_comparison_metadata(comparison, metadata)
    payload_path = (
        experiment_workspace(layout, experiment)
        / "artifacts" #TODO: should be enums not hardcoded strings
        / "derived" #TODO: should be enums not hardcoded strings
        / (
            f"statistical-metadata.{pair.replace(' -> ', '-to-')}.{method.value}"
            f".{safe_slug(comparison_statistic.value)}.json" #TODO: should be enums not hardcoded strings
        )
    )
    payload = cast(
        StableJsonPayload, OrderedDict(statistical_metadata_record=metadata.model_dump(mode="json"))
    )
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_STATISTICAL_SYNTHESIS_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(ArtifactStage.STATISTICS).sha256)
    completion = build_completion_manifest(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=ArtifactStage.STATISTICS,
        upstream_artifact_ids=tuple(input_artifact_ids),
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.OTHER.value), payload, Sha256Digest(fingerprint)
            ),
            artifact_type=ArtifactType.OTHER,
            semantic_producer_coordinates=coordinates,
            producer_stage=ArtifactStage.STATISTICS,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=tuple(input_artifact_ids),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            material_runtime_sha256=runtime_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_sha256,
            schema_version="1.0", #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
            created_git_commit=current_code_revision().commit,
            created_environment_sha256=environment_snapshot().fingerprint_sha256,
            state=ArtifactState.COMPLETED,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
    )
    store.write_completed(manifest, completion)
    return manifest
