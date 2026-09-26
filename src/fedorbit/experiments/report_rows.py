from __future__ import annotations

import json
import math
import statistics
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from pydantic import JsonValue

from fedorbit.analysis.metrics import (
    CrossEntropy,
    confirmation_coverage,
    relative_macro_ce_gain,
)
from fedorbit.analysis.records import MetricRecord, PairedComparisonRecord
from fedorbit.config.loading import active_config
from fedorbit.datasets.materialization import MaterializedClient
from fedorbit.datasets.ontology import TRANSFER_ONTOLOGY, transfer_concept_for, transfer_eligibility
from fedorbit.experiments.catalogue import ExperimentCatalogue
from fedorbit.experiments.classification import EvidenceStatusRow
from fedorbit.experiments.metric_persistence import latest_completed_manifest
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.evidence import FigureSeries, TableScalar
from fedorbit.infrastructure.manifests import DatasetManifest
from fedorbit.infrastructure.workspace import WorkspaceLayout, experiment_workspace
from fedorbit.types import (
    PRINCIPAL_EVALUATION_CONDITION,
    ClientRole,
    Coefficient,
    ConceptCount,
    ConfidenceIntervalText,
    DatasetId,
    DatasetModality,
    DirectedPairName,
    Estimate,
    EvaluationConditionName,
    ExperimentLocalMethod,
    ExperimentName,
    FineLabel,
    Fraction,
    Index,
    MetricId,
    MultiplicityFamily,
    OracleTransferConcept,
    RandomSeed,
    RelativeGain,
    ReportColumnName,
    ReportSeriesName,
    ScalabilityBlockPattern,
    SignificanceLevel,
    Split,
    StableJsonPayload,
    StorageLayoutSegment,
    SupportCount,
    TransferMethod,
    WeakSignalBoundaryDimension,
)

_WEAK_SIGNAL_BOUNDARY_DIMENSIONS = tuple(WeakSignalBoundaryDimension)


@dataclass(frozen=True, slots=True)
class TransferGainFigureData:
    series: tuple[FigureSeries, ...]
    pair_labels: tuple[ReportSeriesName, ...]


def _realized_gain(
    reference: Estimate | None,
    method: Estimate | None,
) -> RelativeGain | None:
    if reference is None or method is None:
        return None
    return relative_macro_ce_gain(CrossEntropy(reference), CrossEntropy(method)).relative


def base_model_pilot_dataset_manifests(layout: WorkspaceLayout) -> tuple[DatasetManifest, ...]:
    workspace = experiment_workspace(layout, ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT)
    manifests: list[DatasetManifest] = []
    for dataset in DatasetId:
        path = (
            workspace
            / StorageLayoutSegment.ARTIFACTS
            / StorageLayoutSegment.DERIVED
            / f"dataset-manifest.{dataset.value}.json"
        )
        if path.is_file():
            manifests.append(DatasetManifest.model_validate_json(path.read_text(encoding="utf-8")))
    return tuple(manifests)


def dataset_client_roles() -> Mapping[DatasetId, ClientRole]:
    return OrderedDict(
        (dataset, client.role)
        for dataset, client in active_config().scientific.datasets.clients.items()
    )


def dataset_modality_by_dataset() -> Mapping[DatasetId, DatasetModality]:
    modalities: OrderedDict[DatasetId, DatasetModality] = OrderedDict()
    for dataset in DatasetId:
        modalities[dataset] = (
            DatasetModality.NETWORK
            if dataset in {DatasetId.EDGE_IIOTSET_NETWORK, DatasetId.TON_IOT_NETWORK}
            else DatasetModality.HOST
        )
    return modalities


def excluded_class_counts(
    manifests: Sequence[DatasetManifest],
) -> Mapping[DatasetId, Index]:
    return OrderedDict(
        (manifest.dataset, max(0, manifest.feature_quality.dropped_feature_count))
        for manifest in manifests
    )


def experiment_matrix_rows(
    catalogue: ExperimentCatalogue,
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for name in catalogue.registered_names():
        definition = catalogue.definition(name)
        rows.append(
            OrderedDict(
                (
                    (ReportColumnName.EXPERIMENT, name.value),
                    (ReportColumnName.CLASSIFICATION, definition.classification.value),
                    (
                        ReportColumnName.DATASETS_OR_PAIRS,
                        ", ".join(scope.value for scope in definition.datasets_or_pairs),
                    ),
                    (
                        ReportColumnName.METHODS,
                        ", ".join(str(method) for method in definition.methods),
                    ),
                    (
                        ReportColumnName.REGISTERED_SEEDS,
                        ", ".join(str(seed) for seed in definition.seeds),
                    ),
                    (ReportColumnName.CONDITIONS, str(definition.conditions)),
                    (ReportColumnName.DERIVED_PLANNED_CELLS, definition.derived_planned_cells),
                    (
                        ReportColumnName.PREREQUISITES,
                        ", ".join(str(prerequisite) for prerequisite in definition.prerequisites),
                    ),
                    (
                        ReportColumnName.EVIDENCE_RELATIONSHIP,
                        (
                            ", ".join(consumer.value for consumer in definition.evidence_consumers)
                            if definition.evidence_consumers
                            else "report export"
                        ),
                    ),
                )
            )
        )
    return tuple(rows)


def real_transfer_gain_series(
    comparisons: Sequence[PairedComparisonRecord],
) -> TransferGainFigureData:
    selected = [
        record
        for record in comparisons
        if record.method_a == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
        and record.method_b == TransferMethod.LOCAL_ONLY
        and record.mean_difference is not None
    ]
    if not selected:
        return TransferGainFigureData((), ())
    selected.sort(key=lambda record: record.pair)
    mids: list[Coefficient] = []
    lows: list[Coefficient] = []
    highs: list[Coefficient] = []
    pair_labels: list[ReportSeriesName] = []
    for record in selected:
        mid = record.mean_difference
        if mid is None:
            continue
        mids.append(mid)
        lows.append(record.bca_ci_low if record.bca_ci_low is not None else mid)
        highs.append(record.bca_ci_high if record.bca_ci_high is not None else mid)
        pair_labels.append(ReportSeriesName(record.pair))
    if not mids:
        return TransferGainFigureData((), ())
    series = (
        FigureSeries(
            name=ReportSeriesName(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value),
            x=tuple(mids),
            y=tuple(float(index) for index in range(len(mids))),
            x_low=tuple(lows),
            x_high=tuple(highs),
        ),
    )
    return TransferGainFigureData(series, tuple(pair_labels))


def baseline_paired_difference_series(
    metric_records: Sequence[MetricRecord],
) -> tuple[FigureSeries, ...]:
    pairs = sorted({record.pair for record in metric_records})
    baseline_methods = (
        TransferMethod.LOCAL_SIR,
        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
    )
    series: list[FigureSeries] = []
    for pair in pairs:
        for method in baseline_methods:
            x_values: list[Coefficient] = []
            y_values: list[Coefficient] = []
            for record in metric_records:
                if (
                    record.pair != pair
                    or record.method != method
                    or record.condition != PRINCIPAL_EVALUATION_CONDITION.name
                    or record.metric_name != MetricId.RELATIVE_MACRO_CE_GAIN
                    or not record.valid
                    or record.metric_value is None
                ):
                    continue
                x_values.append(float(record.seed))
                y_values.append(record.metric_value)
            if x_values:
                series.append(
                    FigureSeries(
                        name=ReportSeriesName(method.value),
                        x=tuple(x_values),
                        y=tuple(y_values),
                        panel=ReportSeriesName(pair),
                    )
                )
    return tuple(series)


def _median(
    values: Sequence[Estimate],
) -> Estimate | None:
    if not values:
        return None
    return statistics.median(values)


def percentile_95(
    values: Sequence[Estimate],
) -> Estimate | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return float(np.quantile(values, 0.95, method="linear"))


def _metric_values(
    records: Sequence[MetricRecord], method: TransferMethod, metric_name: MetricId
) -> list[Estimate]:
    return [
        record.metric_value
        for record in records
        if record.method == method
        and record.metric_name == metric_name
        and record.valid
        and record.metric_value is not None
    ]


def _parse_k_pattern_condition(
    condition: EvaluationConditionName,
) -> tuple[ConceptCount, ScalabilityBlockPattern] | None:
    if not condition.startswith("k"):
        return None
    k_text, separator, pattern = condition[1:].partition("-")
    if not separator or not k_text.isdigit() or not pattern:
        return None
    try:
        block_pattern = ScalabilityBlockPattern(pattern)
    except ValueError:
        return None
    return int(k_text), block_pattern


def exact_solver_results_rows(
    records_with_support: Sequence[tuple[MetricRecord, SupportCount | None]],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    groups: OrderedDict[
        tuple[ConceptCount, ScalabilityBlockPattern, SupportCount | None], list[MetricRecord]
    ] = OrderedDict()
    for record, support in records_with_support:
        parsed = _parse_k_pattern_condition(record.condition)
        if parsed is None:
            continue
        groups.setdefault((*parsed, support), []).append(record)
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for (k, pattern, support), entries in groups.items():
        errors = _metric_values(
            entries, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, MetricId.ABSOLUTE_OBJECTIVE_ERROR
        )
        validity = _metric_values(
            entries,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            MetricId.CORRESPONDENCE_CERTIFICATE_VALIDITY,
        )
        runtimes = _metric_values(
            entries, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, MetricId.WALL_TIME
        )
        memory = _metric_values(
            entries, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, MetricId.PEAK_HOST_RSS
        )
        active_images = _metric_values(
            entries, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, MetricId.ACTIVE_IMAGE_CANDIDATES
        )
        lap_calls = _metric_values(
            entries, TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, MetricId.LAP_CALLS
        )
        qap_runtimes = _metric_values(entries, TransferMethod.GENERIC_EXACT_QAP, MetricId.WALL_TIME)
        qap_timeouts = _metric_values(
            entries, TransferMethod.GENERIC_EXACT_QAP, MetricId.TIMEOUT_INDICATOR
        )
        dense_runtimes = _metric_values(
            entries, TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK, MetricId.WALL_TIME
        )
        rows.append(
            OrderedDict(
                (
                    (
                        ReportColumnName.K,
                        k,
                    ),
                    (
                        ReportColumnName.BLOCK_PATTERN,
                        pattern,
                    ),
                    (
                        ReportColumnName.SUPPORT,
                        support,
                    ),
                    (
                        ReportColumnName.TRUTH_AVAILABILITY,
                        bool(errors),
                    ),
                    (
                        ReportColumnName.EXACT_MISMATCHES,
                        sum(1 for value in validity if value == 0.0),
                    ),
                    (
                        ReportColumnName.MAXIMUM_ABSOLUTE_ERROR,
                        max(errors) if errors else None,
                    ),
                    (
                        ReportColumnName.RUNTIME_MEDIAN,
                        _median(runtimes),
                    ),
                    (
                        ReportColumnName.RUNTIME_P95,
                        percentile_95(runtimes),
                    ),
                    (
                        ReportColumnName.QAP_RUNTIME,
                        _median(qap_runtimes),
                    ),
                    (
                        ReportColumnName.DENSE_RUNTIME,
                        _median(dense_runtimes),
                    ),
                    (
                        ReportColumnName.TIMEOUTS,
                        sum(1 for value in qap_timeouts if value == 1.0),
                    ),
                    (
                        ReportColumnName.MEMORY,
                        _median(memory),
                    ),
                    (
                        ReportColumnName.ACTIVE_IMAGES,
                        _median(active_images),
                    ),
                    (
                        ReportColumnName.LAP_CALLS,
                        _median(lap_calls),
                    ),
                )
            )
        )
    return tuple(rows)


def scalability_results_rows(
    records_with_support: Sequence[tuple[MetricRecord, SupportCount | None]],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    groups: OrderedDict[
        tuple[ConceptCount, ScalabilityBlockPattern, SupportCount | None, TransferMethod],
        list[MetricRecord],
    ] = OrderedDict()
    for record, support in records_with_support:
        parsed = _parse_k_pattern_condition(record.condition)
        if parsed is None:
            continue
        groups.setdefault((*parsed, support, record.method), []).append(record)
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for (k, pattern, support, method), entries in groups.items():
        runtimes = _metric_values(entries, method, MetricId.WALL_TIME)
        rss = _metric_values(entries, method, MetricId.PEAK_HOST_RSS)
        cuda_memory = _metric_values(entries, method, MetricId.PEAK_CUDA_ALLOCATED_BYTES)
        active_images = _metric_values(entries, method, MetricId.ACTIVE_IMAGE_CANDIDATES)
        lap_calls = _metric_values(entries, method, MetricId.LAP_CALLS)
        timeouts = _metric_values(entries, method, MetricId.TIMEOUT_INDICATOR)
        predicted_work = _metric_values(entries, method, MetricId.PREDICTED_WORK_COORDINATE)
        rows.append(
            OrderedDict(
                (
                    (
                        ReportColumnName.K,
                        k,
                    ),
                    (
                        ReportColumnName.BLOCK,
                        pattern,
                    ),
                    (
                        ReportColumnName.SUPPORT,
                        support,
                    ),
                    (
                        ReportColumnName.METHOD,
                        method.value,
                    ),
                    (
                        ReportColumnName.N_S,
                        _median(active_images),
                    ),
                    (
                        ReportColumnName.LAP_CALLS,
                        _median(lap_calls),
                    ),
                    (
                        ReportColumnName.CUTS,
                        None,
                    ),
                    (
                        ReportColumnName.RUNTIME_MEDIAN,
                        _median(runtimes),
                    ),
                    (
                        ReportColumnName.RUNTIME_P95,
                        percentile_95(runtimes),
                    ),
                    (
                        ReportColumnName.RSS,
                        _median(rss),
                    ),
                    (
                        ReportColumnName.CUDA_MEMORY,
                        _median(cuda_memory),
                    ),
                    (
                        ReportColumnName.TIMEOUT,
                        sum(1 for value in timeouts if value == 1.0),
                    ),
                    (
                        ReportColumnName.EXACTNESS_STATUS,
                        None,
                    ),
                    (
                        ReportColumnName.PREDICTED_WORK,
                        _median(predicted_work),
                    ),
                )
            )
        )
    return tuple(rows)


def ablation_results_rows(
    records: Sequence[MetricRecord],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    pairs = sorted({record.pair for record in records})
    ablation_methods = sorted(
        {record.method for record in records if record.method != TransferMethod.LOCAL_ONLY},
        key=lambda method: method.value,
    )
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for pair in pairs:
        full_ce = _metric_value_by_condition(
            records,
            pair,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            PRINCIPAL_EVALUATION_CONDITION.name,
        )
        for method in ablation_methods:
            ablation_ce = _metric_value_by_condition(
                records, pair, method, PRINCIPAL_EVALUATION_CONDITION.name
            )
            if ablation_ce is None:
                continue
            difference_vs_full = ablation_ce - full_ce if full_ce is not None else None
            rows.append(
                OrderedDict(
                    (
                        (
                            ReportColumnName.ABLATION,
                            method.value,
                        ),
                        (
                            ReportColumnName.PAIR,
                            pair,
                        ),
                        (
                            ReportColumnName.REALIZED_GAIN,
                            None,
                        ),
                        (
                            ReportColumnName.DIFFERENCE_VS_FULL,
                            difference_vs_full,
                        ),
                        (
                            ReportColumnName.EQUIVALENCE,
                            None,
                        ),
                        (
                            ReportColumnName.RETAINED_GAIN,
                            None,
                        ),
                        (
                            ReportColumnName.CONFIRMATION_SAFETY,
                            None,
                        ),
                    )
                )
            )
    return tuple(rows)


def _metric_value_by_condition(
    records: Sequence[MetricRecord],
    pair: DirectedPairName,
    method: TransferMethod,
    condition: EvaluationConditionName,
    metric_name: MetricId = MetricId.MACRO_CROSS_ENTROPY,
) -> float | None:
    matches = [
        record
        for record in records
        if record.pair == pair
        and record.method == method
        and record.condition == condition
        and record.metric_name == metric_name
        and record.valid
        and record.metric_value is not None
    ]
    if not matches:
        return None
    return statistics.fmean(
        record.metric_value for record in matches if record.metric_value is not None
    )


def sparsity_and_dense_results_rows(
    sparsity_records: Sequence[MetricRecord],
    local_only_records: Sequence[MetricRecord],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    pairs = sorted({record.pair for record in sparsity_records})
    conditions = sorted({record.condition for record in sparsity_records})
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for pair in pairs:
        local_only_ce = _metric_value_by_condition(
            local_only_records,
            pair,
            TransferMethod.LOCAL_ONLY,
            PRINCIPAL_EVALUATION_CONDITION.name,
        )
        for condition in conditions:
            method = next(
                (
                    record.method
                    for record in sparsity_records
                    if record.pair == pair and record.condition == condition
                ),
                None,
            )
            if method is None:
                continue
            condition_ce = _metric_value_by_condition(sparsity_records, pair, method, condition)
            realized_gain = _realized_gain(local_only_ce, condition_ce)
            runtime = _metric_value_by_condition(
                sparsity_records,
                pair,
                method,
                condition,
                metric_name=MetricId.WALL_TIME,
            )
            memory = _metric_value_by_condition(
                sparsity_records,
                pair,
                method,
                condition,
                metric_name=MetricId.PEAK_HOST_RSS,
            )
            coverage = _metric_value_by_condition(
                sparsity_records,
                pair,
                method,
                condition,
                metric_name=MetricId.PROPOSAL_ACCEPTANCE_RATE,
            )
            certified = _metric_value_by_condition(
                sparsity_records,
                pair,
                method,
                condition,
                metric_name=MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE,
            )
            rows.append(
                OrderedDict(
                    (
                        (
                            ReportColumnName.SUPPORT_OR_DENSE_CONDITION,
                            condition,
                        ),
                        (
                            ReportColumnName.PAIR,
                            pair,
                        ),
                        (
                            ReportColumnName.REALIZED_GAIN,
                            realized_gain,
                        ),
                        (
                            ReportColumnName.CERTIFIED_VALUE,
                            certified,
                        ),
                        (
                            ReportColumnName.RUNTIME,
                            runtime,
                        ),
                        (
                            ReportColumnName.MEMORY,
                            memory,
                        ),
                        (
                            ReportColumnName.CONFIRMATION_COVERAGE,
                            coverage,
                        ),
                        (
                            ReportColumnName.DENSE_MINUS_SPARSE_DIFFERENCE,
                            None,
                        ),
                    )
                )
            )
    return tuple(rows)


def generalization_results_rows(
    metric_records: Sequence[MetricRecord],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    pairs = sorted({record.pair for record in metric_records})
    method_order = (
        TransferMethod.LOCAL_ONLY,
        TransferMethod.LOCAL_SIR,
        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
    )
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for pair in pairs:
        for method in method_order:
            valid_seeds = len(
                {
                    record.seed
                    for record in metric_records
                    if record.pair == pair
                    and record.method == method
                    and record.metric_name == MetricId.MACRO_CROSS_ENTROPY
                    and record.valid
                }
            )
            if valid_seeds == 0:
                continue
            rows.append(
                OrderedDict(
                    (
                        (ReportColumnName.PAIR, pair),
                        (ReportColumnName.METHOD, method.value),
                        (ReportColumnName.VALID_SEEDS, valid_seeds),
                        (
                            ReportColumnName.TEST_MACRO_CE,
                            _metric_value_by_condition(
                                metric_records, pair, method, PRINCIPAL_EVALUATION_CONDITION.name
                            ),
                        ),
                        (
                            ReportColumnName.MACRO_F1,
                            _metric_value_by_condition(
                                metric_records,
                                pair,
                                method,
                                PRINCIPAL_EVALUATION_CONDITION.name,
                                metric_name=MetricId.MACRO_F1,
                            ),
                        ),
                        (
                            ReportColumnName.BALANCED_ACCURACY,
                            _metric_value_by_condition(
                                metric_records,
                                pair,
                                method,
                                PRINCIPAL_EVALUATION_CONDITION.name,
                                metric_name=MetricId.BALANCED_ACCURACY,
                            ),
                        ),
                        (ReportColumnName.GAIN_VS_LOCAL, None),
                        (ReportColumnName.BCA_CI_LOW, None),
                        (ReportColumnName.BCA_CI_HIGH, None),
                        (ReportColumnName.RAW_P, None),
                        (ReportColumnName.HOLM_P, None),
                        (ReportColumnName.STRICT_VALIDITY, None),
                        (ReportColumnName.CONFIRMATION_COVERAGE, None),
                        (ReportColumnName.IS_SECONDARY_PAIR, True),
                    )
                )
            )
    return tuple(rows)


def confirmation_results_rows(
    records: Sequence[MetricRecord],
    comparisons: Sequence[PairedComparisonRecord] = (),
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    pairs = sorted({record.pair for record in records})
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for pair in pairs:
        verdicts = [
            record.metric_value
            for record in records
            if record.pair == pair
            and record.method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
            and record.metric_name == MetricId.PROPOSAL_ACCEPTANCE_RATE
            and record.valid
            and record.metric_value is not None
        ]
        if not verdicts:
            continue
        proposals = len(verdicts)
        accepted = sum(1 for value in verdicts if value == 1.0)

        def _mean(metric_name: MetricId, pair_name: DirectedPairName = pair) -> float | None:
            values = [
                record.metric_value
                for record in records
                if record.pair == pair_name
                and record.method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
                and record.metric_name == metric_name
                and record.valid
                and record.metric_value is not None
            ]
            if not values:
                return None
            return sum(values) / len(values)

        rows.append(
            OrderedDict(
                (
                    (
                        ReportColumnName.PAIR,
                        pair,
                    ),
                    (
                        ReportColumnName.PROPOSALS,
                        proposals,
                    ),
                    (
                        ReportColumnName.ACCEPTED,
                        accepted,
                    ),
                    (
                        ReportColumnName.HARMFUL_ACCEPTED_RATE,
                        _mean(MetricId.HARMFUL_ACCEPTED_RATE),
                    ),
                    (
                        ReportColumnName.USEFUL_ACCEPTED_RATE,
                        _mean(MetricId.USEFUL_ACCEPTED_RATE),
                    ),
                    (
                        ReportColumnName.BENEFICIAL_REJECTED_RATE,
                        _mean(MetricId.BENEFICIAL_REJECTED_RATE),
                    ),
                    (
                        ReportColumnName.COVERAGE,
                        _confirmation_coverage(records, pair, accepted, proposals),
                    ),
                    (
                        ReportColumnName.NO_CONFIRM_HARMFUL_RATE,
                        _mean(MetricId.HARM_RATE_NO_CONFIRM),
                    ),
                    (
                        ReportColumnName.ARR,
                        _mean(MetricId.ABSOLUTE_RISK_REDUCTION),
                    ),
                    (
                        ReportColumnName.RRR,
                        _mean(MetricId.RELATIVE_RISK_REDUCTION),
                    ),
                    (
                        ReportColumnName.CI,
                        _confirmation_ci(comparisons, pair),
                    ),
                    (
                        ReportColumnName.P,
                        _confirmation_p(comparisons, pair),
                    ),
                )
            )
        )
    return tuple(rows)


def _confirmation_contrast(
    comparisons: Sequence[PairedComparisonRecord], pair: DirectedPairName
) -> PairedComparisonRecord | None:
    for record in comparisons:
        if record.family == MultiplicityFamily.CONFIRMATION_SAFETY and record.pair == pair:
            return record
    return None


def _confirmation_ci(
    comparisons: Sequence[PairedComparisonRecord], pair: DirectedPairName
) -> ConfidenceIntervalText | None:
    record = _confirmation_contrast(comparisons, pair)
    if record is None or record.bca_ci_low is None or record.bca_ci_high is None:
        return None
    return ConfidenceIntervalText(f"[{record.bca_ci_low}, {record.bca_ci_high}]")


def _confirmation_p(
    comparisons: Sequence[PairedComparisonRecord], pair: DirectedPairName
) -> SignificanceLevel | None:
    record = _confirmation_contrast(comparisons, pair)
    if record is None:
        return None
    return record.holm_p


def _coupling_gap_row(
    condition_or_pair: DirectedPairName | EvaluationConditionName | str,
    support: SupportCount | None,
    method: TransferMethod,
    gap_values: Sequence[Estimate],
    fixed_action_values: Sequence[Estimate],
    ci: ConfidenceIntervalText | None,
    holm_p: SignificanceLevel | None,
) -> Mapping[ReportColumnName, TableScalar]:
    materiality = active_config().scientific.materiality.coupling_objective_units
    above_materiality = sum(1 for value in gap_values if value > materiality)
    return OrderedDict(
        (
            (
                ReportColumnName.CONDITION_OR_PAIR,
                condition_or_pair,
            ),
            (ReportColumnName.SUPPORT, support),
            (ReportColumnName.METHOD, method.value),
            (
                ReportColumnName.VALID_UNITS,
                len(gap_values),
            ),
            (
                ReportColumnName.FIXED_ACTION_GAP,
                statistics.fmean(fixed_action_values) if fixed_action_values else None,
            ),
            (
                ReportColumnName.ROBUST_COUPLING_GAP,
                statistics.fmean(gap_values) if gap_values else None,
            ),
            (
                ReportColumnName.FRACTION_ABOVE_MATERIALITY,
                above_materiality / len(gap_values) if gap_values else None,
            ),
            (
                ReportColumnName.CI,
                ci,
            ),
            (
                ReportColumnName.HOLM_P,
                holm_p,
            ),
        )
    )


def coupling_mechanism_results_rows(
    synthetic_records_with_support: Sequence[tuple[MetricRecord, SupportCount | None]],
    real_packet_records: Sequence[MetricRecord],
    comparison_records: Sequence[PairedComparisonRecord],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    synthetic_cells = sorted(
        {
            (record.condition, support, record.method)
            for record, support in synthetic_records_with_support
            if record.method
            in {
                TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                TransferMethod.COUPLING_DESTROYED_FEDORBIT,
            }
        },
        key=lambda cell: (cell[0], -1 if cell[1] is None else cell[1], cell[2].value),
    )
    for condition, support, method in synthetic_cells:
        gap_values = [
            float(record.metric_value)
            for record, record_support in synthetic_records_with_support
            if record.condition == condition
            and record_support == support
            and record.method == method
            and record.metric_name == MetricId.ROBUST_COUPLING_VALUE_GAP
            and record.valid
            and record.metric_value is not None
        ]
        fixed_action_values = [
            float(record.metric_value)
            for record, record_support in synthetic_records_with_support
            if record.condition == condition
            and record_support == support
            and record.method == method
            and record.metric_name == MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP
            and record.valid
            and record.metric_value is not None
        ]
        if not gap_values:
            continue
        rows.append(
            _coupling_gap_row(
                condition, support, method, gap_values, fixed_action_values, None, None
            )
        )
    pairs = sorted({record.pair for record in real_packet_records})
    comparisons_by_pair = OrderedDict(
        (comparison.pair, comparison)
        for comparison in comparison_records
        if comparison.family == MultiplicityFamily.COUPLING_MECHANISM
        and comparison.method_a == ExperimentLocalMethod.EXACT_ORBIT
    )
    for pair in pairs:
        gap_values = [
            float(record.metric_value)
            for record in real_packet_records
            if record.pair == pair
            and record.method == TransferMethod.MATCHED_RESOURCE_RECTANGULAR
            and record.metric_name == MetricId.ROBUST_COUPLING_VALUE_GAP
            and record.valid
            and record.metric_value is not None
        ]
        fixed_action_values = [
            float(record.metric_value)
            for record in real_packet_records
            if record.pair == pair
            and record.method == TransferMethod.MATCHED_RESOURCE_RECTANGULAR
            and record.metric_name == MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP
            and record.valid
            and record.metric_value is not None
        ]
        if not gap_values:
            continue
        comparison = comparisons_by_pair.get(DirectedPairName(pair))
        ci = (
            ConfidenceIntervalText(f"[{comparison.bca_ci_low:.4g}, {comparison.bca_ci_high:.4g}]")
            if comparison is not None
            and comparison.bca_ci_low is not None
            and comparison.bca_ci_high is not None
            else None
        )
        holm_p = comparison.holm_p if comparison is not None else None
        rows.append(
            _coupling_gap_row(
                pair,
                None,
                TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                gap_values,
                fixed_action_values,
                ci,
                holm_p,
            )
        )
    return tuple(rows)


def _boundary_dimension_and_setting(
    condition: EvaluationConditionName,
) -> tuple[str, str]:
    for dimension in _WEAK_SIGNAL_BOUNDARY_DIMENSIONS:
        prefix = f"{dimension}-"
        if condition.startswith(prefix):
            return dimension, condition[len(prefix) :]
    return "semantic-sufficiency-partition", condition


def failure_boundary_results_rows(
    weak_signal_records: Sequence[MetricRecord],
    semantic_records: Sequence[MetricRecord],
    primary_transfer_records: Sequence[MetricRecord],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for experiment_records in (weak_signal_records, semantic_records):
        cells: set[tuple[DirectedPairName, EvaluationConditionName, TransferMethod]] = {
            (record.pair, record.condition, record.method)
            for record in experiment_records
            if record.method != TransferMethod.LOCAL_ONLY
        }
        for pair, condition, method in sorted(
            cells, key=lambda cell: (cell[0], cell[1], cell[2].value)
        ):
            dimension, setting = _boundary_dimension_and_setting(condition)
            local_only_ce = _metric_value_by_condition(
                experiment_records, pair, TransferMethod.LOCAL_ONLY, condition
            )
            if local_only_ce is None:
                local_only_ce = _metric_value_by_condition(
                    primary_transfer_records,
                    pair,
                    TransferMethod.LOCAL_ONLY,
                    PRINCIPAL_EVALUATION_CONDITION.name,
                )
            method_ce = _metric_value_by_condition(experiment_records, pair, method, condition)
            realized_gain = _realized_gain(local_only_ce, method_ce)
            certified_value = _metric_value_by_condition(
                experiment_records,
                pair,
                method,
                condition,
                metric_name=MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE,
            )
            abstention = _metric_value_by_condition(
                experiment_records,
                pair,
                method,
                condition,
                metric_name=MetricId.ABSTENTION_INDICATOR,
            )
            null_node_count = _metric_value_by_condition(
                experiment_records,
                pair,
                method,
                condition,
                metric_name=MetricId.NULL_NODE_COUNT,
            )
            confirmation_coverage = _metric_value_by_condition(
                experiment_records,
                pair,
                method,
                condition,
                metric_name=MetricId.PROPOSAL_ACCEPTANCE_RATE,
            )
            rows.append(
                OrderedDict(
                    (
                        (
                            ReportColumnName.BOUNDARY_DIMENSION,
                            dimension,
                        ),
                        (
                            ReportColumnName.SETTING,
                            setting,
                        ),
                        (
                            ReportColumnName.PAIR,
                            pair,
                        ),
                        (
                            ReportColumnName.METHOD,
                            method.value,
                        ),
                        (
                            ReportColumnName.CERTIFIED_VALUE,
                            certified_value,
                        ),
                        (
                            ReportColumnName.REALIZED_GAIN,
                            realized_gain,
                        ),
                        (
                            ReportColumnName.ABSTENTION,
                            abstention,
                        ),
                        (
                            ReportColumnName.NULL_NODE_COUNT,
                            null_node_count,
                        ),
                        (
                            ReportColumnName.CONFIRMATION_COVERAGE,
                            confirmation_coverage,
                        ),
                        (
                            ReportColumnName.STATE,
                            "Completed",
                        ),
                    )
                )
            )
    return tuple(rows)


def numeric_row_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
    x_key: ReportColumnName,
    y_key: ReportColumnName,
) -> tuple[FigureSeries, ...]:
    xs: list[Coefficient] = []
    ys: list[Coefficient] = []
    for index, row in enumerate(rows):
        y_value = row.get(y_key)
        if not isinstance(y_value, int | float):
            continue
        x_value = row.get(x_key)
        xs.append(float(x_value) if isinstance(x_value, int | float) else float(index))
        ys.append(float(y_value))
    if not xs:
        return ()
    return (FigureSeries(name=ReportSeriesName(y_key), x=tuple(xs), y=tuple(ys)),)


def sparsity_figure_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[FigureSeries, ...]:
    xs: list[Coefficient] = []
    ys: list[Coefficient] = []
    sizes: list[Coefficient] = []
    for row in rows:
        runtime = row.get(ReportColumnName.RUNTIME)
        gain = row.get(ReportColumnName.REALIZED_GAIN)
        memory = row.get(ReportColumnName.MEMORY)
        if not isinstance(runtime, int | float) or not isinstance(gain, int | float):
            continue
        xs.append(float(runtime))
        ys.append(float(gain))
        sizes.append(float(memory) if isinstance(memory, int | float) else 1.0)
    if not xs:
        return ()
    return (
        FigureSeries(
            name=ReportSeriesName("sparsity"),
            x=tuple(xs),
            y=tuple(ys),
            marker_sizes=tuple(sizes),
        ),
    )


def _confirmation_coverage(
    records: Sequence[MetricRecord],
    pair: DirectedPairName,
    accepted: Index,
    proposals: Index,
) -> Fraction | None:
    recorded = [
        record.metric_value
        for record in records
        if record.pair == pair
        and record.method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
        and record.metric_name == MetricId.COVERAGE_CONFIRM
        and record.valid
        and record.metric_value is not None
    ]
    if recorded:
        measured: Fraction = sum(recorded) / len(recorded)
        return measured
    return confirmation_coverage(accepted, proposals)


def confirmation_figure_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[FigureSeries, ...]:
    by_pair: OrderedDict[DirectedPairName, tuple[Coefficient, Coefficient, Coefficient]] = (
        OrderedDict()
    )
    for row in rows:
        pair = row.get(ReportColumnName.PAIR)
        coverage = row.get(ReportColumnName.COVERAGE)
        harm = row.get(ReportColumnName.HARMFUL_ACCEPTED_RATE)
        no_confirm_harm = row.get(ReportColumnName.NO_CONFIRM_HARMFUL_RATE)
        if not isinstance(pair, str):
            continue
        if not isinstance(coverage, int | float):
            continue
        if not isinstance(harm, int | float) or not isinstance(no_confirm_harm, int | float):
            continue
        by_pair[DirectedPairName(pair)] = (
            float(coverage),
            float(no_confirm_harm),
            float(harm),
        )
    return tuple(
        FigureSeries(
            name=ReportSeriesName(pair),
            x=(coverage,),
            y=(no_confirm_harm,),
            arrow_x=(coverage,),
            arrow_y=(harm,),
        )
        for pair, (coverage, no_confirm_harm, harm) in by_pair.items()
    )


def semantic_sufficiency_series(
    records: Sequence[MetricRecord],
) -> tuple[FigureSeries, ...]:
    orbit: OrderedDict[
        tuple[DirectedPairName, RandomSeed, TransferMethod, EvaluationConditionName], Estimate
    ] = OrderedDict()
    local_ce: OrderedDict[
        tuple[DirectedPairName, RandomSeed, EvaluationConditionName], Estimate
    ] = OrderedDict()
    method_ce: OrderedDict[
        tuple[DirectedPairName, RandomSeed, TransferMethod, EvaluationConditionName], Estimate
    ] = OrderedDict()
    for record in records:
        if not record.valid or record.metric_value is None:
            continue
        if record.metric_name == MetricId.ORBIT_SIZE:
            orbit[(record.pair, record.seed, record.method, record.condition)] = record.metric_value
        if record.metric_name == MetricId.MACRO_CROSS_ENTROPY:
            if record.method == TransferMethod.LOCAL_ONLY:
                local_ce[(record.pair, record.seed, record.condition)] = record.metric_value
            else:
                method_ce[(record.pair, record.seed, record.method, record.condition)] = (
                    record.metric_value
                )
    by_method: OrderedDict[TransferMethod, list[tuple[Coefficient, Coefficient]]] = OrderedDict()
    for key, ce in method_ce.items():
        pair, seed, method, condition = key
        baseline = local_ce.get((pair, seed, condition))
        size = orbit.get(key)
        if size is None or size <= 0.0:
            continue
        gain = _realized_gain(baseline, ce)
        if gain is None:
            continue
        by_method.setdefault(method, []).append((math.log(size), gain))
    return tuple(
        FigureSeries(
            name=ReportSeriesName(method.value),
            x=tuple(point[0] for point in points),
            y=tuple(point[1] for point in points),
        )
        for method, points in by_method.items()
        if points
    )


def failure_boundary_figure_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[FigureSeries, ...]:
    grouped: OrderedDict[ReportSeriesName, list[tuple[Coefficient, Coefficient]]] = OrderedDict()
    for index, row in enumerate(rows):
        dimension = row.get(ReportColumnName.BOUNDARY_DIMENSION)
        gain = row.get(ReportColumnName.REALIZED_GAIN)
        if not isinstance(dimension, str) or not isinstance(gain, int | float):
            continue
        setting = row.get(ReportColumnName.SETTING)
        x_value = float(setting) if isinstance(setting, int | float) else float(index)
        grouped.setdefault(ReportSeriesName(dimension), []).append((x_value, float(gain)))
    return tuple(
        FigureSeries(
            name=dimension,
            x=tuple(point[0] for point in points),
            y=tuple(point[1] for point in points),
        )
        for dimension, points in grouped.items()
        if points
    )


def scalability_figure_series(
    rows: Sequence[Mapping[ReportColumnName, TableScalar]],
) -> tuple[FigureSeries, ...]:
    return numeric_row_series(
        rows, ReportColumnName.PREDICTED_WORK, ReportColumnName.RUNTIME_MEDIAN
    )


def map_value_bound_series(
    records: Sequence[MetricRecord],
) -> tuple[FigureSeries, ...]:
    bounds: OrderedDict[tuple[EvaluationConditionName, RandomSeed], Estimate] = OrderedDict()
    values: OrderedDict[tuple[EvaluationConditionName, RandomSeed], Estimate] = OrderedDict()
    for record in records:
        if not record.valid or record.metric_value is None:
            continue
        key = (record.condition, record.seed)
        if record.metric_name == MetricId.ORBIT_RADIUS_MAP_BOUND:
            bounds[key] = record.metric_value
        if record.metric_name == MetricId.EXACT_MAP_ACTION_VALUE:
            values[key] = record.metric_value
    xs: list[Coefficient] = []
    ys: list[Coefficient] = []
    for key, bound in bounds.items():
        value = values.get(key)
        if value is None:
            continue
        xs.append(bound)
        ys.append(value)
    if not xs:
        return ()
    return (
        FigureSeries(
            name=ReportSeriesName(MetricId.EXACT_MAP_ACTION_VALUE.value),
            x=tuple(xs),
            y=tuple(ys),
        ),
    )


def evidence_status_rows(
    rows: Sequence[EvidenceStatusRow],
) -> tuple[Mapping[ReportColumnName, TableScalar], ...]:
    return tuple(
        OrderedDict(
            (
                (
                    ReportColumnName.QUESTION,
                    row.question,
                ),
                (
                    ReportColumnName.CLASSIFICATION,
                    row.classification,
                ),
                (
                    ReportColumnName.FINAL_STATE,
                    row.final_state,
                ),
                (
                    ReportColumnName.MATERIALITY_RESULT,
                    row.materiality_result,
                ),
                (
                    ReportColumnName.STATISTICAL_RESULT,
                    row.statistical_result,
                ),
                (
                    ReportColumnName.EVIDENCE_COMPLETENESS,
                    row.evidence_completeness,
                ),
                (
                    ReportColumnName.SCOPE,
                    row.scope,
                ),
                (
                    ReportColumnName.SUPPORTING_TABLE,
                    row.supporting_table,
                ),
                (
                    ReportColumnName.SUPPORTING_FIGURE,
                    row.supporting_figure,
                ),
                (
                    ReportColumnName.FORBIDDEN_WORDING,
                    row.forbidden_wording,
                ),
            )
        )
        for row in rows
    )


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
        null_reason: str | None = None
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


def transfer_ontology_and_null_padding_rows(
    store: ArtifactStore,
) -> tuple[
    Mapping[ReportColumnName, TableScalar],
    ...,
]:
    manifest = latest_completed_manifest(
        store, ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION
    )
    if manifest is None or len(manifest.payload_paths) != 1:
        return ()
    payload_path = Path(manifest.payload_paths[0])
    if not payload_path.is_file():
        return ()
    payload = cast(Mapping[str, JsonValue], json.loads(payload_path.read_text(encoding="utf-8")))
    rows: list[Mapping[ReportColumnName, TableScalar]] = []
    for pair_entry in cast(list[Mapping[str, JsonValue]], payload.get("primary_pairs", [])):
        for ontology_row in cast(
            list[Mapping[str, JsonValue]],
            pair_entry.get("transfer_ontology", []),
        ):
            support = cast(
                Mapping[str, JsonValue],
                ontology_row.get(
                    "support_counts",
                    OrderedDict[str, JsonValue](),
                ),
            )
            rows.append(
                OrderedDict(
                    (
                        (
                            ReportColumnName.CANDIDATE_CONCEPT,
                            cast(str | None, ontology_row.get("candidate_concept")),
                        ),
                        (
                            ReportColumnName.PAIR,
                            cast(str | None, ontology_row.get("pair")),
                        ),
                        (
                            ReportColumnName.COARSE_GROUP,
                            cast(str | None, ontology_row.get("coarse_group")),
                        ),
                        (
                            ReportColumnName.SOURCE_REAL_OR_NULL,
                            ("real" if ontology_row.get("source_real") else "null"),
                        ),
                        (
                            ReportColumnName.TARGET_REAL_OR_NULL,
                            ("real" if ontology_row.get("target_real") else "null"),
                        ),
                        (
                            ReportColumnName.SUPPORT_COUNTS,
                            (
                                f"source_train={support.get('source_train')},"
                                f"source_meta={support.get('source_meta')},"
                                f"target_meta={support.get('target_meta')},"
                                f"target_confirm={support.get('target_confirm')},"
                                f"target_test={support.get('target_test')}"
                            ),
                        ),
                        (
                            ReportColumnName.ACTION_ELIGIBILITY,
                            cast(bool | None, ontology_row.get("action_eligibility")),
                        ),
                        (
                            ReportColumnName.NULL_REASON,
                            cast(str | None, ontology_row.get("null_reason")),
                        ),
                    )
                )
            )
    return tuple(rows)
