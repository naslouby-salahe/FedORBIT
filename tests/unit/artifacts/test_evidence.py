from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import cast

from fedorbit.analysis.records import MetricDirection, MetricRecord
from fedorbit.config.loading import active_config
from fedorbit.config.models import ReportingPrecisionConfig
from fedorbit.datasets.common import file_sha256
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.evidence import (
    EXPORT_LEDGER_KEY,
    EvidenceFigure,
    EvidenceTable,
    FigureSeries,
    VerifiedEvidenceWriter,
    export_dependency_fingerprint,
    figure_svg_bytes,
    format_balanced_accuracy,
    format_integer,
    format_interval_estimate,
    format_macro_f1,
    format_memory_mib,
    format_metric_value,
    format_p_value,
    format_runtime_seconds,
    format_scientific_metric,
)
from fedorbit.infrastructure.manifests import (
    CompletionManifest,
    ReusableArtifactManifest,
    artifact_id,
    completion_manifest_self_hash,
)
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import (
    PRINCIPAL_EVALUATION_CONDITION,
    ArtifactIdentifier,
    ArtifactSchemaVersion,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    CompletionValidationState,
    DirectedPairName,
    ExperimentName,
    FieldDescription,
    MetricId,
    MetricUnit,
    ReportArtifactName,
    ReportAxisLabel,
    ReportColumnName,
    ReportSeriesName,
    Sha256Digest,
    StableJsonPayload,
    TerminalState,
    TransferMethod,
)

FINGERPRINT = Sha256Digest("a" * 64)
CONFIGURATION_DIGEST = Sha256Digest("b" * 64)
CODE_DIGEST = Sha256Digest("c" * 64)
COMMIT = "e" * 40
ENVIRONMENT_DIGEST = Sha256Digest("f" * 64)


def _precision() -> ReportingPrecisionConfig:
    return active_config().reporting.precision


def _sample_figure() -> EvidenceFigure:
    return EvidenceFigure(
        x_label=ReportAxisLabel("runtime"),
        y_label=ReportAxisLabel("realized gain"),
        series=(
            FigureSeries(
                name=ReportSeriesName("s=2"),
                x=(1.0, 2.0),
                y=(3.0, 4.0),
                y_low=(2.5, 3.5),
                y_high=(3.5, 4.5),
            ),
        ),
        vertical_reference_lines=(0.0,),
        horizontal_reference_lines=(0.0,),
    )


def _metric_record(
    metric_name: MetricId = MetricId.MACRO_CROSS_ENTROPY,
    metric_unit: MetricUnit = MetricUnit.NATS,
    metric_value: float = 1.234567,
) -> MetricRecord:
    return MetricRecord(
        experiment=ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        pair=DirectedPairName("edge_iiotset_network -> ton_iot_network"),
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        condition=PRINCIPAL_EVALUATION_CONDITION.name,
        seed=1103,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        direction=MetricDirection.LOWER_IS_BETTER,
        evaluation_class_set_sha256=FINGERPRINT,
        input_artifact_ids=(ArtifactIdentifier("upstream"),),
        dependency_fingerprint_sha256=FINGERPRINT,
        valid=True,
        invalid_reason=None,
    )


def _write_metric_artifact(root: Path, metric: MetricRecord) -> ArtifactIdentifier:
    store = ArtifactStore(root)
    payload_path = root / "payloads" / "metric.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(
        json.dumps(OrderedDict(metric_record=metric.model_dump(mode="json"))),
        encoding="utf-8",
    )
    payload_sha256 = file_sha256(payload_path)
    draft = CompletionManifest.model_validate(
        OrderedDict(
            schema_version=ArtifactSchemaVersion.V1,
            semantic_experiment_coordinates="{}",
            producer_stage=ArtifactStage.EVALUATION.value,
            terminal_state=TerminalState.COMPLETED.value,
            dependency_fingerprint_sha256=FINGERPRINT,
            upstream_artifact_ids=(),
            mandatory_artifact_paths=(str(payload_path),),
            mandatory_artifact_sha256=payload_sha256,
            scientific_configuration_sha256=CONFIGURATION_DIGEST,
            relevant_code_sha256=CODE_DIGEST,
            upstream_lineage="{}",
            completion_validation_state=CompletionValidationState.VALIDATED.value,
            completion_written_last=True,
            completion_manifest_sha256="",
        )
    )
    completion = draft.model_copy(
        update=OrderedDict(completion_manifest_sha256=completion_manifest_self_hash(draft))
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactType.PREDICTION,
                cast(StableJsonPayload, OrderedDict(coordinates="{}")),
                FINGERPRINT,
            ),
            artifact_type=ArtifactType.PREDICTION.value,
            semantic_producer_coordinates="{}",
            producer_stage=ArtifactStage.EVALUATION.value,
            dependency_fingerprint_sha256=FINGERPRINT,
            upstream_artifact_ids=(),
            applicable_configuration_sha256=CONFIGURATION_DIGEST,
            relevant_code_sha256=CODE_DIGEST,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_sha256,
            schema_version=ArtifactSchemaVersion.V1,
            created_git_commit=COMMIT,
            created_environment_sha256=ENVIRONMENT_DIGEST,
            state=ArtifactState.COMPLETED.value,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
    )
    store.write_completed(manifest, completion)
    return manifest.artifact_id


def test_figure_svg_bytes_are_deterministic_across_repeated_renders() -> None:
    figure = _sample_figure()
    first = figure_svg_bytes(figure)
    second = figure_svg_bytes(figure)
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_canonical_figure_svg_suppresses_nondeterministic_creation_metadata() -> None:
    rendered = figure_svg_bytes(_sample_figure())
    assert b"dc:date" not in rendered
    assert b"dc:creator" not in rendered


def test_project_evidence_figure_bytes_are_stable_across_writes(tmp_path: Path) -> None:
    layout = build_layout(tmp_path)
    writer = VerifiedEvidenceWriter(ArtifactStore(layout.execution_root), layout)
    first = writer.write_project_evidence_figure(
        _sample_figure(), ReportArtifactName.SCALABILITY_FIGURE
    )
    initial = first.read_bytes()
    second = writer.write_project_evidence_figure(
        _sample_figure(), ReportArtifactName.SCALABILITY_FIGURE
    )
    assert second == first
    assert second.read_bytes() == initial


def test_project_evidence_table_is_written_as_canonical_csv(tmp_path: Path) -> None:
    layout = build_layout(tmp_path)
    writer = VerifiedEvidenceWriter(ArtifactStore(layout.execution_root), layout)
    table = EvidenceTable(
        columns=(ReportColumnName.PAIR, ReportColumnName.REALIZED_GAIN),
        rows=(("edge_iiotset_network -> ton_iot_network", "0.1235"),),
    )
    destination = writer.write_project_evidence_table(table, ReportArtifactName.ABLATION_RESULTS)
    assert destination.name == f"{ReportArtifactName.ABLATION_RESULTS}.csv"
    assert destination.parent.name == "main"
    rendered = destination.read_text(encoding="utf-8")
    assert rendered.startswith("pair,realized_gain\n")
    assert rendered.endswith("\n")
    assert "0.1235" in rendered


def test_scientific_metric_uses_configured_decimals() -> None:
    assert format_scientific_metric(0.123456) == "0.1235"
    assert format_scientific_metric(1.0) == "1.0000"


def test_macro_f1_and_balanced_accuracy_use_configured_decimals() -> None:
    assert format_macro_f1(0.987654) == "0.9877"
    assert format_balanced_accuracy(0.5) == "0.5000"


def test_p_value_renders_less_than_literal_below_the_configured_threshold() -> None:
    precision = _precision()
    assert format_p_value(precision.p_value_less_than_threshold / 2.0) == "<0.0001"
    assert format_p_value(0.012345) == "0.0123"


def test_runtime_seconds_and_memory_use_configured_decimals() -> None:
    assert format_runtime_seconds(12.3456) == "12.346"
    assert format_memory_mib(64.25) == "64.2"


def test_counts_are_rendered_as_integers() -> None:
    assert format_integer(7.0) == "7"
    assert format_integer(0.0) == "0"


def test_interval_estimate_renders_estimate_with_bounds() -> None:
    assert format_interval_estimate(0.12345, 0.1, 0.15) == "0.1235 [0.1000, 0.1500]"
    assert format_interval_estimate(0.1234, None, None) == "0.1234"


def test_metric_value_is_formatted_by_registered_unit_and_name() -> None:
    assert format_metric_value(_metric_record()) == "1.2346"
    macro_f1 = _metric_record(MetricId.MACRO_F1, MetricUnit.FRACTION, 0.987654)
    assert format_metric_value(macro_f1) == "0.9877"
    balanced = _metric_record(MetricId.BALANCED_ACCURACY, MetricUnit.FRACTION, 0.5)
    assert format_metric_value(balanced) == "0.5000"
    runtime = _metric_record(MetricId.WALL_TIME, MetricUnit.SECONDS, 12.3456)
    assert format_metric_value(runtime) == "12.346"
    memory = _metric_record(MetricId.PEAK_HOST_RSS, MetricUnit.MEBIBYTES, 64.25)
    assert format_metric_value(memory) == "64.2"
    count = _metric_record(MetricId.LAP_CALLS, MetricUnit.COUNT, 7.0)
    assert format_metric_value(count) == "7"


def test_unavailable_metric_value_is_rendered_as_empty() -> None:
    unavailable = _metric_record().model_copy(update=OrderedDict(metric_value=None))
    assert format_metric_value(unavailable) == ""


def test_metric_exports_write_only_registered_canonical_formats(tmp_path: Path) -> None:
    metric = _metric_record()
    store_root = tmp_path / "outputs"
    artifact_identifier = _write_metric_artifact(store_root, metric)
    layout = build_layout(tmp_path)
    writer = VerifiedEvidenceWriter(ArtifactStore(layout.execution_root), layout)
    written = writer.write_metric_exports(metric.experiment, artifact_identifier)
    assert tuple(path.suffix for path in written) == (".json", ".csv", ".svg")
    assert all(path.is_file() for path in written)
    assert not any(path.suffix in {".pdf", ".tex"} for path in written)


def test_metric_records_csv_uses_registered_precision(tmp_path: Path) -> None:
    metric = _metric_record()
    store_root = tmp_path / "outputs"
    artifact_identifier = _write_metric_artifact(store_root, metric)
    layout = build_layout(tmp_path)
    writer = VerifiedEvidenceWriter(ArtifactStore(layout.execution_root), layout)
    written = writer.write_metric_exports(metric.experiment, artifact_identifier)
    csv_path = next(path for path in written if path.suffix == ".csv")
    rendered = csv_path.read_text(encoding="utf-8").splitlines()
    header = rendered[0].split(",")
    row = rendered[1].split(",")
    assert row[header.index("metric_value")] == "1.2346"
    assert row[header.index("metric_unit")] == MetricUnit.NATS.value


def test_export_dependency_fingerprint_is_definition_and_dependency_sensitive() -> None:
    name = ReportArtifactName.ABLATION_RESULTS
    baseline = export_dependency_fingerprint(name, (ArtifactIdentifier("metric-a"),))
    assert export_dependency_fingerprint(name, (ArtifactIdentifier("metric-a"),)) == baseline
    assert export_dependency_fingerprint(name, (ArtifactIdentifier("metric-b"),)) != baseline
    assert export_dependency_fingerprint(name) != baseline
    other = export_dependency_fingerprint(
        ReportArtifactName.SCALABILITY_FIGURE, (ArtifactIdentifier("metric-a"),)
    )
    assert other != baseline


def test_figure_svg_carries_the_export_description_deterministically() -> None:
    description = FieldDescription("dependency-fingerprint-sha256: " + "0" * 64)
    first = figure_svg_bytes(_sample_figure(), description)
    second = figure_svg_bytes(_sample_figure(), description)
    assert first == second
    assert b"dc:description" in first
    assert b"dc:date" not in first
    assert first != figure_svg_bytes(_sample_figure())


def test_project_exports_record_fingerprints_and_reuse_identical_content(tmp_path: Path) -> None:
    layout = build_layout(tmp_path)
    writer = VerifiedEvidenceWriter(ArtifactStore(layout.execution_root), layout)
    table = EvidenceTable(
        columns=(ReportColumnName.PAIR, ReportColumnName.REALIZED_GAIN),
        rows=(("edge_iiotset_network -> ton_iot_network", "0.1235"),),
    )
    name = ReportArtifactName.ABLATION_RESULTS
    dependencies = (ArtifactIdentifier("metric-a"),)
    first = writer.write_project_evidence_table(table, name, dependencies)
    stamp = first.stat().st_mtime_ns
    reused = writer.write_project_evidence_table(table, name, dependencies)
    assert reused == first
    assert reused.stat().st_mtime_ns == stamp
    assert (
        b"dc:date"
        not in writer.write_project_evidence_figure(
            _sample_figure(), name, dependencies
        ).read_bytes()
    )
    ledger = json.loads(
        (layout.project_summary / "reproducibility" / "execution" / "execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert ledger[EXPORT_LEDGER_KEY][name.value] == export_dependency_fingerprint(
        name, dependencies
    )
