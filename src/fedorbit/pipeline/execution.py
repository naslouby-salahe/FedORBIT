from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from fedorbit.analysis.synthesis import StudyAnalysis, analyse_study
from fedorbit.config.loading import section_digest
from fedorbit.config.models import ExperimentConfig, FedorbitConfig
from fedorbit.datasets.device_data import DeviceData
from fedorbit.datasets.preparation import (
    build_device,
    device_sources,
    parse_device,
    serialise_device,
    source_digests,
)
from fedorbit.infrastructure.artifacts import (
    InputDigest,
    Provenance,
    classify_artifact,
    file_digest,
    publish_artifact,
)
from fedorbit.infrastructure.runtime import code_revision, source_digest
from fedorbit.infrastructure.workspace import Workspace
from fedorbit.study.records import AnyRecord, parse_records, serialise_records
from fedorbit.study.runner import run_experiment
from fedorbit.types import (
    DatasetId,
    DeviceName,
    EvidenceState,
    ExperimentId,
    OverwritePolicy,
)

SHARED_PREPARATION_SOURCES = (
    "datasets/device_data.py",
    "datasets/preparation.py",
    "types.py",
    "infrastructure/runtime.py",
)
PREPARED_SOURCES = {
    DatasetId.NBAIOT: ("datasets/nbaiot.py", *SHARED_PREPARATION_SOURCES),
    DatasetId.GOTHAM: ("datasets/gotham.py", *SHARED_PREPARATION_SOURCES),
}
RECORD_SOURCES = (
    "datasets",
    "detection",
    "study",
    "config",
    "types.py",
    "infrastructure/runtime.py",
)
ANALYSIS_SOURCES = (
    "analysis",
    "study/records.py",
    "config",
    "types.py",
    "infrastructure/runtime.py",
)

EvidenceRows = list[AnyRecord]


class EvidenceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    repository_root: Path
    raw_directory: Path
    workspace: Workspace
    package_directory: Path


def _prepared_provenance(
    config: FedorbitConfig,
    context: ExecutionContext,
    dataset: DatasetId,
    inputs: tuple[InputDigest, ...],
) -> Provenance:
    return Provenance(
        config_digest=section_digest(config.datasets, config.base_seed),
        source_digest=source_digest(context.package_directory, PREPARED_SOURCES[dataset]),
        inputs=inputs,
    )


def prepare_dataset(
    config: FedorbitConfig,
    context: ExecutionContext,
    dataset: DatasetId,
    overwrite: OverwritePolicy,
) -> dict[DeviceName, EvidenceState]:
    log = structlog.get_logger()
    revision = code_revision(context.repository_root)
    outcome: dict[DeviceName, EvidenceState] = {}
    for device, files in device_sources(config, context.raw_directory, dataset).items():
        path = context.workspace.prepared_path(dataset, device)
        base = _prepared_provenance(config, context, dataset, ())
        state = classify_artifact(path, base, None, compare_inputs=False)
        if state is EvidenceState.VALID and overwrite is OverwritePolicy.REUSE:
            log.info("prepare", dataset=dataset.value, device=device, reuse=True)
            outcome[device] = state
            continue
        started = time.monotonic()
        data = build_device(config, context.raw_directory, dataset, device, files)
        provenance = _prepared_provenance(
            config, context, dataset, source_digests(context.raw_directory, files)
        )
        publish_artifact(
            path,
            serialise_device(data),
            len(data.support_pool) + len(data.test_benign) + len(data.test_attack),
            revision,
            provenance,
        )
        outcome[device] = EvidenceState.VALID
        log.info(
            "prepare",
            dataset=dataset.value,
            device=device,
            reuse=False,
            purged=data.purged_duplicate_rows,
            elapsed=round(time.monotonic() - started, 2),
        )
    return outcome


def load_devices(
    config: FedorbitConfig, context: ExecutionContext, dataset: DatasetId
) -> dict[DeviceName, DeviceData]:
    devices: dict[DeviceName, DeviceData] = {}
    base = _prepared_provenance(config, context, dataset, ())
    for device in device_sources(config, context.raw_directory, dataset):
        path = context.workspace.prepared_path(dataset, device)
        state = classify_artifact(path, base, None, compare_inputs=False)
        if state is not EvidenceState.VALID:
            raise EvidenceUnavailableError(
                f"prepared data for {dataset.value}/{device} is {state.value}; run preprocess"
            )
        devices[device] = parse_device(dataset, device, path.read_bytes())
    return devices


def _prepared_inputs(
    config: FedorbitConfig, context: ExecutionContext, dataset: DatasetId
) -> tuple[InputDigest, ...]:
    return tuple(
        InputDigest(
            name=f"{dataset.value}/{device}",
            sha256=file_digest(context.workspace.prepared_path(dataset, device)),
        )
        for device in sorted(device_sources(config, context.raw_directory, dataset))
    )


def _records_provenance(
    config: FedorbitConfig, context: ExecutionContext, experiment: ExperimentConfig
) -> Provenance:
    return Provenance(
        config_digest=section_digest(
            experiment,
            config.detectors,
            config.operating_point,
            config.partner_similarity,
            config.base_seed,
        ),
        source_digest=source_digest(context.package_directory, RECORD_SOURCES),
        inputs=_prepared_inputs(config, context, experiment.dataset),
    )


def experiment_state(
    config: FedorbitConfig, context: ExecutionContext, experiment: ExperimentConfig
) -> EvidenceState:
    path = context.workspace.records_path(experiment.id)
    try:
        provenance = _records_provenance(config, context, experiment)
    except FileNotFoundError:
        return EvidenceState.MISSING if not path.exists() else EvidenceState.STALE
    return classify_artifact(path, provenance, None, compare_inputs=True)


def execute_experiment(
    config: FedorbitConfig,
    context: ExecutionContext,
    identifier: ExperimentId,
    overwrite: OverwritePolicy,
) -> EvidenceState:
    experiment = config.experiment(identifier)
    log = structlog.get_logger()
    state = experiment_state(config, context, experiment)
    if state is EvidenceState.VALID and overwrite is OverwritePolicy.REUSE:
        log.info("experiment", experiment=identifier.value, reuse=True)
        return state
    devices = load_devices(config, context, experiment.dataset)
    started = time.monotonic()
    rows = run_experiment(config, experiment, devices)
    publish_artifact(
        context.workspace.records_path(identifier),
        serialise_records(rows),
        len(rows),
        code_revision(context.repository_root),
        _records_provenance(config, context, experiment),
    )
    log.info(
        "experiment",
        experiment=identifier.value,
        reuse=False,
        records=len(rows),
        elapsed=round(time.monotonic() - started, 2),
    )
    return EvidenceState.VALID


def load_verified_records(
    config: FedorbitConfig, context: ExecutionContext, experiment: ExperimentConfig
) -> EvidenceRows:
    state = experiment_state(config, context, experiment)
    if state is not EvidenceState.VALID:
        raise EvidenceUnavailableError(f"records for {experiment.id.value} are {state.value}")
    return parse_records(context.workspace.records_path(experiment.id).read_bytes())


def analysis_provenance(config: FedorbitConfig, context: ExecutionContext) -> Provenance:
    inputs = tuple(
        InputDigest(
            name=experiment.id.value,
            sha256=file_digest(context.workspace.records_path(experiment.id)),
        )
        for experiment in config.experiments
    )
    return Provenance(
        config_digest=section_digest(config),
        source_digest=source_digest(context.package_directory, ANALYSIS_SOURCES),
        inputs=inputs,
    )


def analysis_state(config: FedorbitConfig, context: ExecutionContext) -> EvidenceState:
    for experiment in config.experiments:
        if experiment_state(config, context, experiment) is not EvidenceState.VALID:
            return EvidenceState.INCOMPLETE
    return classify_artifact(
        context.workspace.analysis_path(),
        analysis_provenance(config, context),
        None,
        compare_inputs=True,
    )


def execute_analysis(
    config: FedorbitConfig, context: ExecutionContext, overwrite: OverwritePolicy
) -> StudyAnalysis:
    records = {
        experiment.id: load_verified_records(config, context, experiment)
        for experiment in config.experiments
    }
    path = context.workspace.analysis_path()
    if (
        overwrite is OverwritePolicy.REUSE
        and analysis_state(config, context) is EvidenceState.VALID
    ):
        return StudyAnalysis.model_validate_json(path.read_text(encoding="utf-8"))
    analysis = analyse_study(config, records)
    publish_artifact(
        path,
        analysis.model_dump_json(indent=2).encode("utf-8"),
        len(analysis.contrasts),
        code_revision(context.repository_root),
        analysis_provenance(config, context),
    )
    return analysis


def build_context(
    config: FedorbitConfig,
    repository_root: Path,
    package_directory: Path,
    workspace_directory: Path,
    results_directory: Path,
) -> ExecutionContext:
    return ExecutionContext(
        repository_root=repository_root,
        raw_directory=repository_root / config.datasets.raw_directory,
        workspace=Workspace(root=workspace_directory, results_directory=results_directory),
        package_directory=package_directory,
    )
