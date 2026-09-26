from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import structlog

from fedorbit.config.models import ConditionSpec, ExperimentConfig, FedorbitConfig
from fedorbit.datasets.device_data import DeviceData
from fedorbit.detection.metrics import (
    OperatingPointMetrics,
    auroc,
    operating_point,
    threshold_at_rate,
    worst_case_auroc_standard_error,
)
from fedorbit.detection.moments import summarise
from fedorbit.infrastructure.runtime import derive_seed, require_cuda
from fedorbit.study.channels import (
    AutoencoderInputs,
    SharedSummaries,
    build_autoencoder_scorer,
    build_gaussian_scorer,
)
from fedorbit.study.partners import PartnerSelectionContext, select_partners
from fedorbit.study.records import (
    AnyRecord,
    CellRecord,
    IneligibleDeviceRecord,
    InfeasibleSupportRecord,
)
from fedorbit.types import (
    ConditionName,
    DetectorKind,
    DeviceName,
    FloatMatrix,
    FloatVector,
    RandomPurpose,
    ReplicateIndex,
    ShareChannel,
    SupportSize,
)

EvidenceRows = list[AnyRecord]
Scorer = Callable[[FloatMatrix], FloatVector]
ScorerBuilder = Callable[[FloatMatrix], Scorer]


@dataclass(frozen=True, slots=True)
class ScaleDiagnostics:
    scale_mismatch: float
    split_half_instability: float


def log_scale_gap(first: FloatVector, second: FloatVector, floor: float) -> float:
    return float(np.median(np.abs(np.log((first + floor) / (second + floor)))))


def scale_diagnostics(rows: FloatMatrix, full_pool: FloatMatrix, floor: float) -> ScaleDiagnostics:
    half = len(rows) // 2
    return ScaleDiagnostics(
        scale_mismatch=log_scale_gap(rows.std(axis=0), full_pool.std(axis=0), floor),
        split_half_instability=log_scale_gap(
            rows[:half].std(axis=0), rows[half:].std(axis=0), floor
        ),
    )


def cross_fit_threshold(build: ScorerBuilder, rows: FloatMatrix, nominal_rate: float) -> float:
    half = len(rows) // 2
    first, second = rows[:half], rows[half:]
    held_out = np.concatenate([build(second)(first), build(first)(second)])
    return threshold_at_rate(held_out, nominal_rate)


def _shared_for(
    spec: ConditionSpec, context: PartnerSelectionContext
) -> tuple[SharedSummaries | None, tuple[DeviceName, ...]]:
    if spec.partner_policy is None:
        return None, ()
    selected = select_partners(spec, context)
    return SharedSummaries(tuple(context.partners[name] for name in selected)), selected


class ExperimentRunner:
    def __init__(
        self,
        config: FedorbitConfig,
        experiment: ExperimentConfig,
        devices: Mapping[DeviceName, DeviceData],
    ) -> None:
        self._config = config
        self._experiment = experiment
        self._devices = devices
        self._log = structlog.get_logger()
        self._torch_device = (
            require_cuda() if experiment.detector is DetectorKind.AUTOENCODER else None
        )
        self._summaries = {name: summarise(data.support_pool) for name, data in devices.items()}
        self._pools = {name: data.support_pool for name, data in devices.items()}

    def run(self) -> EvidenceRows:
        records: EvidenceRows = []
        limit = self._config.statistics.maximum_auroc_standard_error
        for name in sorted(self._devices):
            data = self._devices[name]
            if not data.is_evaluation_target:
                continue
            standard_error = worst_case_auroc_standard_error(
                len(data.test_benign), len(data.test_attack)
            )
            if standard_error > limit:
                records.append(
                    IneligibleDeviceRecord(
                        experiment=self._experiment.id,
                        dataset=data.dataset,
                        device=data.device,
                        benign_rows=len(data.test_benign),
                        attack_rows=len(data.test_attack),
                        worst_case_standard_error=standard_error,
                    )
                )
                continue
            records.extend(self._run_device(data))
            self._log.info(
                "device-complete",
                experiment=self._experiment.id.value,
                dataset=data.dataset.value,
                device=data.device,
            )
        return records

    def _run_device(self, target: DeviceData) -> EvidenceRows:
        experiment = self._experiment
        records: EvidenceRows = []
        pool_size = len(target.support_pool)
        for support_size in experiment.support_sizes:
            if pool_size < support_size:
                records.append(
                    InfeasibleSupportRecord(
                        experiment=experiment.id,
                        dataset=target.dataset,
                        device=target.device,
                        support_size=support_size,
                        available_rows=pool_size,
                    )
                )
                continue
            for replicate in range(experiment.replicates):
                records.extend(self._run_cell(target, support_size, ReplicateIndex(replicate)))
        records.extend(self._run_reference(target))
        return records

    def _partner_context(
        self, target: DeviceData, rows: FloatMatrix, seed_coordinates: tuple[str, ...]
    ) -> PartnerSelectionContext:
        base = self._config.base_seed
        return PartnerSelectionContext(
            partners={name: s for name, s in self._summaries.items() if name != target.device},
            deployable_reference_scale=rows.std(axis=0),
            oracle_reference_scale=target.support_pool.std(axis=0),
            similarity_scale_floor=self._config.partner_similarity.scale_floor,
            seed=derive_seed(base, RandomPurpose.PARTNER_SELECTION, *seed_coordinates),
        )

    def _run_cell(
        self, target: DeviceData, support_size: SupportSize, replicate: ReplicateIndex
    ) -> EvidenceRows:
        base = self._config.base_seed
        coordinates = (target.dataset.value, target.device, str(support_size), str(replicate))
        window_generator = np.random.default_rng(
            derive_seed(base, RandomPurpose.SUPPORT_WINDOW, *coordinates)
        )
        start = int(window_generator.integers(0, len(target.support_pool) - support_size + 1))
        rows = target.support_pool[start : start + support_size]
        diagnostics = scale_diagnostics(
            rows, target.support_pool, self._config.detectors.gaussian.minimum_scale
        )
        context = self._partner_context(target, rows, coordinates)
        records: EvidenceRows = []
        for name in sorted(self._experiment.conditions):
            spec = self._experiment.conditions[name]
            if spec.channel is ShareChannel.LOCAL_FULL_SUPPORT:
                continue
            records.append(
                self._evaluate(
                    target,
                    name,
                    spec,
                    rows,
                    context,
                    support_size,
                    replicate,
                    diagnostics,
                    coordinates,
                )
            )
        return records

    def _run_reference(self, target: DeviceData) -> EvidenceRows:
        records: EvidenceRows = []
        pool = target.support_pool
        diagnostics = ScaleDiagnostics(0.0, 0.0)
        coordinates = (target.dataset.value, target.device, "reference")
        context = self._partner_context(target, pool, coordinates)
        for name in sorted(self._experiment.conditions):
            spec = self._experiment.conditions[name]
            if spec.channel is ShareChannel.LOCAL_FULL_SUPPORT:
                records.append(
                    self._evaluate(
                        target,
                        name,
                        spec,
                        pool,
                        context,
                        SupportSize(len(pool)),
                        ReplicateIndex(0),
                        diagnostics,
                        coordinates,
                    )
                )
        return records

    def _evaluate(
        self,
        target: DeviceData,
        name: ConditionName,
        spec: ConditionSpec,
        rows: FloatMatrix,
        context: PartnerSelectionContext,
        support_size: SupportSize,
        replicate: ReplicateIndex,
        diagnostics: ScaleDiagnostics,
        coordinates: tuple[str, ...],
    ) -> CellRecord:
        base = self._config.base_seed
        shared, selected = _shared_for(spec, context)
        detectors = self._config.detectors
        operating: OperatingPointMetrics | None = None
        if self._experiment.detector is DetectorKind.GAUSSIAN:

            def build(subset: FloatMatrix) -> Scorer:
                return build_gaussian_scorer(spec, subset, shared, detectors.gaussian).score

            scorer = build(rows)
            benign_scores = scorer(target.test_benign)
            attack_scores = scorer(target.test_attack)
            if self._experiment.report_operating_point:
                threshold = cross_fit_threshold(
                    build, rows, self._config.operating_point.nominal_false_positive_rate
                )
                operating = operating_point(threshold, benign_scores, attack_scores)
        else:
            if self._torch_device is None:
                raise RuntimeError("autoencoder experiments need an execution device")
            inputs = AutoencoderInputs(
                rows=rows,
                shared=shared,
                partner_pools={name: self._pools[name] for name in selected},
                selected_partners=selected,
                initialisation_seed=derive_seed(
                    base, RandomPurpose.NETWORK_INITIALISATION, *coordinates
                ),
                partner_sample_seed=derive_seed(base, RandomPurpose.PARTNER_SAMPLE, *coordinates),
            )
            autoencoder = build_autoencoder_scorer(
                spec, inputs, detectors.gaussian, detectors.autoencoder, self._torch_device
            )
            benign_scores = autoencoder.score(target.test_benign)
            attack_scores = autoencoder.score(target.test_attack)
        return CellRecord(
            experiment=self._experiment.id,
            dataset=target.dataset,
            device=target.device,
            support_size=support_size,
            replicate=replicate,
            condition=name,
            auroc=auroc(benign_scores, attack_scores),
            false_positive_rate=None if operating is None else operating.false_positive_rate,
            true_positive_rate=None if operating is None else operating.true_positive_rate,
            scale_mismatch=diagnostics.scale_mismatch,
            split_half_instability=diagnostics.split_half_instability,
        )


def run_experiment(
    config: FedorbitConfig, experiment: ExperimentConfig, devices: Mapping[DeviceName, DeviceData]
) -> EvidenceRows:
    return ExperimentRunner(config, experiment, devices).run()
