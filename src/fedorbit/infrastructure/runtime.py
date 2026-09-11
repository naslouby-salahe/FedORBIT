from __future__ import annotations

import hashlib
import resource
import subprocess
import time
from collections import OrderedDict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

import numpy as np
import psutil
import structlog
import torch
from structlog.typing import FilteringBoundLogger

from fedorbit.config.loading import active_config, repository_root
from fedorbit.infrastructure.environment import EnvironmentSnapshot
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactState,
    ByteCount,
    DatasetId,
    DerivedSeed,
    ElapsedSeconds,
    ExecutionStageName,
    ExperimentName,
    GitRevision,
    MemoryMib,
    RandomSeed,
    ReuseDecision,
    RngNamespace,
    SemanticCoordinates,
    SerializedPacket,
    Sha256Digest,
    StableJsonPayload,
    TorchPrecision,
    stable_json,
)

OBSERVED_PEAK_MEMORY_TO_RAW_BYTES_RATIO = 35.0
MAXIMUM_MEMORY_BUDGET_FRACTION = 0.65


@dataclass(frozen=True, slots=True)
class MemoryBudgetEstimate:
    raw_bytes: ByteCount
    estimated_peak_bytes: ByteCount
    available_bytes: ByteCount
    budget_bytes: ByteCount

    @property
    def within_budget(self) -> bool:
        return self.estimated_peak_bytes <= self.budget_bytes


def estimate_memory_budget(raw_bytes: ByteCount) -> MemoryBudgetEstimate:
    available_bytes: ByteCount = psutil.virtual_memory().available
    estimated_peak_bytes: ByteCount = round(raw_bytes * OBSERVED_PEAK_MEMORY_TO_RAW_BYTES_RATIO)
    budget_bytes: ByteCount = round(available_bytes * MAXIMUM_MEMORY_BUDGET_FRACTION)
    return MemoryBudgetEstimate(
        raw_bytes=raw_bytes,
        estimated_peak_bytes=estimated_peak_bytes,
        available_bytes=available_bytes,
        budget_bytes=budget_bytes,
    )


_set_deterministic_algorithms = cast(Callable[[bool], None], torch.use_deterministic_algorithms)


class PrincipalDeterminismError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DeterministicBackendState:
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    matmul_fp32_precision: TorchPrecision
    conv_fp32_precision: TorchPrecision
    stochastic_rounding: bool
    default_dtype: TorchPrecision
    float32_matmul_precision: TorchPrecision


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise PrincipalDeterminismError(
            "principal execution requires CUDA; no CUDA device is available"
        )


def apply_deterministic_backend(require_cuda_device: bool = True) -> None:
    if require_cuda_device:
        require_cuda()
    _set_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    matmul_precision = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    if matmul_precision is not None:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    conv = getattr(torch.backends.cudnn, "conv", None)
    if conv is not None:
        conv_precision = getattr(conv, "fp32_precision", None)
        if conv_precision is not None:
            conv.fp32_precision = "ieee"
    stochastic_rounding = getattr(torch.backends.cuda.matmul, "stochastic_rounding", None)
    if stochastic_rounding is not None:
        torch.backends.cuda.matmul.stochastic_rounding = False
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")


@contextmanager
def principal_determinism() -> Generator[None]:
    apply_deterministic_backend(require_cuda_device=True)
    yield


@contextmanager
def test_determinism() -> Generator[None]:
    apply_deterministic_backend(require_cuda_device=False)
    yield


@dataclass(slots=True)
class EfficiencyMeasurement:
    wall_time_seconds: ElapsedSeconds = 0.0
    peak_host_rss_mib: MemoryMib = 0.0
    peak_cuda_allocated_bytes: ByteCount = 0


@dataclass(frozen=True, slots=True)
class _EfficiencyMeasurementHandle:
    result: EfficiencyMeasurement = field(default_factory=EfficiencyMeasurement)


def _peak_host_rss_mib() -> MemoryMib:
    rss: MemoryMib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return rss


@contextmanager
def measure_efficiency() -> Generator[_EfficiencyMeasurementHandle]:
    handle = _EfficiencyMeasurementHandle()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started_at = time.monotonic()
    try:
        yield handle
    finally:
        handle.result.wall_time_seconds = time.monotonic() - started_at
        handle.result.peak_host_rss_mib = _peak_host_rss_mib()
        if torch.cuda.is_available():
            handle.result.peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated()


def _conv_fp32_precision() -> TorchPrecision:
    conv = getattr(torch.backends.cudnn, "conv", None)
    if conv is not None:
        precision = getattr(conv, "fp32_precision", None)
        if precision is not None:
            return TorchPrecision(str(precision))
    return TorchPrecision("absent")


def _matmul_fp32_precision() -> TorchPrecision:
    precision = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    if precision is not None:
        return TorchPrecision(str(precision))
    return TorchPrecision("absent")


def _stochastic_rounding() -> bool:
    if hasattr(torch.backends.cuda.matmul, "stochastic_rounding"):
        return bool(torch.backends.cuda.matmul.stochastic_rounding)
    return False


def deterministic_backend_state() -> DeterministicBackendState:
    return DeterministicBackendState(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        matmul_fp32_precision=_matmul_fp32_precision(),
        conv_fp32_precision=_conv_fp32_precision(),
        stochastic_rounding=_stochastic_rounding(),
        default_dtype=TorchPrecision(str(torch.get_default_dtype())),
        float32_matmul_precision=TorchPrecision(torch.get_float32_matmul_precision()),
    )


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def assert_float32_training(dtype: torch.dtype) -> None:
    if dtype != torch.float32:
        raise PrincipalDeterminismError(f"principal training must remain float32; found {dtype}")


@dataclass(frozen=True, slots=True)
class ExecutionLogEvent:
    occurred_at: datetime
    cell_coordinates: SemanticCoordinates
    artifact_id: ArtifactIdentifier | None
    state: ArtifactState
    stage: ExecutionStageName | None = None
    experiment: ExperimentName | None = None
    dataset: DatasetId | None = None
    seed: RandomSeed | None = None
    elapsed_seconds: ElapsedSeconds | None = None
    reuse_decision: ReuseDecision | None = None


class ExecutionLogger:
    def __init__(self, logger: FilteringBoundLogger) -> None:
        self._logger = logger

    def record(self, event: ExecutionLogEvent) -> None:
        self._logger.info(
            "execution_event",
            occurred_at=event.occurred_at.isoformat(),
            cell_coordinates=event.cell_coordinates.value,
            artifact_id=event.artifact_id.value if event.artifact_id is not None else None,
            state=event.state.value,
            stage=event.stage,
            experiment=event.experiment.value if event.experiment is not None else None,
            dataset=event.dataset.value if event.dataset is not None else None,
            seed=event.seed,
            elapsed_seconds=event.elapsed_seconds,
            reuse_decision=event.reuse_decision,
        )

    def event(self, event_name: str #TODO: should be enum not hardcoded string
              , **fields: object) -> None:
        self._logger.info(event_name, **fields)


def execution_logger() -> ExecutionLogger:
    return ExecutionLogger(cast(FilteringBoundLogger, structlog.get_logger("fedorbit.execution")))


class IncompatibleIdentityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CodeRevision:
    commit: GitRevision

    def identity(self) -> GitRevision:
        return self.commit


def _git_head() -> GitRevision:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return GitRevision("no-git")
    return GitRevision(result.stdout.strip())


def current_code_revision() -> CodeRevision:
    return CodeRevision(commit=_git_head())


def _seed_digest() -> Sha256Digest:
    randomness = active_config().scientific.randomness
    return Sha256Digest(
        hashlib.sha256(
            stable_json(
                cast(
                    StableJsonPayload,
                    OrderedDict(
                        pilot_seeds=list(randomness.pilot_seeds),
                        confirmatory_seeds=list(randomness.confirmatory_seeds),
                        statistical_seed=randomness.statistical_seed,
                    ),
                )
            ).encode("utf-8")
        ).hexdigest()
    )


@dataclass(frozen=True, slots=True)
class ReproducibilityIdentity:
    config_digest: Sha256Digest
    seed_digest: Sha256Digest
    environment_fingerprint: Sha256Digest
    code_revision: CodeRevision
    statistical_identity_digest: Sha256Digest

    def fingerprint(self) -> Sha256Digest:
        return Sha256Digest(
            hashlib.sha256(
                "|".join(
                    (
                        self.config_digest,
                        self.seed_digest,
                        self.statistical_identity_digest,
                    )
                ).encode("utf-8")
            ).hexdigest()
        )


def statistical_identity_digest() -> Sha256Digest:
    scientific = active_config().scientific
    statistics = scientific.statistics
    payload = stable_json(
        cast(
            StableJsonPayload,
            OrderedDict(
                dataset_ids=list(scientific.datasets.clients),
                primary_pairs=[
                    [pair.source.value, pair.target.value]
                    for pair in scientific.datasets.primary_directed_pairs
                ],
                secondary_pairs=[
                    [pair.source.value, pair.target.value]
                    for pair in scientific.datasets.secondary_directed_pairs
                ],
                split=scientific.split.duplicate_safe_chronological_intervals.model_dump(
                    mode="json"
                ),
                preprocessing=scientific.preprocessing.model_dump(mode="json"),
                seeds=_seed_digest(),
                confidence_level=statistics.confidence_level,
                evaluation_criteria=scientific.evaluation_criteria.model_dump(mode="json"),
                materiality=scientific.materiality.model_dump(mode="json"),
            ),
        )
    )
    return Sha256Digest(hashlib.sha256(payload.encode("utf-8")).hexdigest())


def build_reproducibility_identity(environment: EnvironmentSnapshot) -> ReproducibilityIdentity:
    code_revision = current_code_revision()
    config = active_config()
    return ReproducibilityIdentity(
        config_digest=Sha256Digest(
            hashlib.sha256(stable_json(config.model_dump(mode="json")).encode("utf-8")).hexdigest()
        ),
        seed_digest=_seed_digest(),
        environment_fingerprint=environment.fingerprint_sha256,
        code_revision=code_revision,
        statistical_identity_digest=statistical_identity_digest(),
    )


def compatible(current: ReproducibilityIdentity, recorded: ReproducibilityIdentity) -> bool:
    return current.fingerprint() == recorded.fingerprint()


def reject_incompatible(
    current: ReproducibilityIdentity, recorded: ReproducibilityIdentity
) -> None:
    if not compatible(current, recorded):
        raise IncompatibleIdentityError(
            "recorded reproducibility identity is incompatible with the current execution context"
        )


SEED32_MODULUS = 2**32


class SeedDerivationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SeedDerivationRequest:
    base_seed: RandomSeed
    namespace: RngNamespace
    stable_coordinates: StableJsonPayload


def derive_seed32(request: SeedDerivationRequest) -> DerivedSeed:
    coordinates_text = stable_json(request.stable_coordinates)
    payload = f"FedORBIT|{request.base_seed}|{request.namespace.value}|{coordinates_text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % SEED32_MODULUS


@dataclass(frozen=True, slots=True)
class SeedPlan:
    base_seed: RandomSeed
    coordinates_json: SerializedPacket
    streams: tuple[SeedStream, ...]

    def seed_for(self, namespace: RngNamespace) -> DerivedSeed:
        for stream in self.streams:
            if stream.namespace == namespace:
                return stream.seed
        raise SeedDerivationError(f"namespace not in plan: {namespace}")


@dataclass(frozen=True, slots=True)
class SeedStream:
    namespace: RngNamespace
    seed: DerivedSeed


@dataclass(frozen=True, slots=True)
class SeedPlanRequest:
    base_seed: RandomSeed
    coordinates: StableJsonPayload


def seed_plan(request: SeedPlanRequest) -> SeedPlan:
    coordinates_json_value = stable_json(request.coordinates)
    return SeedPlan(
        base_seed=request.base_seed,
        coordinates_json=SerializedPacket(coordinates_json_value),
        streams=tuple(
            SeedStream(
                namespace,
                derive_seed32(
                    SeedDerivationRequest(request.base_seed, namespace, request.coordinates)
                ),
            )
            for namespace in RngNamespace
        ),
    )


@dataclass(frozen=True, slots=True)
class NumpyGeneratorRequest:
    seed: DerivedSeed


@dataclass(frozen=True, slots=True)
class NumpyGeneratorStream:
    generator: np.random.Generator


def numpy_generator(request: NumpyGeneratorRequest) -> NumpyGeneratorStream:
    return NumpyGeneratorStream(np.random.default_rng(request.seed))


@dataclass(frozen=True, slots=True)
class TorchGeneratorRequest:
    seed: DerivedSeed


@dataclass(frozen=True, slots=True)
class TorchGeneratorStream:
    generator: torch.Generator


def torch_generator(request: TorchGeneratorRequest) -> TorchGeneratorStream:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(request.seed)
    return TorchGeneratorStream(generator)


@dataclass(frozen=True, slots=True)
class StatisticalBootstrapRequest:
    statistical_seed: RandomSeed
    contrast_coordinates: StableJsonPayload


def statistical_bootstrap_stream(request: StatisticalBootstrapRequest) -> NumpyGeneratorStream:
    stream_seed = derive_seed32(
        SeedDerivationRequest(
            request.statistical_seed,
            RngNamespace.STATISTICAL_BOOTSTRAP,
            request.contrast_coordinates,
        )
    )
    return numpy_generator(NumpyGeneratorRequest(stream_seed))
