from __future__ import annotations

import hashlib
import logging
import subprocess
from collections import OrderedDict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import numpy as np
import torch

from fedorbit.config.loading import active_config, repository_root
from fedorbit.infrastructure.environment import EnvironmentSnapshot
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactState,
    DerivedSeed,
    RandomSeed,
    RngNamespace,
    SemanticCoordinates,
    StableJsonPayload,
    stable_json,
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
    matmul_fp32_precision: str
    conv_fp32_precision: str
    stochastic_rounding: bool
    default_dtype: str
    float32_matmul_precision: str


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


def _conv_fp32_precision() -> str:
    conv = getattr(torch.backends.cudnn, "conv", None)
    if conv is not None:
        precision = getattr(conv, "fp32_precision", None)
        if precision is not None:
            return str(precision)
    return "absent"


def _matmul_fp32_precision() -> str:
    precision = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    if precision is not None:
        return str(precision)
    return "absent"


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
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
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
    stage: str | None = None
    experiment: str | None = None
    dataset: str | None = None
    seed: int | None = None
    elapsed_seconds: float | None = None
    reuse_decision: str | None = None


class ExecutionLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def record(self, event: ExecutionLogEvent) -> None:
        self._logger.info(
            "execution_event",
            extra=OrderedDict(
                occurred_at=event.occurred_at.isoformat(),
                cell_coordinates=event.cell_coordinates.value,
                artifact_id=event.artifact_id.value if event.artifact_id is not None else None,
                state=event.state.value,
                stage=event.stage,
                experiment=event.experiment,
                dataset=event.dataset,
                seed=event.seed,
                elapsed_seconds=event.elapsed_seconds,
                reuse_decision=event.reuse_decision,
            ),
        )


def execution_logger() -> ExecutionLogger:
    return ExecutionLogger(logging.getLogger("fedorbit.execution"))


class IncompatibleIdentityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CodeRevision:
    commit: str
    dirty: bool
    tree_digest: str

    def identity(self) -> str:
        suffix = "-dirty" if self.dirty else ""
        return f"{self.commit}{suffix}"


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return "no-git"
    return result.stdout.strip()


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _source_tree_digest() -> str:
    digest = hashlib.sha256()
    source_root = repository_root() / "src"
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(source_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def current_code_revision() -> CodeRevision:
    return CodeRevision(
        commit=_git_head(),
        dirty=_git_dirty(),
        tree_digest=_source_tree_digest(),
    )


def _seed_digest() -> str:
    randomness = active_config().scientific.randomness
    return hashlib.sha256(
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


@dataclass(frozen=True, slots=True)
class ReproducibilityIdentity:
    config_digest: str
    seed_digest: str
    environment_fingerprint: str
    code_revision: CodeRevision
    statistical_identity_digest: str

    def fingerprint(self) -> str:
        return hashlib.sha256(
            "|".join(
                (
                    self.config_digest,
                    self.seed_digest,
                    self.environment_fingerprint,
                    self.code_revision.identity(),
                    self.statistical_identity_digest,
                )
            ).encode("utf-8")
        ).hexdigest()


def statistical_identity_digest(environment: EnvironmentSnapshot) -> str:
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
                environment=environment.fingerprint_sha256,
            ),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_reproducibility_identity(environment: EnvironmentSnapshot) -> ReproducibilityIdentity:
    code_revision = current_code_revision()
    config = active_config()
    return ReproducibilityIdentity(
        config_digest=hashlib.sha256(
            stable_json(config.model_dump(mode="json")).encode("utf-8")
        ).hexdigest(),
        seed_digest=_seed_digest(),
        environment_fingerprint=environment.fingerprint_sha256,
        code_revision=code_revision,
        statistical_identity_digest=statistical_identity_digest(environment),
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
    coordinates_json: str
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
        coordinates_json=coordinates_json_value,
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
