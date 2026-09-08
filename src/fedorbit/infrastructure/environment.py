from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum

import psutil
import torch

from fedorbit.config.loading import active_config
from fedorbit.types import (
    ByteCount,
    CpuName,
    CudaVersion,
    GpuName,
    OperatingSystemRelease,
    PythonVersion,
    Sha256Digest,
    stable_json,
)

DEPENDENCY_SPECS = (
    ("pytorch", "torch"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("scikit_learn", "scikit-learn"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("highspy_highs", "highspy"),
    ("pyscipopt", "pyscipopt"),
    ("pydantic", "pydantic"),
    ("typer", "typer"),
    ("psutil", "psutil"),
)


@dataclass(frozen=True, slots=True)
class DependencyVersion:
    configured_key: str
    distribution: str
    configured: str
    observed: str

    @property
    def matches(self) -> bool:
        return self.configured == self.observed


@dataclass(frozen=True, slots=True)
class HardwareIdentity:
    gpu_name: GpuName | None
    gpu_memory_bytes: ByteCount | None
    cuda_available: bool
    driver_cuda_version: CudaVersion | None
    torch_cuda_version: CudaVersion | None
    cpu_name: CpuName
    ram_bytes: ByteCount
    os_release: OperatingSystemRelease


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    python_version: PythonVersion
    dependencies: tuple[DependencyVersion, ...]
    hardware: HardwareIdentity
    fingerprint_sha256: Sha256Digest


def observed_python_version() -> PythonVersion:
    return PythonVersion(platform.python_version())


def observed_dependencies() -> tuple[DependencyVersion, ...]:
    environment = active_config().environment
    observed: list[DependencyVersion] = []
    for configured_key, distribution in DEPENDENCY_SPECS:
        configured = getattr(environment, configured_key)
        observed.append(
            DependencyVersion(
                configured_key=configured_key,
                distribution=distribution,
                configured=configured,
                observed=importlib.metadata.version(distribution),
            )
        )
    return tuple(observed)


def observed_hardware() -> HardwareIdentity:
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory = None
    if cuda_available:
        gpu_name = GpuName(torch.cuda.get_device_name(0))
        gpu_memory = _gpu_memory_bytes()
    return HardwareIdentity(
        gpu_name=gpu_name,
        gpu_memory_bytes=gpu_memory,
        cuda_available=cuda_available,
        driver_cuda_version=_driver_version(),
        torch_cuda_version=(
            CudaVersion(torch.version.cuda) if torch.version.cuda is not None else None
        ),
        cpu_name=CpuName(platform.processor() or platform.machine()),
        ram_bytes=psutil.virtual_memory().total,
        os_release=OperatingSystemRelease(platform.platform()),
    )


class NvidiaSmiQuery(StrEnum):
    GPU_MEMORY_TOTAL = "memory.total"
    DRIVER_VERSION = "driver_version"


class NvidiaSmiFormat(StrEnum):
    CSV_NO_HEADER_NO_UNITS = "csv,noheader,nounits"
    CSV_NO_HEADER = "csv,noheader"


def _gpu_memory_bytes() -> ByteCount | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={NvidiaSmiQuery.GPU_MEMORY_TOTAL.value}",
                f"--format={NvidiaSmiFormat.CSV_NO_HEADER_NO_UNITS.value}",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        memory_bytes: ByteCount = int(result.stdout.strip()) * 1024 * 1024
        return memory_bytes
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _driver_version() -> CudaVersion | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={NvidiaSmiQuery.DRIVER_VERSION.value}",
                f"--format={NvidiaSmiFormat.CSV_NO_HEADER.value}",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return CudaVersion(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _fingerprint(snapshot: EnvironmentSnapshot) -> Sha256Digest:
    dependencies = OrderedDict(
        (dependency.configured_key, dependency.observed) for dependency in snapshot.dependencies
    )
    hardware: OrderedDict[str, str | int | bool | None] = OrderedDict(
        gpu_name=snapshot.hardware.gpu_name,
        gpu_memory_bytes=snapshot.hardware.gpu_memory_bytes,
        cuda_available=snapshot.hardware.cuda_available,
        driver_cuda_version=snapshot.hardware.driver_cuda_version,
        torch_cuda_version=snapshot.hardware.torch_cuda_version,
        cpu_name=snapshot.hardware.cpu_name,
        ram_bytes=snapshot.hardware.ram_bytes,
        os_release=snapshot.hardware.os_release,
    )
    payload: OrderedDict[
        str,
        str | OrderedDict[str, str] | OrderedDict[str, str | int | bool | None],
    ] = OrderedDict(
        python_version=snapshot.python_version,
        dependencies=dependencies,
        hardware=hardware,
    )
    stable = stable_json(payload)
    return Sha256Digest(hashlib.sha256(stable.encode("utf-8")).hexdigest())


def environment_snapshot() -> EnvironmentSnapshot:
    dependencies = observed_dependencies()
    hardware = observed_hardware()
    snapshot = EnvironmentSnapshot(
        python_version=observed_python_version(),
        dependencies=dependencies,
        hardware=hardware,
        fingerprint_sha256=Sha256Digest(""),
    )
    return EnvironmentSnapshot(
        python_version=snapshot.python_version,
        dependencies=snapshot.dependencies,
        hardware=snapshot.hardware,
        fingerprint_sha256=_fingerprint(snapshot),
    )


def reference_gpu_matches() -> bool:
    hardware = observed_hardware()
    reference = active_config().runtime.reference_model_gpu
    if hardware.gpu_name is None or hardware.gpu_memory_bytes is None:
        return False
    reference_parts = reference.split()
    if len(reference_parts) >= 2 and reference_parts[-1] == "GB" and reference_parts[-2].isdigit():
        reference_parts = reference_parts[:-2]
    normalized_reference = " ".join(reference_parts)
    name_matches = hardware.gpu_name.strip() == normalized_reference
    memory_gib = hardware.gpu_memory_bytes / (1024**3)
    memory_matches = 15.0 <= memory_gib <= 17.0
    return name_matches and memory_matches
