from __future__ import annotations

import hashlib
import platform
from collections import OrderedDict
from dataclasses import dataclass

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
    hardware: HardwareIdentity
    fingerprint_sha256: Sha256Digest


def observed_python_version() -> PythonVersion:
    return PythonVersion(platform.python_version())


def observed_hardware() -> HardwareIdentity:
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu_name = GpuName(torch.cuda.get_device_name(0))
        memory = properties.total_memory
        if isinstance(memory, int):
            gpu_memory = memory
    return HardwareIdentity(
        gpu_name=gpu_name,
        gpu_memory_bytes=gpu_memory,
        cuda_available=cuda_available,
        driver_cuda_version=None,
        torch_cuda_version=(
            CudaVersion(torch.version.cuda) if torch.version.cuda is not None else None
        ),
        cpu_name=CpuName(platform.processor() or platform.machine()),
        ram_bytes=psutil.virtual_memory().total,
        os_release=OperatingSystemRelease(platform.platform()),
    )


def _fingerprint(snapshot: EnvironmentSnapshot) -> Sha256Digest:
    hardware: OrderedDict[
        str,
        str | int | bool | None,
    ] = OrderedDict(
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
        str | OrderedDict[str, str | int | bool | None],
    ] = OrderedDict(
        python_version=snapshot.python_version,
        hardware=hardware,
    )
    stable = stable_json(payload)
    return Sha256Digest(hashlib.sha256(stable.encode("utf-8")).hexdigest())


def environment_snapshot() -> EnvironmentSnapshot:
    hardware = observed_hardware()
    python_version = observed_python_version()
    snapshot = EnvironmentSnapshot(
        python_version=python_version,
        hardware=hardware,
        fingerprint_sha256=Sha256Digest("0" * 64),
    )
    return EnvironmentSnapshot(
        python_version=python_version,
        hardware=hardware,
        fingerprint_sha256=_fingerprint(snapshot),
    )


def reference_gpu_matches() -> bool:
    hardware = observed_hardware()
    reference = active_config().runtime.reference_model_gpu
    if hardware.gpu_name is None or hardware.gpu_memory_bytes is None:
        return False
    reference_parts = reference.split()
    configured_gib: int | None = None
    if len(reference_parts) >= 2 and reference_parts[-1] == "GB" and reference_parts[-2].isdigit():
        configured_gib = int(reference_parts[-2])
        reference_parts = reference_parts[:-2]
    normalized_reference = " ".join(reference_parts)
    name_matches = hardware.gpu_name.strip() == normalized_reference
    if configured_gib is None:
        return name_matches
    memory_gib = hardware.gpu_memory_bytes / (1024**3)
    return name_matches and abs(memory_gib - configured_gib) <= 1.0
