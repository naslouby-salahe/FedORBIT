from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tomllib
from dataclasses import dataclass
from typing import cast

import psutil
import torch

from fedorbit.config.loading import repository_root
from fedorbit.config.models import FedorbitConfig

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
    ("pytest", "pytest"),
    ("pytest_cov", "pytest-cov"),
)


class EnvironmentMismatchError(RuntimeError):
    pass


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
    gpu_name: str | None
    gpu_memory_bytes: int | None
    cuda_available: bool
    driver_cuda_version: str | None
    torch_cuda_version: str | None
    cpu_name: str
    ram_bytes: int
    os_release: str


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    python_version: str
    dependencies: tuple[DependencyVersion, ...]
    hardware: HardwareIdentity
    fingerprint_sha256: str

    def mismatches(self) -> tuple[DependencyVersion, ...]:
        return tuple(dependency for dependency in self.dependencies if not dependency.matches)


def observed_python_version() -> str:
    return platform.python_version()


def observed_dependencies(config: FedorbitConfig) -> tuple[DependencyVersion, ...]:
    environment = config.environment
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
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = _gpu_memory_bytes()
    return HardwareIdentity(
        gpu_name=gpu_name,
        gpu_memory_bytes=gpu_memory,
        cuda_available=cuda_available,
        driver_cuda_version=_driver_version(),
        torch_cuda_version=torch.version.cuda,
        cpu_name=platform.processor() or platform.machine(),
        ram_bytes=psutil.virtual_memory().total,
        os_release=platform.platform(),
    )


def _gpu_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return int(result.stdout.strip()) * 1024 * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _fingerprint(snapshot: EnvironmentSnapshot) -> str:
    payload = {
        "python_version": snapshot.python_version,
        "dependencies": {
            dependency.configured_key: dependency.observed for dependency in snapshot.dependencies
        },
        "hardware": {
            "gpu_name": snapshot.hardware.gpu_name,
            "gpu_memory_bytes": snapshot.hardware.gpu_memory_bytes,
            "cuda_available": snapshot.hardware.cuda_available,
            "driver_cuda_version": snapshot.hardware.driver_cuda_version,
            "torch_cuda_version": snapshot.hardware.torch_cuda_version,
            "cpu_name": snapshot.hardware.cpu_name,
            "ram_bytes": snapshot.hardware.ram_bytes,
            "os_release": snapshot.hardware.os_release,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def environment_snapshot(config: FedorbitConfig) -> EnvironmentSnapshot:
    dependencies = observed_dependencies(config)
    hardware = observed_hardware()
    snapshot = EnvironmentSnapshot(
        python_version=observed_python_version(),
        dependencies=dependencies,
        hardware=hardware,
        fingerprint_sha256="",
    )
    return EnvironmentSnapshot(
        python_version=snapshot.python_version,
        dependencies=snapshot.dependencies,
        hardware=snapshot.hardware,
        fingerprint_sha256=_fingerprint(snapshot),
    )


def validate_environment(config: FedorbitConfig, strict: bool = True) -> EnvironmentSnapshot:
    snapshot = environment_snapshot(config)
    deviations: list[str] = []
    configured_python = config.environment.python
    if snapshot.python_version != configured_python:
        deviations.append(
            f"python: configured {configured_python}, observed {snapshot.python_version}"
        )
    for dependency in snapshot.dependencies:
        if not dependency.matches:
            deviations.append(
                f"{dependency.configured_key}: configured {dependency.configured}, "
                f"observed {dependency.observed}"
            )
    if strict and deviations:
        raise EnvironmentMismatchError("; ".join(deviations))
    return snapshot


@dataclass(frozen=True, slots=True)
class LockfileSummary:
    hashed_package_count: int
    package_names: tuple[str, ...]

    @property
    def all_packages_hashed(self) -> bool:
        return self.hashed_package_count == len(self.package_names)


def _package_has_hash(package: dict[str, object]) -> bool:
    source = package.get("source")
    if isinstance(source, dict) and any(
        marker in source for marker in ("editable", "path", "git", "url")
    ):
        return True
    sdist = package.get("sdist")
    if isinstance(sdist, dict) and "hash" in sdist:
        return True
    wheels = package.get("wheels")
    if isinstance(wheels, list):
        for wheel in cast(list[dict[str, object]], wheels):
            if "hash" in wheel:
                return True
    return False


def validate_lockfile(
    config: FedorbitConfig, allow_deviations: frozenset[str] = frozenset()
) -> LockfileSummary:
    lock_path = repository_root() / "uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError("uv.lock is missing; the dependency lock is required")
    with lock_path.open("rb") as handle:
        lock = tomllib.load(handle)
    raw_packages = lock.get("package", [])
    if not isinstance(raw_packages, list) or not raw_packages:
        raise ValueError("uv.lock contains no package entries")
    packages = cast(list[dict[str, object]], raw_packages)
    package_names: list[str] = []
    locked_versions: dict[str, str] = {}
    for package in packages:
        name = package.get("name")
        if not isinstance(name, str):
            raise ValueError("uv.lock package entry lacks a name")
        if not _package_has_hash(package):
            raise ValueError(f"uv.lock package without hashes: {name}")
        package_names.append(name)
        version = package.get("version")
        if isinstance(version, str):
            locked_versions[name] = version
    environment = config.environment
    expected = {
        "torch": environment.pytorch,
        "numpy": environment.numpy,
        "scipy": environment.scipy,
        "scikit-learn": environment.scikit_learn,
        "pandas": environment.pandas,
        "pyarrow": environment.pyarrow,
        "highspy": environment.highspy_highs,
        "pyscipopt": environment.pyscipopt,
        "pydantic": environment.pydantic,
        "typer": environment.typer,
        "psutil": environment.psutil,
        "pytest": environment.pytest,
        "pytest-cov": environment.pytest_cov,
    }
    for distribution, configured in expected.items():
        locked = locked_versions.get(distribution)
        if locked != configured and distribution not in allow_deviations:
            raise EnvironmentMismatchError(
                f"lockfile version for {distribution} is {locked}, configured {configured}"
            )
    return LockfileSummary(
        hashed_package_count=len(package_names),
        package_names=tuple(package_names),
    )


def reference_gpu_matches(config: FedorbitConfig) -> bool:
    hardware = observed_hardware()
    reference = config.runtime.reference_model_gpu
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
