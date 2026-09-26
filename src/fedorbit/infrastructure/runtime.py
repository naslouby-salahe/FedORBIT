from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import structlog
import torch

from fedorbit.types import RandomPurpose, RandomSeed, Sha256Digest

SEED_BYTES = 4
CUBLAS_WORKSPACE_SETTING = ":4096:8"


class ExecutionDeviceUnavailableError(RuntimeError):
    pass


def derive_seed(base: RandomSeed, purpose: RandomPurpose, *coordinates: str) -> RandomSeed:
    material = "|".join((str(base), purpose.value, *coordinates))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return RandomSeed(int.from_bytes(digest[:SEED_BYTES], "big"))


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise ExecutionDeviceUnavailableError("the autoencoder experiments require a CUDA device")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_SETTING)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda")


def source_digest(package_directory: Path, relative_paths: tuple[str, ...]) -> Sha256Digest:
    digest = hashlib.sha256()
    files: set[Path] = set()
    for relative in relative_paths:
        target = package_directory / relative
        files.update(target.rglob("*.py") if target.is_dir() else (target,))
    for path in sorted(files):
        digest.update(path.relative_to(package_directory).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return Sha256Digest(digest.hexdigest())


def code_revision(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
    )
