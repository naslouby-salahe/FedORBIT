from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass

from fedorbit.config.loading import repository_root
from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.serialization import stable_json
from fedorbit.runtime.environment import EnvironmentSnapshot


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


def _seed_digest(config: FedorbitConfig) -> str:
    randomness = config.scientific.randomness
    return hashlib.sha256(
        stable_json(
            {
                "pilot_seeds": list(randomness.pilot_seeds),
                "confirmatory_seeds": list(randomness.confirmatory_seeds),
                "statistical_seed": randomness.statistical_seed,
            }
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


def statistical_identity_digest(config: FedorbitConfig, environment: EnvironmentSnapshot) -> str:
    scientific = config.scientific
    statistics = scientific.statistics
    payload = stable_json(
        {
            "dataset_ids": list(scientific.datasets.clients),
            "primary_pairs": [
                [pair.source.value, pair.target.value]
                for pair in scientific.datasets.primary_directed_pairs
            ],
            "secondary_pairs": [
                [pair.source.value, pair.target.value]
                for pair in scientific.datasets.secondary_directed_pairs
            ],
            "split": scientific.split.duplicate_safe_chronological_intervals.model_dump(
                mode="json"
            ),
            "preprocessing": scientific.preprocessing.model_dump(mode="json"),
            "seeds": _seed_digest(config),
            "confidence_level": statistics.confidence_level,
            "evaluation_criteria": scientific.evaluation_criteria.model_dump(mode="json"),
            "materiality": scientific.materiality.model_dump(mode="json"),
            "environment": environment.fingerprint_sha256,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_reproducibility_identity(
    config: FedorbitConfig, environment: EnvironmentSnapshot
) -> ReproducibilityIdentity:
    code_revision = current_code_revision()
    return ReproducibilityIdentity(
        config_digest=hashlib.sha256(
            stable_json(config.model_dump(mode="json")).encode("utf-8")
        ).hexdigest(),
        seed_digest=_seed_digest(config),
        environment_fingerprint=environment.fingerprint_sha256,
        code_revision=code_revision,
        statistical_identity_digest=statistical_identity_digest(config, environment),
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
