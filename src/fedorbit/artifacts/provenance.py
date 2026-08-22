from __future__ import annotations

from dataclasses import dataclass

from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.config.models import FedorbitConfig
from fedorbit.runtime.environment import environment_snapshot
from fedorbit.runtime.reproducibility import current_code_revision


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    artifact_id: str
    created_git_commit: str
    dependency_lock_sha256: str
    operating_system: str
    hardware: str
    driver: str
    environment_sha256: str


def provenance_record(
    config: FedorbitConfig, manifest: ReusableArtifactManifest
) -> ProvenanceRecord:
    environment = environment_snapshot(config)
    revision = current_code_revision()
    return ProvenanceRecord(
        artifact_id=manifest.artifact_id,
        created_git_commit=revision.commit,
        dependency_lock_sha256=manifest.created_environment_sha256,
        operating_system=environment.hardware.os_release,
        hardware=environment.hardware.cpu_name,
        driver=environment.hardware.driver_cuda_version or "unknown",
        environment_sha256=environment.fingerprint_sha256,
    )
