from __future__ import annotations

import ast
import hashlib
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path

from fedorbit.config.loading import repository_root
from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.canonical import canonical_json
from fedorbit.domain.records import SemanticCell

STAGES = (
    "raw",
    "preprocessing",
    "eligibility",
    "pilot_selection",
    "training",
    "scoring",
    "response",
    "target_importance",
    "correspondence",
    "confirmation",
    "multi_source_selection",
    "evaluation",
    "statistics",
    "reporting",
)

STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "raw": (),
    "preprocessing": ("raw",),
    "eligibility": ("preprocessing",),
    "pilot_selection": ("preprocessing", "eligibility"),
    "training": ("preprocessing", "eligibility", "pilot_selection"),
    "scoring": ("training", "preprocessing"),
    "response": ("preprocessing", "scoring"),
    "target_importance": ("training", "scoring"),
    "correspondence": ("response", "target_importance"),
    "confirmation": ("correspondence", "response"),
    "multi_source_selection": ("confirmation", "correspondence"),
    "evaluation": ("confirmation", "scoring"),
    "statistics": ("evaluation",),
    "reporting": ("statistics",),
}

RUNTIME_COMPONENTS: dict[str, tuple[str, ...]] = {
    "raw": ("numpy", "pandas"),
    "preprocessing": ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn"),
    "eligibility": ("numpy",),
    "pilot_selection": ("numpy", "scipy"),
    "training": ("torch", "numpy", "torch-cuda"),
    "scoring": ("torch", "numpy"),
    "response": ("numpy", "scipy", "torch"),
    "target_importance": ("numpy", "torch"),
    "correspondence": ("highspy", "pyscipopt", "numpy", "scipy"),
    "confirmation": ("numpy", "scipy", "torch"),
    "multi_source_selection": ("numpy", "scipy"),
    "evaluation": ("numpy", "scipy", "scikit-learn"),
    "statistics": ("numpy", "scipy"),
    "reporting": (),
}

NON_MATERIAL_COMPONENTS = frozenset({"matplotlib", "plotly", "jinja2", "pandoc"})


class FingerprintError(ValueError):
    pass


def _module_source_digest(module_name: str, visited: set[str]) -> str:
    if module_name in visited:
        return ""
    visited.add(module_name)
    digest = hashlib.sha256()
    module_path = repository_root() / "src" / Path(*module_name.split(".")).with_suffix(".py")
    if not module_path.is_file():
        module_path = module_path.with_name(module_path.stem) / "__init__.py"
    if not module_path.is_file():
        raise FingerprintError(f"module not found: {module_name}")
    digest.update(module_path.read_bytes())
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("fedorbit"):
            digest.update(_module_source_digest(node.module, visited).encode("utf-8"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("fedorbit"):
                    digest.update(_module_source_digest(alias.name, visited).encode("utf-8"))
    return digest.hexdigest()


def implementation_fingerprint(producer_module: str) -> str:
    if not producer_module.startswith("fedorbit"):
        raise FingerprintError(f"producer must be a fedorbit module: {producer_module}")
    return _module_source_digest(producer_module, set())


@dataclass(frozen=True, slots=True)
class RuntimeFingerprint:
    components: tuple[str, ...]
    versions: tuple[tuple[str, str], ...]
    digest: str

    @property
    def sha256(self) -> str:
        return self.digest


def runtime_fingerprint(stage: str) -> RuntimeFingerprint:
    if stage not in STAGE_DEPENDENCIES:
        raise FingerprintError(f"unknown stage: {stage}")
    components = RUNTIME_COMPONENTS.get(stage, ())
    versions: list[tuple[str, str]] = []
    for distribution in components:
        if distribution == "torch-cuda":
            import torch

            versions.append(("torch-cuda", torch.version.cuda or "unknown"))
            continue
        try:
            versions.append((distribution, importlib.metadata.version(distribution)))
        except importlib.metadata.PackageNotFoundError:
            raise FingerprintError(f"runtime component not installed: {distribution}") from None
    payload = canonical_json({"components": components, "versions": versions})
    return RuntimeFingerprint(
        components=components,
        versions=tuple(versions),
        digest=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )


def configuration_subset_digest(config: FedorbitConfig, relevant_sections: frozenset[str]) -> str:
    section_values: dict[str, object] = {}
    for section in relevant_sections:
        if section == "generators":
            section_values[section] = config.generators.model_dump(mode="json")
        elif section == "action":
            section_values[section] = config.scientific.action.model_dump(mode="json")
        elif section == "models":
            section_values[section] = {
                "training": config.scientific.training.model_dump(mode="json"),
                "base_model_pilot": config.scientific.base_model_pilot.model_dump(mode="json"),
            }
        elif section == "response":
            section_values[section] = {
                "source_response_pilot": config.scientific.source_response_pilot.model_dump(
                    mode="json"
                ),
                "source_response_final": config.scientific.source_response_final.model_dump(
                    mode="json"
                ),
                "target_response_diagnostic": (
                    config.scientific.target_response_diagnostic.model_dump(mode="json")
                ),
            }
        elif section == "confirmation":
            section_values[section] = config.scientific.confirmation.model_dump(mode="json")
        elif section == "evaluation":
            section_values[section] = {
                "metrics": config.scientific.metrics.model_dump(mode="json"),
                "statistics": config.scientific.statistics.model_dump(mode="json"),
            }
        elif section == "statistics":
            section_values[section] = config.scientific.statistics.model_dump(mode="json")
        elif section == "experiments":
            section_values[section] = config.experiments.model_dump(mode="json")
        elif section == "simplification_rules":
            section_values[section] = config.scientific.simplification_rules.model_dump(mode="json")
    return hashlib.sha256(canonical_json(section_values).encode("utf-8")).hexdigest()


def stage_dependency_fingerprint(
    stage: str,
    cell: SemanticCell,
    relevance: frozenset[str],
    upstream_artifact_ids: tuple[str, ...],
    config: FedorbitConfig,
    config_sections: frozenset[str],
    producer_module: str,
) -> str:
    implementation = implementation_fingerprint(producer_module)
    runtime = runtime_fingerprint(stage)
    configuration = configuration_subset_digest(config, config_sections)
    payload = canonical_json(
        {
            "stage": stage,
            "semantic_coordinates": cell.identity_json(relevance),
            "upstream_artifact_ids": list(upstream_artifact_ids),
            "configuration_sha256": configuration,
            "implementation_sha256": implementation,
            "runtime_sha256": runtime.sha256,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
