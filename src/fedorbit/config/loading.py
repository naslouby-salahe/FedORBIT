from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, JsonValue

from fedorbit.config.models import FedorbitConfig
from fedorbit.types import Sha256Digest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "fedorbit.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> FedorbitConfig:
    document: JsonValue = yaml.safe_load(path.read_text(encoding="utf-8"))
    return FedorbitConfig.model_validate(document)


def config_digest(config: FedorbitConfig) -> Sha256Digest:
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return Sha256Digest(hashlib.sha256(canonical.encode("utf-8")).hexdigest())


def section_digest(*sections: BaseModel | int) -> Sha256Digest:
    payload = [
        section.model_dump(mode="json") if isinstance(section, BaseModel) else section
        for section in sections
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return Sha256Digest(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
