from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import yaml

from fedorbit.config.models import FedorbitConfig
from fedorbit.config.validation import validate_cross_field_contract


def repository_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("FedORBIT repository root not found from package location")


def default_config_path() -> Path:
    return repository_root() / "configs" / "fedorbit.yaml"


def load_fedorbit_config(path: Path | None = None) -> FedorbitConfig:
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration file must contain a mapping: {config_path}")
    config = FedorbitConfig.model_validate(raw)
    validate_cross_field_contract(config)
    return config


def stable_json(config: FedorbitConfig) -> str:
    payload = config.model_dump(mode="json")
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def contract_snapshot_path() -> Path:
    return repository_root() / "configs" / "scientific_contract_snapshot.json"


def snapshot_matches_contract(config: FedorbitConfig) -> bool:
    snapshot = contract_snapshot_path()
    if not snapshot.is_file():
        return False
    expected = snapshot.read_text(encoding="utf-8").strip()
    return stable_json(config) == expected


def write_contract_snapshot(config: FedorbitConfig) -> Path:
    snapshot = contract_snapshot_path()
    rendered = unicodedata.normalize("NFC", stable_json(config)) + "\n"
    snapshot.write_text(rendered, encoding="utf-8")
    return snapshot
