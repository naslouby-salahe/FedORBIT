from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.typed_access import ConfigDocument

from fedorbit.config.loading import (
    canonical_json,
    contract_snapshot_path,
    default_config_path,
    load_fedorbit_config,
    repository_root,
    snapshot_matches_contract,
    write_contract_snapshot,
)
from fedorbit.config.models import FedorbitConfig


def test_repository_root_contains_pyproject() -> None:
    root = repository_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "configs" / "fedorbit.yaml").is_file()


def test_default_config_path_points_at_authoritative_contract() -> None:
    path = default_config_path()
    assert path == repository_root() / "configs" / "fedorbit.yaml"


def test_load_produces_immutable_configuration(fedorbit_config: FedorbitConfig) -> None:
    with pytest.raises(ValidationError):
        fedorbit_config.scientific = fedorbit_config.scientific


def test_load_accepts_explicit_path(config_dict: dict[str, object], tmp_path: Path) -> None:
    path = tmp_path / "copy.yaml"
    path.write_text(json.dumps(config_dict), encoding="utf-8")
    loaded = load_fedorbit_config(path)
    assert loaded == FedorbitConfig.model_validate(config_dict)


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_fedorbit_config(tmp_path / "absent.yaml")


def test_load_rejects_non_mapping_document(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_fedorbit_config(path)


def test_load_rejects_unknown_top_level_field(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value("invented_section", value={"value": 1})
    with pytest.raises(ValidationError):
        FedorbitConfig.model_validate(mutable_config.as_dict())


def test_load_rejects_unknown_nested_field(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value("scientific", "action", "invented_parameter", value=1)
    with pytest.raises(ValidationError):
        FedorbitConfig.model_validate(mutable_config.as_dict())


def test_load_rejects_wrong_type(mutable_config: ConfigDocument) -> None:
    mutable_config.set_value("scientific", "action", "principal_sparse_support", value="two")
    with pytest.raises(ValidationError):
        FedorbitConfig.model_validate(mutable_config.as_dict())


def test_canonical_json_is_deterministic(fedorbit_config: FedorbitConfig) -> None:
    first = canonical_json(fedorbit_config)
    second = canonical_json(fedorbit_config)
    assert first == second
    assert "\n" not in first
    parsed = json.loads(first)
    assert parsed["scientific"]["action"]["principal_sparse_support"] == 2


def test_canonical_json_sorts_keys(fedorbit_config: FedorbitConfig) -> None:
    rendered = canonical_json(fedorbit_config)
    top = json.loads(rendered)
    assert list(top.keys()) == sorted(top.keys())


def test_committed_contract_snapshot_is_current(fedorbit_config: FedorbitConfig) -> None:
    snapshot = contract_snapshot_path()
    assert snapshot.is_file(), "scientific_contract_snapshot.json must be committed"
    assert snapshot_matches_contract(fedorbit_config)


def test_write_snapshot_round_trip(
    fedorbit_config: FedorbitConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "snapshot.json"
    monkeypatch.setattr("fedorbit.config.loading.contract_snapshot_path", lambda: target)
    written = write_contract_snapshot(fedorbit_config)
    assert written == target
    assert target.is_file()
    assert snapshot_matches_contract(fedorbit_config)


def test_snapshot_changed_when_contract_changes(
    fedorbit_config: FedorbitConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "snapshot.json"
    monkeypatch.setattr("fedorbit.config.loading.contract_snapshot_path", lambda: target)
    write_contract_snapshot(fedorbit_config)
    altered = ConfigDocument(fedorbit_config.model_dump(mode="json"))
    altered.set_value("scientific", "action", "principal_sparse_support", value=3)
    changed = FedorbitConfig.model_validate(altered.as_dict())
    assert not snapshot_matches_contract(changed)


def test_all_registered_clients_present(fedorbit_config: FedorbitConfig) -> None:
    clients = fedorbit_config.scientific.datasets.clients
    assert set(clients.keys()) == {
        "edge_iiotset_network",
        "ton_iot_windows10_host",
        "ton_iot_linux_process_host",
        "ton_iot_network",
    }
