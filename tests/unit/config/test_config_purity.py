from __future__ import annotations

import re

from fedorbit.config.loading import repository_root

_OBSERVED_FACT_KEY_PATTERN = re.compile(
    r"(^|_)(observed|row_count|file_count|sha256|checksum|timestamp_range|"
    r"duplicate_count|measurement|created_at)(_|$)",
    re.IGNORECASE,
)

_FIXED_RULE_TERMS = (
    "formula",
    "algorithm",
    "procedure",
    "equation",
    "derivation",
    "proof",
)


def _yaml_text() -> str:
    path = repository_root() / "configs" / "fedorbit.yaml"
    return path.read_text(encoding="utf-8")


def test_configuration_contains_data_only() -> None:
    text = _yaml_text()
    assert ": " in text
    assert not text.startswith("[")


def test_no_observed_raw_data_facts_in_yaml() -> None:
    for line in _yaml_text().splitlines():
        key = line.split(":", 1)[0].strip().strip('"')
        assert not _OBSERVED_FACT_KEY_PATTERN.search(key), f"observed-fact key in YAML: {key}"


def test_no_fixed_rule_prose_in_yaml() -> None:
    text = _yaml_text().lower()
    for term in _FIXED_RULE_TERMS:
        assert term not in text, f"fixed-rule term leaked into YAML: {term}"


def test_yaml_contains_no_block_scalar_procedures() -> None:
    text = _yaml_text()
    assert "|" not in text.replace("||", "")
    assert ">" not in text.replace(">=", "").replace("->", "")


def test_no_scientific_contract_snapshot_exists() -> None:
    snapshot = repository_root() / "configs" / "scientific_contract_snapshot.json"
    assert not snapshot.exists()
