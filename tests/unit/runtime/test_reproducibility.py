from __future__ import annotations

import pytest

from fedorbit.config.models import FedorbitConfig
from fedorbit.runtime.environment import environment_snapshot
from fedorbit.runtime.reproducibility import (
    IncompatibleIdentityError,
    build_reproducibility_identity,
    compatible,
    current_code_revision,
    reject_incompatible,
)


def test_code_revision_records_commit_and_tree_digest() -> None:
    revision = current_code_revision()
    assert len(revision.commit) == 40
    assert len(revision.tree_digest) == 64


def test_identity_is_deterministic(
    fedorbit_config: FedorbitConfig,
) -> None:
    environment = environment_snapshot(fedorbit_config)
    first = build_reproducibility_identity(fedorbit_config, environment)
    second = build_reproducibility_identity(fedorbit_config, environment)
    assert first.fingerprint() == second.fingerprint()
    assert compatible(first, second)


def test_identity_rejects_environment_replacement(
    fedorbit_config: FedorbitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.metadata

    recorded_environment = environment_snapshot(fedorbit_config)
    recorded = build_reproducibility_identity(fedorbit_config, recorded_environment)

    original_version = importlib.metadata.version

    def _fake_version(_name: str) -> str:
        return "0.0.0"

    monkeypatch.setattr(importlib.metadata, "version", _fake_version)
    changed_environment = environment_snapshot(fedorbit_config)
    current = build_reproducibility_identity(fedorbit_config, changed_environment)
    monkeypatch.setattr(importlib.metadata, "version", original_version)

    assert not compatible(current, recorded)
    with pytest.raises(IncompatibleIdentityError):
        reject_incompatible(current, recorded)


def test_identity_rejects_confirmatory_seed_change(
    fedorbit_config: FedorbitConfig,
) -> None:
    from tests.typed_access import ConfigDocument

    environment = environment_snapshot(fedorbit_config)
    recorded = build_reproducibility_identity(fedorbit_config, environment)

    import yaml

    from fedorbit.config.validation import validate_cross_field_contract

    with open("configs/fedorbit.yaml", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    document = ConfigDocument(raw)
    seeds = document.list("scientific", "randomness", "confirmatory_seeds")
    seeds[0] = 9999
    altered_model = FedorbitConfig.model_validate(document.as_dict())
    validate_cross_field_contract(altered_model)
    current = build_reproducibility_identity(altered_model, environment)

    assert not compatible(current, recorded)
    with pytest.raises(IncompatibleIdentityError):
        reject_incompatible(current, recorded)


def test_identity_rejects_statistics_change(
    fedorbit_config: FedorbitConfig,
) -> None:
    environment = environment_snapshot(fedorbit_config)
    recorded = build_reproducibility_identity(fedorbit_config, environment)

    altered = fedorbit_config.model_copy(
        deep=True,
        update={
            "scientific": fedorbit_config.scientific.model_copy(
                update={
                    "statistics": fedorbit_config.scientific.statistics.model_copy(
                        update={"ci_bootstrap_repetitions": 9999}
                    )
                }
            )
        },
    )
    current = build_reproducibility_identity(altered, environment)
    assert not compatible(current, recorded)


def test_identity_rejects_claim_criteria_change(
    fedorbit_config: FedorbitConfig,
) -> None:
    environment = environment_snapshot(fedorbit_config)
    recorded = build_reproducibility_identity(fedorbit_config, environment)

    criteria = fedorbit_config.scientific.claim_criteria
    utility = criteria.strict_cross_telemetry_utility.model_copy(
        update={"successful_primary_pairs_required": 4}
    )
    altered = fedorbit_config.model_copy(
        deep=True,
        update={
            "scientific": fedorbit_config.scientific.model_copy(
                update={
                    "claim_criteria": criteria.model_copy(
                        update={"strict_cross_telemetry_utility": utility}
                    )
                }
            )
        },
    )
    current = build_reproducibility_identity(altered, environment)
    assert not compatible(current, recorded)


def test_identity_includes_dataset_pairing() -> None:
    import yaml
    from tests.typed_access import ConfigDocument

    from fedorbit.config.loading import load_fedorbit_config
    from fedorbit.config.validation import validate_cross_field_contract

    base_config = load_fedorbit_config()
    environment = environment_snapshot(base_config)
    recorded = build_reproducibility_identity(base_config, environment)

    with open("configs/fedorbit.yaml", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    document = ConfigDocument(raw)
    pairs = document.list("scientific", "datasets", "primary_directed_pairs")
    pairs[0], pairs[1] = pairs[1], pairs[0]
    altered_model = FedorbitConfig.model_validate(document.as_dict())
    validate_cross_field_contract(altered_model)
    current = build_reproducibility_identity(altered_model, environment)
    assert not compatible(current, recorded)
