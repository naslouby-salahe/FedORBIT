from __future__ import annotations

from fedorbit.config.context import active_config, application_config, configured
from fedorbit.config.models import FedorbitConfig


def test_configured_context_restores_outer_configuration(fedorbit_config: FedorbitConfig) -> None:
    altered = fedorbit_config.model_copy(
        update={
            "runtime": fedorbit_config.runtime.model_copy(
                update={"reference_model_gpu": "test accelerator"}
            )
        }
    )
    assert active_config().runtime.reference_model_gpu != "test accelerator"
    with configured(altered):
        assert active_config() is altered
    assert active_config().runtime.reference_model_gpu != "test accelerator"


def test_unbound_context_reuses_the_single_validated_application_configuration() -> None:
    application_config.cache_clear()
    first = active_config()
    second = active_config()
    assert first is second
