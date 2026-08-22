from __future__ import annotations

from dataclasses import dataclass

from fedorbit.config.models import FedorbitConfig


class EligibilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TransferEligibility:
    source_eligible: bool
    target_eligible: bool
    source_train_support_passes: bool
    source_meta_support_passes: bool
    target_meta_support_passes: bool
    target_confirm_support_passes: bool
    target_test_support_passes: bool


def _support_passes(support: float, threshold: float) -> bool:
    return support >= threshold


def transfer_eligibility(
    config: FedorbitConfig,
    source_train_support: float,
    source_meta_support: float,
    target_meta_support: float,
    target_confirm_support: float,
    target_test_support: float,
) -> TransferEligibility:
    threshold = config.scientific.preprocessing.feature_missing_or_nonfinite_drop_threshold
    source_train_passes = _support_passes(source_train_support, threshold)
    source_meta_passes = _support_passes(source_meta_support, threshold)
    target_meta_passes = _support_passes(target_meta_support, threshold)
    target_confirm_passes = _support_passes(target_confirm_support, threshold)
    target_test_passes = _support_passes(target_test_support, threshold)
    return TransferEligibility(
        source_eligible=source_train_passes and source_meta_passes,
        target_eligible=target_meta_passes and target_confirm_passes and target_test_passes,
        source_train_support_passes=source_train_passes,
        source_meta_support_passes=source_meta_passes,
        target_meta_support_passes=target_meta_passes,
        target_confirm_support_passes=target_confirm_passes,
        target_test_support_passes=target_test_passes,
    )
