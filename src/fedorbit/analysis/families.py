from __future__ import annotations

from dataclasses import dataclass

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import MultiplicityFamily, TransferMethod


class ContrastRegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RegisteredContrast:
    family: MultiplicityFamily
    name: str
    directed_pair: str
    statistic: str


@dataclass(frozen=True, slots=True)
class FamilyInputState:
    contrast: RegisteredContrast
    available: bool
    unavailable_reason: str | None = None
    raw_p_value: float | None = None


PRIMARY_PAIR_NAMES = ("Edge→Windows", "Windows→Edge", "Edge→Linux", "Linux→Edge")


def _pair_contrast(
    family: MultiplicityFamily,
    name_template: str,
    pair: str,
    statistic: str,
) -> RegisteredContrast:
    return RegisteredContrast(
        family=family,
        name=name_template,
        directed_pair=pair,
        statistic=statistic,
    )


def registered_family_inputs() -> dict[MultiplicityFamily, tuple[RegisteredContrast, ...]]:
    families: dict[MultiplicityFamily, list[RegisteredContrast]] = {
        family: [] for family in MultiplicityFamily
    }
    solver = TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value
    for pair in PRIMARY_PAIR_NAMES:
        families[MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY].append(
            _pair_contrast(
                MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
                f"{solver} vs Local-Only — TEST relative macro-CE gain",
                pair,
                "sign_flip_superiority",
            )
        )
        superiority_name = f"{solver} vs Local-SIR — TEST relative macro-CE gain superiority"
        families[MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR].append(
            _pair_contrast(
                MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR,
                superiority_name,
                pair,
                "sign_flip_superiority",
            )
        )
        equivalence_name = f"{solver} vs Local-SIR — TEST relative macro-CE gain TOST equivalence"
        families[MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR].append(
            _pair_contrast(
                MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR,
                equivalence_name,
                pair,
                "tost_equivalence",
            )
        )
        coupling_name = (
            "Exact correspondence orbit vs Matched-Resource Rectangular — robust coupling value gap"
        )
        families[MultiplicityFamily.COUPLING_MECHANISM].append(
            _pair_contrast(
                MultiplicityFamily.COUPLING_MECHANISM,
                coupling_name,
                pair,
                "sign_flip_against_zero",
            )
        )
        for suffix in ("difference", "TOST equivalence"):
            statistic = "tost_equivalence" if "TOST" in suffix else "sign_flip_superiority"
            point_name = (
                f"{solver} vs Point-Correspondence Commitment — TEST relative macro-CE {suffix}"
            )
            families[MultiplicityFamily.POINT_CORRESPONDENCE_SAFETY].append(
                _pair_contrast(
                    MultiplicityFamily.POINT_CORRESPONDENCE_SAFETY,
                    point_name,
                    pair,
                    statistic,
                )
            )
            ablation_name = (
                f"{solver} vs Coupling-Destroyed FedORBIT — TEST relative macro-CE {suffix}"
            )
            families[MultiplicityFamily.MECHANISM_ABLATIONS].append(
                _pair_contrast(
                    MultiplicityFamily.MECHANISM_ABLATIONS,
                    ablation_name,
                    pair,
                    statistic,
                )
            )
        for sparsity_name in (
            "exact sparse s=1 vs exact sparse s=2",
            "exact sparse s=3 vs exact sparse s=2",
            "dense CCP vs exact sparse s=2",
        ):
            families[MultiplicityFamily.SPARSITY_SENSITIVITY].append(
                _pair_contrast(
                    MultiplicityFamily.SPARSITY_SENSITIVITY,
                    sparsity_name,
                    pair,
                    "sign_flip_difference_common_reference",
                )
            )
        confirmation_name = (
            "FedORBIT Without Confirmation vs FedORBIT Exact-Sparse Solver "
            "with confirmation — harmful-transfer rate difference"
        )
        families[MultiplicityFamily.CONFIRMATION_SAFETY].append(
            _pair_contrast(
                MultiplicityFamily.CONFIRMATION_SAFETY,
                confirmation_name,
                pair,
                "seed_level_rate_difference_sign_flip",
            )
        )
    registry: dict[MultiplicityFamily, tuple[RegisteredContrast, ...]] = {
        family: tuple(contrasts) for family, contrasts in families.items()
    }
    return registry


def build_family_states(
    config: FedorbitConfig,
    available_p_values: dict[str, float],
) -> dict[MultiplicityFamily, tuple[FamilyInputState, ...]]:
    del config
    states: dict[MultiplicityFamily, list[FamilyInputState]] = {}
    for family, contrasts in registered_family_inputs().items():
        entries: list[FamilyInputState] = []
        for contrast in contrasts:
            if contrast.name in available_p_values:
                entries.append(
                    FamilyInputState(
                        contrast=contrast,
                        available=True,
                        raw_p_value=available_p_values[contrast.name],
                    )
                )
            else:
                entries.append(
                    FamilyInputState(
                        contrast=contrast,
                        available=False,
                        unavailable_reason="insufficient valid paired seeds",
                    )
                )
        states[family] = tuple(entries)
    if not any(state.available for entry_states in states.values() for state in entry_states):
        raise ContrastRegistryError("no registered family input has a computable p-value")
    return states
