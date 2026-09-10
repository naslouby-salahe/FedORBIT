from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from fedorbit.types import TransferMethod


class InformationResourceFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_raw_data: bool
    anonymous_source_nodes: bool
    coarse_groups: bool
    source_response: bool
    target_local_response: bool
    fine_names: bool
    exact_map: bool
    confirmation: bool
    predecision_test_access: bool
    strict_compatibility: bool


def information_resource_catalogue() -> Mapping[TransferMethod, InformationResourceFacts]:
    return OrderedDict(
        (
            (
                TransferMethod.LOCAL_ONLY,
                InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=False,
                    coarse_groups=False,
                    source_response=False,
                    target_local_response=False,
                    fine_names=False,
                    exact_map=False,
                    confirmation=False,
                    predecision_test_access=False,
                    strict_compatibility=True,
                ),
            ),
            (
                TransferMethod.LOCAL_SIR,
                InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=False,
                    coarse_groups=True,
                    source_response=False,
                    target_local_response=True,
                    fine_names=False,
                    exact_map=False,
                    confirmation=True,
                    predecision_test_access=False,
                    strict_compatibility=True,
                ),
            ),
            (
                TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=True,
                    coarse_groups=True,
                    source_response=True,
                    target_local_response=False,
                    fine_names=False,
                    exact_map=False,
                    confirmation=True,
                    predecision_test_access=False,
                    strict_compatibility=True,
                ),
            ),
            (
                TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
                InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=True,
                    coarse_groups=True,
                    source_response=True,
                    target_local_response=False,
                    fine_names=False,
                    exact_map=True,
                    confirmation=True,
                    predecision_test_access=False,
                    strict_compatibility=True,
                ),
            ),
            (
                TransferMethod.GENERIC_EXACT_QAP,
                InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=True,
                    coarse_groups=True,
                    source_response=True,
                    target_local_response=False,
                    fine_names=False,
                    exact_map=True,
                    confirmation=True,
                    predecision_test_access=False,
                    strict_compatibility=True,
                ),
            ),
            (
                TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=True,
                    coarse_groups=True,
                    source_response=True,
                    target_local_response=False,
                    fine_names=False,
                    exact_map=False,
                    confirmation=True,
                    predecision_test_access=False,
                    strict_compatibility=True,
                ),
            ),
            (
                TransferMethod.EXACT_MAP_ORACLE,
                InformationResourceFacts(
                    target_raw_data=False,
                    anonymous_source_nodes=True,
                    coarse_groups=True,
                    source_response=True,
                    target_local_response=False,
                    fine_names=True,
                    exact_map=True,
                    confirmation=True,
                    predecision_test_access=False,
                    strict_compatibility=False,
                ),
            ),
        )
    )
