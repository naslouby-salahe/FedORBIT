from __future__ import annotations

from dataclasses import dataclass

from fedorbit.domain.enums import DatasetId
from fedorbit.orbit.correspondence import BlockCorrespondence


class OracleMappingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OracleCorrespondence:
    source_client: DatasetId
    target_client: DatasetId
    correspondence: BlockCorrespondence

    def __post_init__(self) -> None:
        if self.source_client == self.target_client:
            raise OracleMappingError(
                "oracle correspondence requires distinct source and target clients"
            )
