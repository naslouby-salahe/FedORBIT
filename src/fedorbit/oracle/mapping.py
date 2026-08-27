from __future__ import annotations

from dataclasses import dataclass

from fedorbit.domain.enums import OracleTransferConcept
from fedorbit.orbit.correspondence import BlockCorrespondence, PaddedBlockStructure


class OracleMappingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OracleFineConceptMapping:
    source_concepts: tuple[OracleTransferConcept, ...]
    target_concepts: tuple[OracleTransferConcept, ...]

    def correspondence(self, blocks: PaddedBlockStructure) -> BlockCorrespondence:
        if len(self.source_concepts) != len(self.target_concepts):
            raise OracleMappingError(
                "oracle mapping endpoints must contain the same real concept count"
            )
        if len(set(self.source_concepts)) != len(self.source_concepts):
            raise OracleMappingError("oracle source concepts must be unique")
        if len(set(self.target_concepts)) != len(self.target_concepts):
            raise OracleMappingError("oracle target concepts must be unique")
        images = list(range(blocks.total_padded_nodes))
        target_positions = {concept: index for index, concept in enumerate(self.target_concepts)}
        for source_index, concept in enumerate(self.source_concepts):
            target_index = target_positions.get(concept)
            if target_index is None:
                raise OracleMappingError(
                    f"no target concept for oracle source concept {concept.value}"
                )
            if blocks.block_of_node(source_index) != blocks.block_of_node(target_index):
                raise OracleMappingError("oracle mapping crosses configured coarse blocks")
            images[target_index] = source_index
        return BlockCorrespondence(blocks=blocks, images=tuple(images))
