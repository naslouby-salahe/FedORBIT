from __future__ import annotations

from dataclasses import dataclass


class ClassWeightsError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ClassWeights:
    per_class: tuple[float, ...]
    train_counts: tuple[int, ...]
    total: int

    @classmethod
    def from_train_counts(cls, train_counts: tuple[int, ...]) -> ClassWeights:
        total = sum(train_counts)
        if total <= 0:
            raise ClassWeightsError("train counts must contain at least one example")
        if any(count <= 0 for count in train_counts):
            raise ClassWeightsError("every train class must have at least one example")
        n_classes = len(train_counts)
        raw = tuple(total / (n_classes * count) for count in train_counts)
        example_weighted_mean = (
            sum(count * raw_weight for count, raw_weight in zip(train_counts, raw, strict=True))
            / total
        )
        normalized = tuple(raw_weight / example_weighted_mean for raw_weight in raw)
        return cls(normalized, train_counts, total)

    def per_example(self, label_index: int, intervention_multiplier: float = 1.0) -> float:
        if not 0 <= label_index < len(self.per_class):
            raise ClassWeightsError("label index out of range")
        return self.per_class[label_index] * intervention_multiplier
