from __future__ import annotations

from typing import Any, NamedTuple, Sequence

class SignificanceResult(NamedTuple):
    statistic: float
    pvalue: float

def spearmanr(
    a: Sequence[float],
    b: Sequence[float] | None = None,
) -> SignificanceResult: ...
def bootstrap(*args: Any, **kwargs: Any) -> Any: ...

class _ChiSquareDistribution:
    def cdf(self, *args: Any, **kwargs: Any) -> Any: ...

chi2: _ChiSquareDistribution
