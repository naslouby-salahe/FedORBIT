from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def linear_sum_assignment(
    cost_matrix: NDArray[np.float64],
    maximize: bool = False,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]: ...
