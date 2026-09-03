from __future__ import annotations

from enum import Enum

kHighsInf: float

class MatrixFormat(Enum):
    kRowwise: MatrixFormat

class HighsModelStatus(Enum):
    kOptimal: HighsModelStatus

class HighsSparseMatrix:
    format_: MatrixFormat
    start_: list[int]
    index_: list[int]
    value_: list[float]

class HighsLp:
    num_col_: int
    num_row_: int
    col_cost_: list[float]
    col_lower_: list[float]
    col_upper_: list[float]
    row_lower_: list[float]
    row_upper_: list[float]
    a_matrix_: HighsSparseMatrix

class HighsSolution:
    col_value: list[float]

class HighsInfo:
    objective_function_value: float

class Highs:
    def setOptionValue(self, name: str, value: str | int | float | bool) -> None: ...
    def passModel(self, lp: HighsLp) -> None: ...
    def run(self) -> None: ...
    def getModelStatus(self) -> HighsModelStatus: ...
    def getSolution(self) -> HighsSolution: ...
    def getInfo(self) -> HighsInfo: ...
