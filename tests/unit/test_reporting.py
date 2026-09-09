from __future__ import annotations

from fedorbit.reporting import information_resource_matrix_table
from fedorbit.types import TransferMethod


def test_information_resource_matrix_table_covers_registered_methods_in_order() -> None:
    table = information_resource_matrix_table()
    assert table.columns == (
        "method",
        "target_raw_data",
        "anonymous_source_nodes",
        "coarse_groups",
        "source_response",
        "target_local_response",
        "fine_names",
        "exact_map",
        "confirmation",
        "predecision_test_access",
        "strict_compatibility",
    )
    methods = tuple(row[0] for row in table.rows)
    assert methods == (
        TransferMethod.LOCAL_ONLY.value,
        TransferMethod.LOCAL_SIR.value,
        TransferMethod.MATCHED_RESOURCE_RECTANGULAR.value,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT.value,
        TransferMethod.GENERIC_EXACT_QAP.value,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value,
        TransferMethod.EXACT_MAP_ORACLE.value,
    )


def test_information_resource_matrix_table_fedorbit_never_identifies_exact_map() -> None:
    table = information_resource_matrix_table()
    rows_by_method = {row[0]: row for row in table.rows}
    columns = table.columns
    exact_map_index = columns.index("exact_map")
    fedorbit_row = rows_by_method[TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value]
    assert fedorbit_row[exact_map_index] is False
    qap_row = rows_by_method[TransferMethod.POINT_CORRESPONDENCE_COMMITMENT.value]
    assert qap_row[exact_map_index] is True


def test_information_resource_matrix_table_local_only_touches_nothing() -> None:
    table = information_resource_matrix_table()
    rows_by_method = {row[0]: row for row in table.rows}
    local_only_row = rows_by_method[TransferMethod.LOCAL_ONLY.value]
    assert all(value is False for value in local_only_row[1:-1])
    assert local_only_row[-1] is True


def test_information_resource_matrix_table_exact_map_oracle_is_strict_incompatible() -> None:
    table = information_resource_matrix_table()
    rows_by_method = {row[0]: row for row in table.rows}
    oracle_row = rows_by_method[TransferMethod.EXACT_MAP_ORACLE.value]
    assert oracle_row[-1] is False
