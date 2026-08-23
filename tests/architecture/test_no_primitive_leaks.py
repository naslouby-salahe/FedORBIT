from __future__ import annotations

import ast

import pytest

from tests.architecture.scan import (
    CANONICAL_SERIALIZER_BOUNDARY_MODULES,
    iter_source_files,
    public_functions,
    relative_module,
)

COLLECTION_PRIMITIVES = {"dict", "list", "set", "object"}
SCALAR_PRIMITIVES = {"str", "int", "float", "bool"}


def _is_canonical_serializer_boundary(module: str) -> bool:
    return module in CANONICAL_SERIALIZER_BOUNDARY_MODULES


def _annotation_base(annotation: ast.expr | None) -> str | None:
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        return annotation.value.id
    if isinstance(annotation, ast.Constant):
        return "Constant"
    if isinstance(annotation, ast.BinOp):
        left = _annotation_base(annotation.left)
        right = _annotation_base(annotation.right)
        if left in COLLECTION_PRIMITIVES or right in COLLECTION_PRIMITIVES:
            return "dict"
        return None
    return None


def _render(annotation: ast.expr | None) -> str:
    if annotation is None:
        return "<missing>"
    return ast.unparse(annotation)


def signature_violations(function: ast.FunctionDef) -> list[str]:
    violations: list[str] = []
    for argument in function.args.args:
        base = _annotation_base(argument.annotation)
        if base is not None and (
            base in COLLECTION_PRIMITIVES
            or (base in {"dict", "list"} and _subscripted_by_name(argument.annotation, base))
        ):
            violations.append(
                f"collection primitive parameter '{argument.arg}: {_render(argument.annotation)}'"
            )
    return_base = _annotation_base(function.returns)
    if return_base is not None and (
        return_base in COLLECTION_PRIMITIVES
        or (return_base in {"dict", "list"} and _subscripted_by_name(function.returns, return_base))
    ):
        violations.append(f"collection primitive return '{_render(function.returns)}'")
    tuple_violation = _primitive_tuple_annotation(function.returns)
    if tuple_violation:
        violations.append(f"primitive-only tuple return '{_render(function.returns)}'")
    for argument in function.args.args:
        tuple_violation = _primitive_tuple_annotation(argument.annotation)
        if tuple_violation:
            violations.append(
                f"primitive-only tuple parameter '{argument.arg}: {_render(argument.annotation)}'"
            )
    return violations


def _subscripted_by_name(annotation: ast.expr | None, base: str) -> bool:
    return (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == base
    )


def _primitive_tuple_annotation(annotation: ast.expr | None) -> bool:
    if (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == "tuple"
    ):
        slice_node = annotation.slice
        elements = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]
        if all(
            (isinstance(element, ast.Name) and element.id in SCALAR_PRIMITIVES)
            or (
                isinstance(element, ast.Subscript)
                and isinstance(element.value, ast.Name)
                and element.value.id in {"tuple", "list", "dict"}
            )
            for element in elements
        ):
            return True
    return False


def production_signature_violations() -> list[str]:
    violations: list[str] = []
    for path in iter_source_files():
        module = relative_module(path)
        if module.endswith("__init__") or _is_canonical_serializer_boundary(module):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in public_functions(tree):
            for violation in signature_violations(function):
                violations.append(f"{path}:{function.name}: {violation}")
    return violations


def test_production_public_signatures_do_not_leak_primitives() -> None:
    violations = production_signature_violations()
    assert not violations, "\n".join(violations)


def _violations_in_source(source: str) -> list[str]:
    tree = ast.parse(source)
    violations: list[str] = []
    for function in public_functions(tree):
        violations.extend(signature_violations(function))
    return violations


def test_detector_catches_dict_parameter() -> None:
    source = "def handler(payload: dict[str, int]) -> None: ...\n"
    assert any("payload" in violation for violation in _violations_in_source(source))


def test_detector_catches_dict_return() -> None:
    source = "def build() -> dict[str, str]: ...\n"
    assert any("dict[str, str]" in violation for violation in _violations_in_source(source))


def test_detector_catches_list_parameter_and_object_return() -> None:
    source = "def convert(items: list[float]) -> object: ...\n"
    violations = _violations_in_source(source)
    assert len(violations) == 2


def test_detector_catches_primitive_only_tuples() -> None:
    source = "def pair() -> tuple[int, int]: ...\n"
    assert len(_violations_in_source(source)) == 1


def test_detector_allows_typed_records() -> None:
    source = (
        "@dataclass(frozen=True)\n"
        "class Report:\n"
        "    values: tuple[int, ...]\n"
        "\n"
        "\n"
        "def load_report() -> Report: ...\n"
        "def combine(first: Report, second: Report) -> tuple[Report, ...]: ...\n"
        "def count(reports: tuple[Report, ...]) -> int: ...\n"
    )
    assert _violations_in_source(source) == []


def test_detector_allows_numpy_numeric_payloads() -> None:
    source = (
        "import numpy as np\n"
        "from numpy.typing import NDArray\n"
        "\n"
        "\n"
        "def scale(values: NDArray[np.float64]) -> NDArray[np.float64]: ...\n"
    )
    assert _violations_in_source(source) == []


@pytest.mark.parametrize("module", sorted(CANONICAL_SERIALIZER_BOUNDARY_MODULES))
def test_exempt_modules_exist(module: str) -> None:
    from tests.architecture.scan import SRC_ROOT

    path = SRC_ROOT.joinpath(*module.split(".")).with_suffix(".py")
    assert path.exists(), f"stale exemption entry: {module}"
