from __future__ import annotations

import ast
import re

from tests.architecture.scan import (
    SRC_ROOT,
    iter_source_files,
    relative_module,
)

FORBIDDEN_GENERIC_ALIASES = {
    "NonNegativeInt",
    "PositiveInt",
    "NonNegativeFloat",
    "PositiveFloat",
    "UnitInterval",
    "OpenUnitInterval",
    "FiniteFloat",
    "SignedInt",
}

_WORD_BOUNDARY = re.compile(
    r"\b(" + "|".join(sorted(FORBIDDEN_GENERIC_ALIASES, key=len, reverse=True)) + r")\b"
)


def _forbidden_references(source_text: str) -> list[str]:
    return sorted({match.group(1) for match in _WORD_BOUNDARY.finditer(source_text)})


def test_forbidden_generic_aliases_never_used_outside_types_py() -> None:
    for path in iter_source_files():
        if relative_module(path) == "types":
            continue
        references = _forbidden_references(path.read_text(encoding="utf-8"))
        assert not references, (
            f"forbidden generic alias(es) used outside types.py in {path}: {references}"
        )


def test_architecture_scanner_inspects_every_production_source_file() -> None:
    expected = {path for path in SRC_ROOT.rglob("*.py") if "__pycache__" not in path.parts}
    scanned = set(iter_source_files())
    assert scanned == expected, f"architecture scanner ignores: {expected - scanned}"


def test_forbidden_alias_scanner_catches_import() -> None:
    source = "from fedorbit.types import NonNegativeInt\n"
    assert _forbidden_references(source) == ["NonNegativeInt"]


def test_forbidden_alias_scanner_catches_annotation() -> None:
    source = "def run(count: PositiveInt) -> None: ...\n"
    assert _forbidden_references(source) == ["PositiveInt"]


def test_forbidden_alias_scanner_catches_nested_generic() -> None:
    source = "values: list[UnitInterval]\n"
    assert _forbidden_references(source) == ["UnitInterval"]


def test_forbidden_alias_scanner_catches_qualified_name() -> None:
    source = "seed = fedorbit.types.FiniteFloat(3)\n"
    assert _forbidden_references(source) == ["FiniteFloat"]


def test_forbidden_alias_scanner_catches_alias_reassignment() -> None:
    source = "RoundedValue = SignedInt\n"
    assert _forbidden_references(source) == ["SignedInt"]


def test_forbidden_alias_scanner_catches_redeclaration() -> None:
    source = "NonNegativeInt = Annotated[int, Field(ge=0)]\n"
    tree = ast.parse(source)
    assert isinstance(tree, ast.Module)
    assert _forbidden_references(source) == ["NonNegativeInt"]
