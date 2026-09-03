from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.scan import (
    SERIALIZATION_BOUNDARY_MODULES,
    iter_source_files,
    parse_module,
    relative_module,
)


def test_no_any_imports_in_production() -> None:
    for path in iter_source_files():
        tree = parse_module(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {"typing", "types"}:
                for alias in node.names:
                    if alias.name == "Any":
                        raise AssertionError(f"Any import in {path}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "typing" or alias.name.startswith("typing."):
                        text = path.read_text(encoding="utf-8")
                        if "Any" in text:
                            raise AssertionError(f"Any usage in {path}")


def test_no_object_annotations_in_production() -> None:
    for path in iter_source_files():
        if _stable_boundary(path):
            continue
        tree = parse_module(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and _is_object(node.annotation):
                raise AssertionError(f"object annotation in {path}:{node.lineno}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.returns is not None and _is_object(node.returns):
                    raise AssertionError(f"object return annotation in {path}:{node.name}")
                for argument in node.args.args:
                    if argument.annotation is not None and _is_object(argument.annotation):
                        raise AssertionError(f"object parameter annotation in {path}:{node.name}")


def _stable_boundary(path: Path) -> bool:
    return relative_module(path) in SERIALIZATION_BOUNDARY_MODULES


def test_no_typing_object_usage() -> None:
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        assert "typing.cast(object" not in text
        assert "object]>" not in text


def test_no_generic_dict_models_for_domain_concepts() -> None:
    domain_modules = ("fedorbit.types", "fedorbit.config")
    for path in iter_source_files():
        module = relative_module(path)
        if not module.startswith(domain_modules):
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in ("dict[str,", "Dict[str,", "dict[", "Dict["):
            assert pattern not in text, f"anonymous dict pattern {pattern!r} in {path}"


def test_production_never_constructs_raw_dictionary_values() -> None:
    for path in iter_source_files():
        tree = parse_module(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Dict, ast.DictComp)):
                raise AssertionError(f"raw dictionary construction in {path}:{node.lineno}")


def _is_object(annotation: ast.expr) -> bool:
    if isinstance(annotation, ast.Name):
        return annotation.id == "object"
    if isinstance(annotation, ast.Subscript):
        return _is_object(annotation.slice)
    if isinstance(annotation, ast.Tuple):
        return any(_is_object(element) for element in annotation.elts)
    if isinstance(annotation, ast.BinOp):
        return _is_object(annotation.left) or _is_object(annotation.right)
    return False


def test_no_bare_object_casts_in_production() -> None:
    for path in iter_source_files():
        tree = parse_module(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "cast"
                and len(node.args) >= 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "object"
            ):
                raise AssertionError(f"cast to object in {path}:{node.lineno}")
