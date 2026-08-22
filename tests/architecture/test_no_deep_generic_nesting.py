from __future__ import annotations

import ast

from tests.architecture.scan import iter_source_files, parse_module


def _generic_depth(annotation: ast.expr) -> int:
    depth = 0
    current = annotation
    while isinstance(current, ast.Subscript):
        depth += 1
        current = current.slice
    return depth


def test_no_deeply_nested_generic_annotations() -> None:
    for path in iter_source_files():
        tree = parse_module(path)
        for node in ast.walk(tree):
            annotations: list[tuple[str, ast.expr, int]] = []
            if isinstance(node, ast.FunctionDef):
                for argument in node.args.args:
                    if argument.annotation is not None:
                        annotations.append((argument.arg, argument.annotation, node.lineno))
                if node.returns is not None:
                    annotations.append(("return", node.returns, node.lineno))
            elif isinstance(node, ast.AnnAssign):
                name = node.target.id if isinstance(node.target, ast.Name) else "attribute"
                annotations.append((name, node.annotation, node.lineno))
            for name, annotation, line in annotations:
                if _generic_depth(annotation) >= 3:
                    raise AssertionError(
                        f"deeply nested generic annotation in {path}:{line} "
                        f"({name}: {ast.unparse(annotation)}); "
                        "extract a typed aggregate instead"
                    )
