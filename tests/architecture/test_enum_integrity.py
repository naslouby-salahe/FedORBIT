from __future__ import annotations

import ast

from tests.architecture.scan import iter_source_files, parse_module, public_classes

AUTHORITATIVE_ENUM_MODULE = "fedorbit.types"
BOUNDARY_CONSUMER_PACKAGES = {"config", "infrastructure", "reporting", "cli"}


def _enum_names(tree: ast.Module) -> set[str]:
    return {
        klass.name
        for klass in public_classes(tree)
        if any(
            isinstance(base, ast.Name) and base.id in {"Enum", "StrEnum"} for base in klass.bases
        )
    }


def test_authoritative_types_are_consumed_by_boundary_packages() -> None:
    from tests.architecture.scan import SRC_ROOT

    enum_path = SRC_ROOT / "types.py"
    tree = parse_module(enum_path)
    enum_names = _enum_names(tree)
    assert enum_names
    consumers: set[str] = set()
    for path in iter_source_files():
        if path == enum_path:
            continue
        text = path.read_text(encoding="utf-8")
        if "from fedorbit.types import" in text:
            consumers.add(str(path))
    assert consumers, "no production module imports the authoritative type catalogue"


def test_each_authoritative_enum_is_referenced_by_production_code() -> None:
    from tests.architecture.scan import SRC_ROOT

    enum_path = SRC_ROOT / "types.py"
    tree = parse_module(enum_path)
    enum_names = _enum_names(tree)
    production_text = ""
    for path in iter_source_files():
        if path != enum_path:
            production_text += path.read_text(encoding="utf-8")
    for enum_name in enum_names:
        assert enum_name in production_text, (
            f"authoritative enum {enum_name} is never referenced by production code"
        )


def test_configuration_uses_enums_not_free_strings() -> None:
    from fedorbit.config.models import FrozenModel

    annotations = [
        annotation
        for base in FrozenModel.__subclasses__()
        for annotation in base.__annotations__.values()
    ]
    rendered = {str(annotation) for annotation in annotations}
    assert any("DatasetId" in entry for entry in rendered)


def test_no_enum_bypass_via_equivalent_string_constants() -> None:
    for path in iter_source_files():
        if path.name == "types.py":
            continue
        tree = parse_module(path)
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and len(node.value.value) > 10
                and any(
                    marker in node.targets[0].id.lower()
                    for marker in ("method", "split", "state", "client", "seed_role")
                )
            ):
                raise AssertionError(f"enum bypass constant {node.targets[0].id} in {path}")
