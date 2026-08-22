from __future__ import annotations

from tests.architecture.scan import (
    BANNED_NAME_FRAGMENTS,
    VAGUE_MODULE_NAMES,
    iter_source_files,
    parse_module,
    public_classes,
    public_functions,
    relative_module,
)


def test_module_names_are_descriptive() -> None:
    for path in iter_source_files():
        module_name = relative_module(path)
        for fragment in BANNED_NAME_FRAGMENTS:
            assert fragment not in module_name, (
                f"banned name fragment {fragment!r} in module {module_name}"
            )
        leaf = module_name.rsplit(".", 1)[-1]
        assert leaf not in VAGUE_MODULE_NAMES, f"vague module name: {module_name}"
        assert len(leaf) >= 3, f"module name too short: {module_name}"


def test_public_function_names_are_descriptive() -> None:
    for path in iter_source_files():
        tree = parse_module(path)
        for function in public_functions(tree):
            name = function.name
            assert len(name) >= 4, f"function name too short: {path}:{name}"
            assert not name.endswith("2"), f"artificial suffixed name: {path}:{name}"
            for fragment in BANNED_NAME_FRAGMENTS:
                assert fragment not in name, (
                    f"banned fragment {fragment!r} in function name {path}:{name}"
                )


def test_public_class_names_are_descriptive() -> None:
    for path in iter_source_files():
        tree = parse_module(path)
        for klass in public_classes(tree):
            name = klass.name
            assert len(name) >= 4, f"class name too short: {path}:{name}"
            for fragment in BANNED_NAME_FRAGMENTS:
                assert fragment not in name, (
                    f"banned fragment {fragment!r} in class name {path}:{name}"
                )


def test_no_vague_generic_class_names() -> None:
    vague = {"Manager", "Processor", "Helper", "Util", "Base", "Common", "Service"}
    for path in iter_source_files():
        tree = parse_module(path)
        for klass in public_classes(tree):
            assert klass.name not in vague, f"vague class name: {path}:{klass.name}"


def test_no_abbreviated_names_for_domain_concepts() -> None:
    import re

    forbidden_abbreviations = {"ess", "fob", "fbit", "exp", "conf", "corr"}
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        for abbreviation in forbidden_abbreviations:
            assert not re.search(rf"\b{abbreviation}\b", text), (
                f"forbidden abbreviation {abbreviation!r} in {path}"
            )
