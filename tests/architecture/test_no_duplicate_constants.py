from __future__ import annotations

import ast
from collections import Counter

from tests.architecture.scan import iter_source_files, module_level_constants, parse_module


def test_no_duplicate_module_level_constants() -> None:
    values_by_name: dict[str, Counter[str]] = {}
    for path in iter_source_files():
        tree = parse_module(path)
        for assignment in module_level_constants(tree):
            for target in assignment.targets:
                if not isinstance(target, ast.Name):
                    continue
                value = ast.dump(assignment.value)
                values_by_name.setdefault(target.id, Counter())[value] += 1
    for name, counts in values_by_name.items():
        for _value, occurrences in counts.items():
            assert occurrences == 1, (
                f"duplicate constant {name!r} with identical value across modules "
                f"({occurrences} occurrences)"
            )


def test_no_duplicate_enum_member_values_within_an_enum() -> None:
    import re

    enum_files = [path for path in iter_source_files() if path.name == "enums.py"]
    for path in enum_files:
        text = path.read_text(encoding="utf-8")
        blocks = re.split(r"^class ", text, flags=re.MULTILINE)[1:]
        for block in blocks:
            members = re.findall(r"^    ([A-Z0-9_]+) = \"([^\"]+)\"", block, re.MULTILINE)
            values = [value for _member, value in members]
            duplicates = {value for value in values if values.count(value) > 1}
            enum_name = block.split("(", 1)[0].strip()
            assert not duplicates, (
                f"duplicate values within enum {enum_name} in {path}: {duplicates}"
            )
