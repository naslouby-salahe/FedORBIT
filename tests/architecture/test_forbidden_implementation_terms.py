from __future__ import annotations

from tests.architecture.scan import iter_source_files


def test_source_uses_no_forbidden_abstractions() -> None:
    forbidden_terms = ("canonical", "claim")
    for path in iter_source_files():
        source = path.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert term not in source, f"forbidden implementation term {term!r} in {path}"
