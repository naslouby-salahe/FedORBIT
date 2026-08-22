from __future__ import annotations

from tests.architecture.scan import FORBIDDEN_VOCABULARY, iter_source_files


def test_forbidden_vocabulary_absent_from_production() -> None:
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_VOCABULARY:
            assert term.lower() not in text, f"forbidden term {term!r} in {path}"


def test_canonical_method_names_used_verbatim() -> None:
    canonical = {
        "FedORBIT Exact-Sparse Solver",
        "FedORBIT Dense-CCP Fallback",
        "Matched-Resource Rectangular",
        "Point-Correspondence Commitment",
        "Generic Exact QAP",
        "Coupling-Destroyed FedORBIT",
        "FedORBIT Without Confirmation",
    }
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        for name in canonical:
            variant = name.replace(" ", "").replace("-", "_")
            if variant in text and name not in text:
                raise AssertionError(
                    f"non-canonical method name spelling in {path}: {variant!r} "
                    f"(canonical: {name!r})"
                )


def test_dense_ccp_never_described_as_exact() -> None:
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for phrase in ("dense exact", "exact dense", "dense-exact solver"):
            assert phrase not in lowered, f"{phrase!r} claim in {path}"


def test_stale_terminology_absent() -> None:
    stale = ("calibration stage", "threshold cache", "workflow engine", "orchestrator")
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8").lower()
        for term in stale:
            assert term not in text, f"stale terminology {term!r} in {path}"


def test_oracle_terminology_restricted_to_oracle_package() -> None:
    oracle_only_phrases = (
        "oracle mapping",
        "oracle path",
        "oracle access",
        "true benchmark fine concept",
        "exact benchmark source-target mapping",
        "oracle comparison output",
    )
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8").lower()
        for phrase in oracle_only_phrases:
            assert phrase not in text, (
                f"oracle-only phrase {phrase!r} outside oracle package: {path}"
            )


def test_no_artificial_version_labels() -> None:
    import re

    pattern = re.compile(r"\b(final2|copy2|version_2|v2|v3)\b")
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        match = pattern.search(text)
        assert match is None, f"artificial version label {match.group()!r} in {path}"
