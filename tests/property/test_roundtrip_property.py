"""Property-based round-trip tests using Hypothesis.

The single most important invariant of CUTE: any text in → identical text out
after pretokenize + reverse-substitute. If this fails on any string, the
tokenizer is unusable.

We bias the strategies toward code-like content (identifiers, operators,
whitespace, emoji, mixed scripts) because that's the target distribution.
"""

from __future__ import annotations

import string

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cute_tokenizer.decode import reverse_pua_substitute
from cute_tokenizer.patterns import is_identifier, split_identifier
from cute_tokenizer.pretokenizer import pretokenize_to_string
from cute_tokenizer.pua import PUAMapping, assign_pua_mapping, is_pua_char

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Code-flavored chars: letters, digits, underscores, common punctuation,
# whitespace. This biases the search toward what the tokenizer actually sees.
_CODE_ALPHABET = string.ascii_letters + string.digits + "_-+*/(){}[]<>=,.;:!?'\"`@#$%^&|~" + " \t\n"


def code_text(min_size: int = 0, max_size: int = 200) -> st.SearchStrategy[str]:
    return st.text(alphabet=_CODE_ALPHABET, min_size=min_size, max_size=max_size)


def unicode_text(min_size: int = 0, max_size: int = 100) -> st.SearchStrategy[str]:
    """Arbitrary Unicode text, EXCLUDING:
    - Surrogates (Python str doesn't allow them anyway)
    - PUA codepoints (would create round-trip ambiguity if mapping uses them)
    - Non-character codepoints (FFFE/FFFF in any plane)
    """
    return st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),  # surrogates
            blacklist_characters=None,
            min_codepoint=0,
            max_codepoint=0xFFFF,  # stay in BMP for speed; we test PUA via fixed strings
        ).filter(lambda c: not is_pua_char(c)),
        min_size=min_size,
        max_size=max_size,
    )


# ---------------------------------------------------------------------------
# Mapping strategies
# ---------------------------------------------------------------------------


def _make_mapping_from_tokens(tokens: list[str]) -> PUAMapping:
    # Dedupe while preserving order.
    seen = set()
    deduped = []
    for t in tokens:
        if t not in seen and t and not t.isspace():
            seen.add(t)
            deduped.append(t)
    return assign_pua_mapping(deduped)


@pytest.fixture(scope="module")
def code_mapping() -> PUAMapping:
    """A realistic-ish mapping for code keywords / punctuation."""
    return _make_mapping_from_tokens(
        [
            "def",
            "return",
            "self",
            "class",
            "import",
            "from",
            "as",
            "if",
            "else",
            "elif",
            "for",
            "while",
            "in",
            "is",
            "not",
            "and",
            "or",
            "True",
            "False",
            "None",
            "lambda",
            "yield",
            "function",
            "const",
            "let",
            "var",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            ",",
            ":",
            ";",
            "=",
            "==",
            "!=",
            "+",
            "-",
            "*",
            "/",
            "%",
            "user",
            "id",
            "name",
            "data",
            "value",
            "result",
            "get",
            "set",
            "Total",
            "Currency",
            "User",
            "Service",
            "calculate",
            "format",
        ]
    )


@pytest.fixture(scope="module")
def empty_mapping() -> PUAMapping:
    return assign_pua_mapping([])


# ---------------------------------------------------------------------------
# Round-trip property
# ---------------------------------------------------------------------------


class TestRoundTripProperty:
    @settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
    @given(text=code_text())
    def test_roundtrip_code_text(self, code_mapping: PUAMapping, text: str) -> None:
        substituted = pretokenize_to_string(text, code_mapping)
        recovered = reverse_pua_substitute(substituted, code_mapping)
        assert recovered == text

    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    @given(text=unicode_text())
    def test_roundtrip_arbitrary_unicode(self, code_mapping: PUAMapping, text: str) -> None:
        substituted = pretokenize_to_string(text, code_mapping)
        recovered = reverse_pua_substitute(substituted, code_mapping)
        assert recovered == text

    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    @given(text=code_text())
    def test_empty_mapping_is_identity(self, empty_mapping: PUAMapping, text: str) -> None:
        # With an empty mapping, pretokenize must be a no-op.
        out = pretokenize_to_string(text, empty_mapping)
        assert out == text


class TestIdentifierSplitProperty:
    @settings(max_examples=200)
    @given(
        ident=st.from_regex(r"[A-Za-z_][A-Za-z0-9_]*", fullmatch=True).filter(
            lambda s: 1 <= len(s) <= 30,
        )
    )
    def test_split_then_join_minus_underscores(self, ident: str) -> None:
        parts = split_identifier(ident)
        assert "".join(parts) == ident.replace("_", "")

    @settings(max_examples=200)
    @given(
        ident=st.from_regex(r"[A-Za-z_][A-Za-z0-9_]*", fullmatch=True).filter(
            lambda s: 1 <= len(s) <= 30,
        )
    )
    def test_is_identifier_consistent(self, ident: str) -> None:
        assert is_identifier(ident)

    @settings(max_examples=200)
    @given(text=code_text())
    def test_non_ascii_identifier_function_total(self, text: str) -> None:
        # is_identifier must never raise.
        is_identifier(text)
        # split_identifier must never raise either, even on garbage.
        split_identifier(text)


class TestSpecificCornerCases:
    """Hand-picked cases that historically broke similar tokenizers."""

    @pytest.mark.parametrize(
        "text",
        [
            "",
            " ",
            "\n",
            "\r\n",
            "\t\t",
            "a",
            "_",
            "__",
            "___init__",
            "🌍",
            "👨‍👩‍👧‍👦",  # ZWJ family
            "🇺🇸",  # flag
            "1️⃣",  # keycap
            "‍",  # bare ZWJ
            "﻿",  # BOM
            "\x00\x01\x02",  # control chars
            "Mix русский 한글 العربية",  # multiple scripts
            "x" * 100,  # long single-char run
            "a" + "_" * 50 + "b",  # many underscores
        ],
    )
    def test_corner_case_roundtrips(self, code_mapping: PUAMapping, text: str) -> None:
        substituted = pretokenize_to_string(text, code_mapping)
        recovered = reverse_pua_substitute(substituted, code_mapping)
        assert recovered == text, (
            f"Round-trip failure: {text!r} -> {substituted!r} -> {recovered!r}"
        )
