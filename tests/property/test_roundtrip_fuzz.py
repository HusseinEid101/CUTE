"""Heavier Unicode roundtrip fuzz — covers supplementary planes, ZWJ
sequences, RTL scripts, BOM, control chars, and combining marks beyond
what the standard property suite covers.

Default `max_examples` is modest for CI; nightly fuzz can run with
`pytest --hypothesis-seed=0 -m fuzz` and a higher example budget.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cute_tokenizer.decode import reverse_pua_substitute
from cute_tokenizer.pretokenizer import pretokenize_to_string
from cute_tokenizer.pua import PUAMapping, assign_pua_mapping, is_pua_char

pytestmark = pytest.mark.property


@pytest.fixture(scope="module")
def fuzz_mapping() -> PUAMapping:
    """Mapping covering common code keywords + a handful of long identifiers
    so that substitution actually exercises the longest-match path."""
    return assign_pua_mapping(
        [
            "def",
            "return",
            "self",
            "class",
            "import",
            "user_id",
            "calculateTotal",
            "HTTPRequestParser",
            "MAX_BUFFER_SIZE",
            "lambda",
            "function",
            "const",
            "async",
            "await",
            "throw",
            "catch",
        ]
    )


def _full_unicode_text(min_size: int = 0, max_size: int = 80) -> st.SearchStrategy[str]:
    """Arbitrary Unicode including supplementary planes, excluding surrogates,
    PUA codepoints, and non-character codepoints."""
    return st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),  # surrogates
            min_codepoint=0,
            max_codepoint=0x10FFFF,
        ).filter(lambda c: not is_pua_char(c)),
        min_size=min_size,
        max_size=max_size,
    )


class TestFullUnicodeRoundtrip:
    @settings(
        max_examples=500,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        deadline=None,
    )
    @given(text=_full_unicode_text())
    def test_full_plane_roundtrip(self, fuzz_mapping: PUAMapping, text: str) -> None:
        substituted = pretokenize_to_string(text, fuzz_mapping)
        recovered = reverse_pua_substitute(substituted, fuzz_mapping)
        assert recovered == text


class TestUnicodeTorture:
    """Hand-picked strings that historically broke other tokenizers."""

    @pytest.mark.parametrize(
        "text",
        [
            # Supplementary plane samples
            "𐀀𐀁𐀂",  # Linear B
            "𓀀𓀁𓀂",  # Egyptian hieroglyphs
            "🌍🌎🌏",  # plane 1 emoji
            # ZWJ stress
            "👨‍👩‍👧‍👦 family",
            "👨🏻‍💻 dev",
            "🏳️‍🌈 flag",
            # RTL + bidi
            "Hello שלום مرحبا",
            "العربية و English و עברית",
            "abc‫RTL embedded‬abc",  # explicit bidi controls
            # Combining marks
            "café résumé naïve",
            "áb́ć",  # combining acute on each char
            # BOM and zero-widths
            "﻿With BOM",
            "before​after",  # zero-width space
            "before‌after",  # zero-width non-joiner
            # Control chars
            "\x00\x01\x02 hello\x07world",
            "tab\there\nnewline",
            # Mixed code + emoji + RTL
            "def 🚀(): return الـsystem",  # noqa: RUF001 — intentional RTL stress
            # Pathological repetitions
            "́" * 50,
            "_" * 60,
            "ab" * 200,
            # Unicode normalization corners
            "Å",  # ANGSTROM SIGN U+212B
            "Å",  # LATIN CAPITAL A WITH RING U+00C5
            "Å",  # decomposed A + combining ring U+030A
        ],
    )
    def test_torture_roundtrips(self, fuzz_mapping: PUAMapping, text: str) -> None:
        substituted = pretokenize_to_string(text, fuzz_mapping)
        recovered = reverse_pua_substitute(substituted, fuzz_mapping)
        assert recovered == text, (
            f"Torture roundtrip failed: {text!r} -> {substituted!r} -> {recovered!r}"
        )
