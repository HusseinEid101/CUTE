"""Unit tests for cute_tokenizer.pretokenizer."""

from __future__ import annotations

import pytest

from cute_tokenizer.pretokenizer import (
    count_pua_substitutions,
    cute_split_text,
    pretokenize_to_string,
)
from cute_tokenizer.pua import PUAMapping, assign_pua_mapping


class TestCuteSplitText:
    def test_empty(self, small_mapping: PUAMapping) -> None:
        assert cute_split_text("", small_mapping) == []

    def test_join_reconstructs_when_no_substitution(self) -> None:
        empty = assign_pua_mapping([])
        text = "def foo(x): return x + 1"
        assert "".join(cute_split_text(text, empty)) == text

    def test_known_token_substituted(self, small_mapping: PUAMapping) -> None:
        pieces = cute_split_text("def x", small_mapping)
        # 'def' should appear as its PUA char
        assert small_mapping.word_to_pua["def"] in pieces

    def test_whitespace_preserved(self, small_mapping: PUAMapping) -> None:
        text = "def   foo"
        pieces = cute_split_text(text, small_mapping)
        # Recombining + reverse-substituting must give back the original.
        joined = "".join(small_mapping.pua_to_word.get(p, p) for p in pieces)
        assert joined == text

    def test_identifier_split_with_known_parts(self, small_mapping: PUAMapping) -> None:
        # 'calculate' and 'Total' are in the small_mapping vocab.
        pieces = cute_split_text("calculateTotal", small_mapping)
        assert small_mapping.word_to_pua["calculate"] in pieces
        assert small_mapping.word_to_pua["Total"] in pieces

    def test_identifier_underscores_preserved(self, small_mapping: PUAMapping) -> None:
        text = "user_id"
        pieces = cute_split_text(text, small_mapping)
        joined = "".join(small_mapping.pua_to_word.get(p, p) for p in pieces)
        assert joined == text

    def test_unknown_identifier_passed_through(self) -> None:
        empty = assign_pua_mapping([])
        text = "myCompletelyUnknownIdentifier"
        pieces = cute_split_text(text, empty)
        joined = "".join(pieces)
        assert joined == text

    def test_emoji_passes_through(self, small_mapping: PUAMapping) -> None:
        text = "x = 🚀"
        pieces = cute_split_text(text, small_mapping)
        joined = "".join(small_mapping.pua_to_word.get(p, p) for p in pieces)
        assert joined == text

    def test_mixed_content(self, small_mapping: PUAMapping) -> None:
        text = "def foo(self): return self.user_id"
        pieces = cute_split_text(text, small_mapping)
        joined = "".join(small_mapping.pua_to_word.get(p, p) for p in pieces)
        assert joined == text


class TestPretokenizeToString:
    def test_idempotent_on_unmapped(self) -> None:
        empty = assign_pua_mapping([])
        text = "no substitution happens"
        assert pretokenize_to_string(text, empty) == text

    def test_substitutes_known_words(self, small_mapping: PUAMapping) -> None:
        out = pretokenize_to_string("def foo", small_mapping)
        assert small_mapping.word_to_pua["def"] in out
        assert "def" not in out  # 'def' the literal word should be gone


class TestCountPuaSubstitutions:
    def test_zero_when_no_mapping(self) -> None:
        empty = assign_pua_mapping([])
        assert count_pua_substitutions("hello world", empty) == 0

    def test_counts_substitutions(self, small_mapping: PUAMapping) -> None:
        # 'def', 'return', 'self', '(', ')', ':' are in small_mapping
        text = "def foo(self): return self"
        n = count_pua_substitutions(text, small_mapping)
        # At minimum: def, (, ), :, return, self, self → 7 substitutions
        assert n >= 6


class TestRoundTripWithReverseSubst:
    """Full round-trip via cute_split_text + reverse PUA substitution."""

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "x",
            "hello",
            "def foo(x): return x",
            "class Foo: pass",
            "user_id = 42",
            "calculate_total + getUserInfo + HTTPRequestParser",
            "🌍 hello 🚀",
            "tab\there\nand newline",
            "   leading and trailing   ",
        ],
    )
    def test_roundtrip(self, small_mapping: PUAMapping, text: str) -> None:
        from cute_tokenizer.decode import reverse_pua_substitute

        substituted = pretokenize_to_string(text, small_mapping)
        recovered = reverse_pua_substitute(substituted, small_mapping)
        assert recovered == text
