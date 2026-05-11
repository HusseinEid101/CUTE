"""Unit tests for cute_tokenizer.decode."""

from __future__ import annotations

from cute_tokenizer.decode import reverse_pua_substitute
from cute_tokenizer.pua import PUA_BMP_START, PUAMapping, assign_pua_mapping


class TestReversePuaSubstitute:
    def test_empty(self, small_mapping: PUAMapping) -> None:
        assert reverse_pua_substitute("", small_mapping) == ""

    def test_no_pua_chars_passthrough(self, small_mapping: PUAMapping) -> None:
        text = "no special chars here 🌍"
        assert reverse_pua_substitute(text, small_mapping) == text

    def test_single_substitution(self, small_mapping: PUAMapping) -> None:
        pua_def = small_mapping.word_to_pua["def"]
        assert reverse_pua_substitute(pua_def, small_mapping) == "def"

    def test_multiple_substitutions(self, small_mapping: PUAMapping) -> None:
        pua_def = small_mapping.word_to_pua["def"]
        pua_self = small_mapping.word_to_pua["self"]
        text = f"{pua_def} foo({pua_self}):"
        # Note: '(', ')', ':' are also in the mapping; we use the literal form here.
        recovered = reverse_pua_substitute(text, small_mapping)
        assert recovered == "def foo(self):"

    def test_unknown_pua_char_preserved(self) -> None:
        empty = assign_pua_mapping([])
        # A PUA char NOT in the mapping should pass through unchanged.
        text = f"hello{chr(PUA_BMP_START)}world"
        assert reverse_pua_substitute(text, empty) == text

    def test_empty_mapping_passthrough(self) -> None:
        empty = assign_pua_mapping([])
        text = "anything goes"
        assert reverse_pua_substitute(text, empty) == text

    def test_round_trip_against_pretokenize(self, small_mapping: PUAMapping) -> None:
        from cute_tokenizer.pretokenizer import pretokenize_to_string

        original = "def calculate(self, user_id): return self"
        substituted = pretokenize_to_string(original, small_mapping)
        # Confirm substitution actually happened
        assert substituted != original
        recovered = reverse_pua_substitute(substituted, small_mapping)
        assert recovered == original
