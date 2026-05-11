"""Unit tests for cute_tokenizer.pua."""

from __future__ import annotations

import pytest

from cute_tokenizer.pua import (
    PUA_BMP_END,
    PUA_BMP_START,
    PUA_P15_END,
    PUA_P15_START,
    PUA_P16_END,
    PUA_P16_START,
    PUA_TOTAL,
    PUAMapping,
    assign_pua_mapping,
    find_pua_codepoints,
    is_pua_char,
)


class TestIsPuaChar:
    @pytest.mark.parametrize(
        "cp",
        [PUA_BMP_START, PUA_BMP_END, PUA_P15_START, PUA_P15_END, PUA_P16_START, PUA_P16_END],
    )
    def test_in_range(self, cp: int) -> None:
        assert is_pua_char(chr(cp))

    @pytest.mark.parametrize("cp", [0, 0x41, 0xDFFF, 0xFB00, 0x100000 - 1])
    def test_out_of_range(self, cp: int) -> None:
        assert not is_pua_char(chr(cp))

    def test_multi_char_string(self) -> None:
        assert not is_pua_char("ab")

    def test_empty_string(self) -> None:
        assert not is_pua_char("")


class TestAssignPuaMapping:
    def test_empty_input(self) -> None:
        m = assign_pua_mapping([])
        assert m.size == 0
        assert m.word_to_pua == {}
        assert m.pua_to_word == {}

    def test_bmp_used_first(self) -> None:
        m = assign_pua_mapping(["a", "b", "c"])
        codepoints = sorted(ord(c) for c in m.pua_to_word)
        assert codepoints[0] == PUA_BMP_START
        assert codepoints[-1] == PUA_BMP_START + 2

    def test_priority_order_preserved(self) -> None:
        """First token in the input list gets the lowest codepoint."""
        m = assign_pua_mapping(["first", "second", "third"])
        assert ord(m.word_to_pua["first"]) == PUA_BMP_START
        assert ord(m.word_to_pua["second"]) == PUA_BMP_START + 1
        assert ord(m.word_to_pua["third"]) == PUA_BMP_START + 2

    def test_bijective(self) -> None:
        m = assign_pua_mapping(["a", "b", "c", "d"])
        for w, c in m.word_to_pua.items():
            assert m.pua_to_word[c] == w

    def test_overflows_to_p15_then_p16(self) -> None:
        # Generate enough tokens to cross BMP into P15 by exactly 1.
        from cute_tokenizer.pua import PUA_BMP_SIZE

        tokens = [f"tok_{i}" for i in range(PUA_BMP_SIZE + 1)]
        m = assign_pua_mapping(tokens)
        last_cp = ord(m.word_to_pua[tokens[-1]])
        assert last_cp == PUA_P15_START

    def test_skip_corpus_codepoints(self) -> None:
        skip = frozenset([PUA_BMP_START, PUA_BMP_START + 5])
        m = assign_pua_mapping(["a", "b", "c"], corpus_pua_codepoints=skip)
        assigned = {ord(c) for c in m.pua_to_word}
        assert PUA_BMP_START not in assigned
        assert PUA_BMP_START + 5 not in assigned
        assert m.skipped_codepoints == (PUA_BMP_START, PUA_BMP_START + 5)

    def test_duplicate_input_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            assign_pua_mapping(["a", "b", "a"])

    def test_overflow_raises(self) -> None:
        # Asking for more slots than total available must raise.
        too_many = [f"t{i}" for i in range(PUA_TOTAL + 1)]
        with pytest.raises(ValueError, match="PUA slots"):
            assign_pua_mapping(too_many)

    def test_pua_chars_property(self) -> None:
        m = assign_pua_mapping(["x", "y", "z"])
        chars = m.pua_chars
        assert sorted(chars, key=ord) == chars  # sorted for determinism
        assert len(chars) == 3


class TestFindPuaCodepoints:
    def test_no_pua(self) -> None:
        assert find_pua_codepoints("hello world 🌍") == set()

    def test_with_pua(self) -> None:
        text = f"hello{chr(PUA_BMP_START)}world{chr(PUA_P15_START)}"
        assert find_pua_codepoints(text) == {PUA_BMP_START, PUA_P15_START}

    def test_empty(self) -> None:
        assert find_pua_codepoints("") == set()


class TestPuaMappingInvariants:
    def test_construction_validates_bijection(self) -> None:
        """Mismatched dicts should raise."""
        with pytest.raises(ValueError):
            PUAMapping(
                word_to_pua={"a": chr(PUA_BMP_START)},
                pua_to_word={chr(PUA_BMP_START): "b"},  # mismatched!
                skipped_codepoints=(),
            )

    def test_construction_validates_pua_chars(self) -> None:
        """Non-PUA char in mapping should raise."""
        with pytest.raises(ValueError, match="non-PUA"):
            PUAMapping(
                word_to_pua={"a": "X"},  # ASCII X, not PUA
                pua_to_word={"X": "a"},
                skipped_codepoints=(),
            )
