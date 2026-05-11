"""Unit tests for cute_tokenizer.merge_policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cute_tokenizer.merge_policy import (
    assert_invariants,
    audit_and_filter_tokenizer_file,
    filter_pua_pua_merges,
    find_pua_pua_merges,
)
from cute_tokenizer.pua import PUA_BMP_START, PUAMapping


def _make_mapping(words: list[str]) -> PUAMapping:
    word_to_pua = {w: chr(PUA_BMP_START + i) for i, w in enumerate(words)}
    pua_to_word = {c: w for w, c in word_to_pua.items()}
    return PUAMapping(word_to_pua=word_to_pua, pua_to_word=pua_to_word, skipped_codepoints=())


def _minimal_tokenizer_json(
    *,
    extra_merges: list[str] | None = None,
    added: list[str] | None = None,
    extra_vocab: dict[str, int] | None = None,
) -> dict:
    vocab = {chr(b): b for b in range(256)}
    if extra_vocab:
        vocab.update(extra_vocab)
    return {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": list(extra_merges or []),
            "unk_token": None,
        },
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [{"content": c} for c in (added or [])],
    }


class TestFindPuaPuaMerges:
    def test_empty_merges(self) -> None:
        tj = _minimal_tokenizer_json()
        assert find_pua_pua_merges(tj) == []

    def test_finds_pua_pua(self) -> None:
        a, b = chr(PUA_BMP_START), chr(PUA_BMP_START + 1)
        tj = _minimal_tokenizer_json(extra_merges=[f"{a} {b}"])
        out = find_pua_pua_merges(tj)
        assert out == [(0, a, b)]

    def test_ignores_pua_with_normal_char(self) -> None:
        a = chr(PUA_BMP_START)
        tj = _minimal_tokenizer_json(extra_merges=[f"{a} x", f"x {a}"])
        assert find_pua_pua_merges(tj) == []

    def test_ignores_normal_normal(self) -> None:
        tj = _minimal_tokenizer_json(extra_merges=["a b", "c d"])
        assert find_pua_pua_merges(tj) == []


class TestFilterPuaPuaMerges:
    def test_no_op_when_clean(self) -> None:
        tj = _minimal_tokenizer_json(extra_merges=["a b"])
        new, n = filter_pua_pua_merges(tj)
        assert n == 0
        assert new["model"]["merges"] == ["a b"]

    def test_drops_offenders(self) -> None:
        a, b, c = chr(PUA_BMP_START), chr(PUA_BMP_START + 1), chr(PUA_BMP_START + 2)
        tj = _minimal_tokenizer_json(extra_merges=[f"{a} {b}", "x y", f"{a} {c}"])
        new, n = filter_pua_pua_merges(tj)
        assert n == 2
        assert new["model"]["merges"] == ["x y"]


class TestAssertInvariants:
    def test_passes_minimal(self) -> None:
        m = _make_mapping(["x"])
        # PUA char must be reachable; add it as added_tokens for the assertion.
        tj = _minimal_tokenizer_json(added=[chr(PUA_BMP_START)])
        assert_invariants(tj, m)

    def test_rejects_wrong_decoder(self) -> None:
        m = _make_mapping(["x"])
        tj = _minimal_tokenizer_json(added=[chr(PUA_BMP_START)])
        tj["decoder"]["type"] = "WordPiece"
        with pytest.raises(AssertionError, match="decoder"):
            assert_invariants(tj, m)

    def test_rejects_missing_pua(self) -> None:
        m = _make_mapping(["x", "y"])
        # Only register one PUA char; the other should fail.
        tj = _minimal_tokenizer_json(added=[chr(PUA_BMP_START)])
        with pytest.raises(AssertionError, match="PUA"):
            assert_invariants(tj, m)

    def test_pua_in_vocab_satisfies_invariant(self) -> None:
        m = _make_mapping(["x"])
        ch = chr(PUA_BMP_START)
        tj = _minimal_tokenizer_json(extra_vocab={ch: 999})
        assert_invariants(tj, m)


class TestAuditAndFilterFile:
    def test_round_trip(self, tmp_path: Path) -> None:
        m = _make_mapping(["x", "y"])
        a, b = chr(PUA_BMP_START), chr(PUA_BMP_START + 1)
        tj = _minimal_tokenizer_json(extra_merges=[f"{a} {b}", "p q"], added=[a, b])
        path = tmp_path / "tokenizer.json"
        path.write_text(json.dumps(tj), encoding="utf-8")

        stats = audit_and_filter_tokenizer_file(path, m, strict=True)
        assert stats == {"pua_pua_merges_found": 1, "pua_pua_merges_removed": 1}

        # File should be rewritten with the offending merge gone.
        rewritten = json.loads(path.read_text(encoding="utf-8"))
        assert rewritten["model"]["merges"] == ["p q"]
