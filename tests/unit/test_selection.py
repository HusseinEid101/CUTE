"""Unit tests for cute_tokenizer.selection."""

from __future__ import annotations

from collections import Counter

import pytest

from cute_tokenizer.baseline import BaselineTokenizer, NullBaseline
from cute_tokenizer.selection import (
    compute_candidate_score,
    coverage_of,
    is_good_token,
    select_by_coverage,
    select_by_savings,
)

# ---------------------------------------------------------------------------
# Test baseline (deterministic, no tiktoken dependency)
# ---------------------------------------------------------------------------


class FakeBaseline:
    """Deterministic baseline: 1 token per character, with overrides."""

    name: str = "fake"

    def __init__(self, overrides: dict[str, int] | None = None) -> None:
        self._overrides = overrides or {}

    def count_tokens(self, text: str) -> int:
        if text in self._overrides:
            return self._overrides[text]
        return max(1, len(text)) if text else 0


# ---------------------------------------------------------------------------
# is_good_token
# ---------------------------------------------------------------------------


class TestIsGoodToken:
    @pytest.mark.parametrize("tok", ["def", "user_id", "self.", "==", "🌍"])
    def test_accepts(self, tok: str) -> None:
        assert is_good_token(tok)

    @pytest.mark.parametrize("tok", ["", " ", "\t", "\n", "  \n  "])
    def test_rejects_whitespace(self, tok: str) -> None:
        assert not is_good_token(tok)

    def test_rejects_too_long(self) -> None:
        assert not is_good_token("x" * 100, max_len=50)

    @pytest.mark.parametrize(
        "tok", ["(", ")", ",", ":", ";", ".", "[", "]", "{", "}", "=", "a", "b", "1"]
    )
    def test_rejects_single_byte_ascii(self, tok: str) -> None:
        """Step 2 root-cause-3 fix: byte fallback handles these optimally,
        so PUA assignment would waste a slot."""
        assert not is_good_token(tok), f"single-byte {tok!r} should be rejected"

    @pytest.mark.parametrize(
        "tok",
        [
            "a3f5b6c7d8e9f0a1b2c3d4e5f6a7b8c9",  # 32-char hex hash
            "550e8400-e29b-41d4-a716-446655440000",  # UUID
            "deadbeef" * 4,  # repeated hex
        ],
    )
    def test_rejects_hashes_and_uuids(self, tok: str) -> None:
        assert not is_good_token(tok), f"{tok!r} should be rejected as hash/UUID"

    def test_accepts_normal_long_identifier(self) -> None:
        assert is_good_token("HTTPRequestParserConfiguration")

    def test_accepts_short_alphanum_codes(self) -> None:
        # 8 hex chars is below the 16-char hash threshold.
        assert is_good_token("abc123de")


# ---------------------------------------------------------------------------
# compute_candidate_score
# ---------------------------------------------------------------------------


class TestComputeCandidateScore:
    def test_single_baseline_token_scores_zero(self) -> None:
        b = FakeBaseline(overrides={"e": 1})
        assert compute_candidate_score("e", frequency=1_000_000, baseline=b) == 0

    def test_multi_token_savings_proportional_to_frequency(self) -> None:
        b = FakeBaseline(overrides={"self.": 3})
        # savings = 2, score = 2 * 1000 = 2000
        assert compute_candidate_score("self.", frequency=1000, baseline=b) == 2000

    def test_self_dot_outscores_e(self) -> None:
        """The headline assertion: PUA priority must reflect savings, not freq."""
        b = FakeBaseline(overrides={"e": 1, "self.": 3})
        e_score = compute_candidate_score("e", frequency=1_000_000_000, baseline=b)
        self_score = compute_candidate_score("self.", frequency=1_000, baseline=b)
        assert self_score > e_score


# ---------------------------------------------------------------------------
# select_by_savings
# ---------------------------------------------------------------------------


class TestSelectBySavings:
    def test_empty_freq(self) -> None:
        assert select_by_savings(Counter(), NullBaseline(), vocab_budget=10) == []

    def test_zero_budget(self) -> None:
        b: BaselineTokenizer = FakeBaseline(overrides={"return": 2})
        assert select_by_savings(Counter({"return": 100}), b, vocab_budget=0) == []

    def test_savings_priority_over_frequency(self) -> None:
        b = FakeBaseline(overrides={"e": 1, "self.": 3, "return": 2})
        freq = Counter({"e": 1_000_000, "self.": 100, "return": 200})
        out = select_by_savings(freq, b, vocab_budget=10, min_score=1.0)
        # 'e' has savings 0 → excluded
        assert "e" not in out
        # both 'self.' and 'return' have positive savings; ordering by score
        # 'self.' score = 100 * 2 = 200; 'return' score = 200 * 1 = 200; tie → lex
        assert set(out) == {"self.", "return"}
        # On tie, ascending lex
        assert out == ["return", "self."]

    def test_filters_bad_tokens(self) -> None:
        b = FakeBaseline(overrides={"def": 2, "deadbeefcafebabe": 5})
        freq = Counter({"def": 100, "deadbeefcafebabe": 1000, "  ": 1000, "x" * 100: 1000})
        out = select_by_savings(freq, b, vocab_budget=10)
        assert "def" in out
        # hash-shaped → rejected
        assert "deadbeefcafebabe" not in out
        assert "  " not in out
        assert "x" * 100 not in out

    def test_min_score_threshold(self) -> None:
        b = FakeBaseline(overrides={"ab": 2})
        freq = Counter({"ab": 1})  # score = 1 * 1 = 1
        out_pass = select_by_savings(freq, b, vocab_budget=10, min_score=1.0)
        assert out_pass == ["ab"]
        out_fail = select_by_savings(freq, b, vocab_budget=10, min_score=2.0)
        assert out_fail == []

    def test_vocab_budget_caps(self) -> None:
        b = FakeBaseline()  # default len-based scoring
        freq = Counter({f"tok{i:02d}": 100 for i in range(20)})
        out = select_by_savings(freq, b, vocab_budget=5)
        assert len(out) == 5

    def test_supplementary_pua_default_off(self) -> None:
        """Without allow_supplementary_pua, budget is capped at PUA_BMP_SIZE."""
        from cute_tokenizer.pua import PUA_BMP_SIZE

        b = FakeBaseline()
        # Request a budget bigger than BMP.
        big_budget = PUA_BMP_SIZE + 100
        freq = Counter({f"tok{i:05d}": 100 for i in range(big_budget + 50)})
        out = select_by_savings(freq, b, vocab_budget=big_budget)
        assert len(out) == PUA_BMP_SIZE

        out_unlocked = select_by_savings(
            freq, b, vocab_budget=big_budget, allow_supplementary_pua=True
        )
        assert len(out_unlocked) == big_budget

    def test_determinism(self) -> None:
        b = FakeBaseline()
        freq = Counter({chr(0x61 + i) * 3: (i * 7) % 11 + 1 for i in range(26)})
        out1 = select_by_savings(freq, b, vocab_budget=10)
        out2 = select_by_savings(freq.copy(), b, vocab_budget=10)
        assert out1 == out2


# ---------------------------------------------------------------------------
# select_by_coverage (legacy)
# ---------------------------------------------------------------------------


class TestSelectByCoverage:
    def test_empty_freq(self) -> None:
        assert select_by_coverage(Counter()) == []

    def test_zero_total(self) -> None:
        assert select_by_coverage(Counter({"abc": 0, "def": 0})) == []

    def test_priority_order(self) -> None:
        freq = Counter({"abc": 100, "bcd": 50, "cde": 25})
        out = select_by_coverage(freq, coverage_target=1.0)
        assert out == ["abc", "bcd", "cde"]

    def test_lexicographic_tiebreak(self) -> None:
        freq = Counter({"banana": 10, "apple": 10, "cherry": 10})
        out = select_by_coverage(freq, coverage_target=1.0)
        assert out == ["apple", "banana", "cherry"]

    def test_coverage_threshold_stops_early(self) -> None:
        freq = Counter({"abc": 50, "bcd": 30, "cde": 15, "ghi": 5})
        out = select_by_coverage(freq, coverage_target=0.9)
        assert "abc" in out
        assert "bcd" in out
        assert len(out) <= 4

    def test_filtered_tokens_skipped(self) -> None:
        freq = Counter({"def": 100, "  ": 200, "x" * 100: 50, "return": 90})
        out = select_by_coverage(freq, coverage_target=1.0, max_len=50)
        assert "def" in out
        assert "  " not in out
        assert "x" * 100 not in out
        assert "return" in out

    def test_max_tokens_cap(self) -> None:
        freq = Counter({f"tok{i:02d}": 100 - i for i in range(20)})
        out = select_by_coverage(freq, coverage_target=1.0, max_tokens=5)
        assert len(out) == 5

    def test_invalid_coverage_target(self) -> None:
        with pytest.raises(ValueError):
            select_by_coverage(Counter({"abc": 1}), coverage_target=0.0)
        with pytest.raises(ValueError):
            select_by_coverage(Counter({"abc": 1}), coverage_target=1.5)


class TestCoverageOf:
    def test_full_coverage(self) -> None:
        freq = Counter({"abc": 5, "bcd": 3, "cde": 2})
        assert coverage_of(freq, ["abc", "bcd", "cde"]) == 1.0

    def test_partial(self) -> None:
        freq = Counter({"abc": 5, "bcd": 3, "cde": 2})
        assert coverage_of(freq, ["abc"]) == 0.5

    def test_unknown_token(self) -> None:
        freq = Counter({"abc": 5, "bcd": 3})
        assert coverage_of(freq, ["abc", "missing"]) == 5 / 8

    def test_empty_freq(self) -> None:
        assert coverage_of(Counter(), ["abc"]) == 0.0
