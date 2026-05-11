"""Unit tests for cute_tokenizer.baseline."""

from __future__ import annotations

import pytest

from cute_tokenizer.baseline import (
    Cl100kBaseline,
    NullBaseline,
    get_default_baseline,
)

tiktoken = pytest.importorskip("tiktoken")


class TestCl100kBaseline:
    def test_empty_string_is_zero(self) -> None:
        b = Cl100kBaseline()
        assert b.count_tokens("") == 0

    def test_single_char_is_one(self) -> None:
        b = Cl100kBaseline()
        assert b.count_tokens("e") == 1

    def test_multi_char_token_at_least_one(self) -> None:
        b = Cl100kBaseline()
        assert b.count_tokens("self.") >= 1

    def test_self_dot_costs_more_than_e(self) -> None:
        """Sanity: 'self.' costs at least as many tokens as 'e' (the savings
        we care about for PUA assignment)."""
        b = Cl100kBaseline()
        assert b.count_tokens("self.") >= b.count_tokens("e")

    def test_returns_keyword_baseline(self) -> None:
        b = Cl100kBaseline()
        assert b.count_tokens(" return") <= 2

    def test_cache_returns_same_value(self) -> None:
        b = Cl100kBaseline(cache_size=4)
        v1 = b.count_tokens("user_id")
        v2 = b.count_tokens("user_id")
        assert v1 == v2

    def test_name(self) -> None:
        assert Cl100kBaseline().name == "cl100k_base"


class TestNullBaseline:
    def test_empty_is_zero(self) -> None:
        assert NullBaseline().count_tokens("") == 0

    def test_nonempty_is_one(self) -> None:
        # Reset the warning flag for this test
        NullBaseline._warned = False
        with pytest.warns(UserWarning, match="NullBaseline"):
            assert NullBaseline().count_tokens("hello") == 1

    def test_warns_only_once(self) -> None:
        NullBaseline._warned = False
        b = NullBaseline()
        with pytest.warns(UserWarning):
            b.count_tokens("a")
        # Second call should not re-warn (warning state is class-level).
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            b.count_tokens("b")
        assert not caught


class TestGetDefaultBaseline:
    def test_returns_cl100k_when_available(self) -> None:
        b = get_default_baseline()
        # tiktoken is available in test env (importorskip above)
        assert b.name == "cl100k_base"
