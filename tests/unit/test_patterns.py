"""Unit tests for cute_tokenizer.patterns."""

from __future__ import annotations

import pytest

from cute_tokenizer.patterns import (
    is_identifier,
    iter_tokens,
    split_identifier,
)


class TestIsIdentifier:
    @pytest.mark.parametrize(
        "ident",
        ["x", "foo", "_foo", "foo_bar", "Foo", "FooBar", "_", "x1", "user_id_42"],
    )
    def test_valid(self, ident: str) -> None:
        assert is_identifier(ident)

    @pytest.mark.parametrize(
        "not_ident",
        ["", "1foo", "foo-bar", "foo bar", ".foo", "foo.bar", "foo!", "🌍"],
    )
    def test_invalid(self, not_ident: str) -> None:
        assert not is_identifier(not_ident)


class TestSplitIdentifier:
    @pytest.mark.parametrize(
        ("ident", "expected"),
        [
            ("myVar", ["my", "Var"]),
            ("MyVar", ["My", "Var"]),
            ("HTTPRequest", ["HTTP", "Request"]),
            ("HTTPRequestParser", ["HTTP", "Request", "Parser"]),
            ("MAX_BUFFER_SIZE", ["MAX", "BUFFER", "SIZE"]),
            ("get_user_id", ["get", "user", "id"]),
            ("getUserId42", ["get", "User", "Id", "42"]),
            ("foo", ["foo"]),
            ("FOO", ["FOO"]),
            ("", []),
            ("_private", ["private"]),
            ("__dunder__", ["dunder"]),
            ("a1b2c3", ["a", "1", "b", "2", "c", "3"]),
        ],
    )
    def test_known_cases(self, ident: str, expected: list[str]) -> None:
        assert split_identifier(ident) == expected

    @pytest.mark.parametrize(
        "ident",
        ["foo", "fooBar", "FooBar", "foo_bar", "FOO", "foo_BAR_baz", "userIdToString"],
    )
    def test_reconstruct_minus_underscores(self, ident: str) -> None:
        """Concatenating the parts (no underscores) reconstructs the input
        with all underscores stripped."""
        parts = split_identifier(ident)
        assert "".join(parts) == ident.replace("_", "")


class TestIterTokens:
    def test_simple_python(self) -> None:
        tokens = [t for t, _, _ in iter_tokens("def foo(x): return x + 1")]
        assert "def" in tokens
        assert "foo" in tokens
        assert "(" in tokens
        assert ")" in tokens
        assert ":" in tokens
        assert "return" in tokens
        assert "x" in tokens
        assert "+" in tokens
        assert "1" in tokens

    def test_no_whitespace_in_output(self) -> None:
        for tok, _, _ in iter_tokens("a    b\t\nc"):
            assert not tok.isspace()

    def test_offsets_are_consistent(self) -> None:
        text = "hello world"
        for tok, start, end in iter_tokens(text):
            assert text[start:end] == tok

    def test_multi_char_operators(self) -> None:
        tokens = [t for t, _, _ in iter_tokens("a == b && c <= d")]
        assert "==" in tokens
        assert "&&" in tokens
        assert "<=" in tokens

    def test_emoji_handling(self) -> None:
        tokens = [t for t, _, _ in iter_tokens("hello 🌍 world 🚀")]
        assert "hello" in tokens
        assert "world" in tokens
        # At least one emoji token should be captured
        assert any("🌍" in t or "🚀" in t for t in tokens)

    def test_zwj_emoji_family(self) -> None:
        # Family emoji = man + ZWJ + woman + ZWJ + girl + ZWJ + boy
        family = "👨‍👩‍👧‍👦"
        tokens = [t for t, _, _ in iter_tokens(f"hello {family} world")]
        # The family emoji should appear as one token (or the regex captures
        # the components as one emoji sequence).
        joined = "".join(t for t in tokens if t in (family, "👨", "👩", "👧", "👦"))
        # Either the whole sequence or its components — both are acceptable.
        assert family in joined or all(c in joined for c in ("👨", "👩", "👧", "👦"))

    def test_empty_text(self) -> None:
        assert list(iter_tokens("")) == []

    def test_whitespace_only(self) -> None:
        assert list(iter_tokens("   \n\t  ")) == []
