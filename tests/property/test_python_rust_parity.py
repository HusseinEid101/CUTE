"""Python ↔ Rust parity tests for the pre-tokenizer hot path.

Runs the same input through both engines (the Rust extension and the
pure-Python fallback) and asserts byte-identical output. This is the
load-bearing safety net for the "Rust accel doesn't change tokenizer
behavior" claim that underpins the determinism contract.

If the Rust extension is not available (extension failed to load),
these tests are skipped — the Python path is the only one in play
and there's nothing to compare.
"""

from __future__ import annotations

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cute_tokenizer._accel_loader import USE_RUST, accel
from cute_tokenizer.patterns import is_identifier as py_is_identifier
from cute_tokenizer.patterns import split_identifier as py_split_identifier
from cute_tokenizer.pretokenizer import _python_cute_split_text  # type: ignore[attr-defined]
from cute_tokenizer.pua import PUAMapping

pytestmark = [
    pytest.mark.property,
    pytest.mark.skipif(not USE_RUST, reason="Rust _accel extension not loaded"),
]


# ---------------------------------------------------------------------------
# Strategy helpers.
# ---------------------------------------------------------------------------


def _ascii_identifier_chars() -> st.SearchStrategy[str]:
    """ASCII chars that the identifier regex/state-machine cares about."""
    return st.sampled_from(string.ascii_letters + string.digits + "_")


@st.composite
def _ascii_identifier(draw: st.DrawFn, max_len: int = 32) -> str:
    head = draw(st.sampled_from(string.ascii_letters + "_"))
    tail = draw(st.text(_ascii_identifier_chars(), max_size=max_len - 1))
    return head + tail


def _code_text() -> st.SearchStrategy[str]:
    """Mixed-content code-ish text: identifiers, operators, whitespace, unicode."""
    snippets = st.sampled_from(
        [
            "def ",
            "class ",
            "return ",
            "self.",
            "import ",
            "from ",
            "if ",
            "else ",
            "elif ",
            "for ",
            " in ",
            " == ",
            " != ",
            " <= ",
            " >= ",
            " -> ",
            " => ",
            "::",
            ":=",
            "...",
            "..",
            "(",
            ")",
            ",",
            ":",
            ";",
            "[",
            "]",
            "{",
            "}",
            " ",
            "  ",
            "\t",
            "\n",
            "α",  # noqa: RUF001 (greek alpha — testing unicode handling)
            "β",  # greek beta
            "héllo",  # accented latin
            "wörld",
            "café",
            "🎉",
            "🔑🔒",
        ]
    )
    return st.lists(
        st.one_of(_ascii_identifier(), snippets),
        min_size=0,
        max_size=80,
    ).map("".join)


def _empty_mapping() -> PUAMapping:
    return PUAMapping(word_to_pua={}, pua_to_word={}, skipped_codepoints=())


def _make_mapping(words: list[str]) -> PUAMapping:
    """Construct a PUAMapping over the unique entries of `words`, assigning
    sequential supplementary-plane PUA codepoints. Mirrors how `assign_pua_mapping`
    builds production mappings without going through the full pipeline.
    """
    seen: dict[str, str] = {}
    pua_to_word: dict[str, str] = {}
    cp = 0xF0001
    for w in words:
        if not w or w in seen:
            continue
        ch = chr(cp)
        seen[w] = ch
        pua_to_word[ch] = w
        cp += 1
    return PUAMapping(word_to_pua=seen, pua_to_word=pua_to_word, skipped_codepoints=())


# ---------------------------------------------------------------------------
# Parity tests.
# ---------------------------------------------------------------------------


@given(s=st.text(_ascii_identifier_chars(), min_size=0, max_size=40))
@settings(max_examples=400, deadline=None)
def test_is_identifier_parity(s: str) -> None:
    py_result = py_is_identifier(s)
    # Python wrapper currently dispatches to Rust when USE_RUST; call accel
    # directly to ensure we're actually comparing the two implementations.
    rust_result = accel.is_identifier(s)
    assert py_result == rust_result, f"is_identifier({s!r}): py={py_result} rs={rust_result}"


@given(ident=_ascii_identifier())
@settings(max_examples=400, deadline=None)
def test_split_identifier_parity(ident: str) -> None:
    py_result = py_split_identifier(ident)
    rust_result = list(accel.split_identifier(ident))
    assert py_result == rust_result, f"split_identifier({ident!r}): py={py_result} rs={rust_result}"


@given(text=_code_text())
@settings(max_examples=300, deadline=None)
def test_pretokenize_to_string_empty_mapping_is_identity(text: str) -> None:
    """With an empty mapping, both engines must reproduce the input exactly."""
    mapping = _empty_mapping()
    rust_result = accel.pretokenize_to_string(
        text, accel.PreparedMapping(mapping.word_to_pua, mapping.pua_to_word)
    )
    py_result = "".join(_python_cute_split_text(text, mapping))
    assert rust_result == py_result == text


@given(text=_code_text())
@settings(max_examples=300, deadline=None)
def test_cute_split_text_join_round_trips(text: str) -> None:
    """`''.join(cute_split_text(text, empty_mapping))` reconstructs `text`."""
    mapping = _empty_mapping()
    prepared = accel.PreparedMapping(mapping.word_to_pua, mapping.pua_to_word)
    rust_pieces = list(accel.cute_split_text(text, prepared))
    assert "".join(rust_pieces) == text


@given(text=_code_text())
@settings(max_examples=300, deadline=None)
def test_cute_split_text_engines_agree_empty(text: str) -> None:
    """Pieces returned by both engines are identical when no PUA substitution."""
    mapping = _empty_mapping()
    prepared = accel.PreparedMapping(mapping.word_to_pua, mapping.pua_to_word)
    rust_pieces = list(accel.cute_split_text(text, prepared))
    py_pieces = _python_cute_split_text(text, mapping)
    assert rust_pieces == py_pieces


@given(
    text=_code_text(),
    mapping_words=st.lists(_ascii_identifier(), min_size=0, max_size=20, unique=True),
)
@settings(max_examples=200, deadline=None)
def test_pretokenize_to_string_engines_agree_with_mapping(
    text: str, mapping_words: list[str]
) -> None:
    """Both engines produce byte-identical pretokenized output with substitutions."""
    mapping = _make_mapping(mapping_words)
    prepared = accel.PreparedMapping(mapping.word_to_pua, mapping.pua_to_word)
    rust = accel.pretokenize_to_string(text, prepared)
    py = "".join(_python_cute_split_text(text, mapping))
    assert rust == py


@given(
    text=_code_text(),
    mapping_words=st.lists(_ascii_identifier(), min_size=0, max_size=20, unique=True),
)
@settings(max_examples=200, deadline=None)
def test_pretok_decode_round_trip(text: str, mapping_words: list[str]) -> None:
    """For text containing no PUA chars, pretokenize → reverse round-trips."""
    # Reject inputs that already contain PUA chars (they'd survive the round-trip
    # only if they're in the mapping, which complicates the property).
    if any(0xE000 <= ord(c) <= 0xF8FF or 0xF0000 <= ord(c) <= 0x10FFFD for c in text):
        return
    mapping = _make_mapping(mapping_words)
    prepared = accel.PreparedMapping(mapping.word_to_pua, mapping.pua_to_word)
    pre = accel.pretokenize_to_string(text, prepared)
    back = accel.reverse_pua_substitute(pre, prepared)
    assert back == text


@given(text=_code_text())
@settings(max_examples=200, deadline=None)
def test_iter_token_strings_parity(text: str) -> None:
    """Token list from Rust matches Python `iter_tokens` (token strings only)."""
    from cute_tokenizer.patterns import iter_tokens as py_iter_tokens

    py_tokens = [tok for tok, _, _ in py_iter_tokens(text)]
    rust_tokens = list(accel.iter_token_strings(text))
    assert py_tokens == rust_tokens


@given(
    text=_code_text(),
    boost=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    max_len=st.integers(min_value=1, max_value=80),
)
@settings(max_examples=150, deadline=None)
def test_count_in_text_parity(text: str, boost: float, max_len: int) -> None:
    """Frequency counts from both engines must match exactly."""
    from collections import Counter, defaultdict

    # Rust path: get deltas from accel directly.
    rust_c, rust_b = accel.count_in_text(text, boost, max_len)

    # Python path: run the original loop via the fallback.
    py_c: Counter[str] = Counter()
    py_b: defaultdict[str, float] = defaultdict(float)
    from cute_tokenizer.patterns import iter_tokens as py_iter_tokens

    for tok, _, _ in py_iter_tokens(text):
        if len(tok) > max_len:
            continue
        py_c[tok] += 1
        if boost > 0 and py_is_identifier(tok):
            for part in py_split_identifier(tok):
                if part != tok:
                    py_b[part] += boost

    assert dict(py_c) == dict(rust_c)
    # Float comparison: counts of identifier sub-parts times boost. Should be
    # exact because we're summing the same float `boost` integer-many times.
    assert dict(py_b) == dict(rust_b)
