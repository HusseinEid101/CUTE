"""CUTE pre-tokenization (Python, inference-time).

This module is used at inference by `CUTETokenizerFast`. It does NOT plug
into the HuggingFace `PreTokenizer.custom(...)` machinery — that route had
intractable offset-tracking issues (PUA chars have a different length than
the words they replace, breaking NormalizedString slicing).

Instead, the wrapper class substitutes mapped tokens in Python BEFORE
feeding text to the underlying byte-level BPE tokenizer; the tokenizer then
matches PUA chars atomically via AddedTokens.

The token-aware regex + identifier sub-split lives here.

Performance: an Aho-Corasick automaton (`build_pua_automaton`) is provided
for whole-text replacement strategies. The default path uses the regex
tokenizer (per-token dict lookup, also O(n)) which interleaves identifier
sub-splitting cleanly.
"""

from __future__ import annotations

import ahocorasick

from ._accel_loader import USE_RUST, accel, prepare_mapping
from .patterns import TOKEN_REGEX, is_identifier, split_identifier
from .pua import PUAMapping, is_pua_char

# ---------------------------------------------------------------------------
# Aho-Corasick automaton for PUA substitution
# ---------------------------------------------------------------------------


def build_pua_automaton(mapping: PUAMapping) -> ahocorasick.Automaton:
    """Build an Aho-Corasick automaton over the mapping's keys.

    The automaton stores `(token, pua_char)` pairs as values. Used in
    `substitute_pua_in_text` for whole-text replacement, NOT for the
    per-token regex path (which does a direct dict lookup instead).
    """
    automaton = ahocorasick.Automaton()
    for word, pua in mapping.word_to_pua.items():
        automaton.add_word(word, (word, pua))
    automaton.make_automaton()
    return automaton


# ---------------------------------------------------------------------------
# Token-level splitter (used both at training and inference)
# ---------------------------------------------------------------------------


def cute_split_text(text: str, mapping: PUAMapping) -> list[str]:
    """Token-aware split. Returns pieces in order; `''.join(...)` reconstructs `text`.

    Each piece is one of:
    - a single PUA character (mapped whole token)
    - a single PUA character (mapped identifier sub-part)
    - a substring of the original text (literal — whitespace / unmapped run)

    Round-trip property: for any input `text`,
        ''.join(cute_split_text(text, m)) == text  (when no PUA chars were
        substituted) — and after PUA substitution, the original can be
        recovered by the inverse mapping (see decode.py).

    Note: this returns "logical" pieces. The underlying byte-level encoding
    is NOT applied here — that's done by the BPE pre-tokenizer in the
    Rust layer (ByteLevel) for training, and by the underlying tokenizer
    for inference.
    """
    if USE_RUST:
        return accel.cute_split_text(text, prepare_mapping(mapping))
    return _python_cute_split_text(text, mapping)


def _python_cute_split_text(text: str, mapping: PUAMapping) -> list[str]:
    """Pure-Python fallback for `cute_split_text`. Used when the Rust
    extension is unavailable or `CUTE_USE_PYTHON_PRETOKENIZER=1`.
    """
    pieces: list[str] = []
    last_end = 0
    word_to_pua = mapping.word_to_pua

    for m in TOKEN_REGEX.finditer(text):
        start, end = m.span()
        if start > last_end:
            # Inter-token whitespace / chars not matched by TOKEN_REGEX.
            pieces.append(text[last_end:start])
        tok = m.group()

        pua = word_to_pua.get(tok)
        if pua is not None:
            pieces.append(pua)
        elif is_identifier(tok):
            # Try identifier-aware sub-splitting.
            parts = split_identifier(tok)
            if not parts:
                pieces.append(tok)
            else:
                # Reconstruct exactly: split_identifier discards underscores,
                # so we re-walk the original token to preserve them.
                pieces.extend(_emit_identifier_pieces(tok, parts, word_to_pua))
        else:
            pieces.append(tok)

        last_end = end

    if last_end < len(text):
        pieces.append(text[last_end:])

    return pieces


def _emit_identifier_pieces(
    ident: str,
    parts: list[str],
    word_to_pua: dict[str, str],
) -> list[str]:
    """Walk `ident` and emit a piece for each sub-part, preserving underscores.

    `parts` is the underscore-stripped split. We need to weave underscores
    back into the output so `''.join(...)` reconstructs `ident` exactly.

    Strategy: scan `ident` left-to-right; consume one part at a time. When
    we encounter an underscore in `ident` at a position where the current
    part doesn't start, emit it as its own piece.
    """
    if not parts:
        return [ident]

    out: list[str] = []
    i = 0
    for part in parts:
        # Skip and emit any underscores at position i.
        while i < len(ident) and ident[i] == "_":
            out.append("_")
            i += 1
        # Now `ident[i:i+len(part)]` should equal `part`.
        if ident[i : i + len(part)] != part:
            # Defensive: if we ever desync, fall back to atomic emission.
            return [ident]
        pua = word_to_pua.get(part)
        out.append(pua if pua is not None else part)
        i += len(part)

    while i < len(ident):
        out.append(ident[i])
        i += 1

    return out


def pretokenize_to_string(text: str, mapping: PUAMapping) -> str:
    """Produce the canonical input string for the underlying BPE tokenizer.

    This is the string that, when passed to the underlying ByteLevel BPE
    tokenizer, will be encoded into the final ID sequence. PUA chars are
    embedded; literal text outside the mapping is passed through unchanged
    (the ByteLevel pre-tokenizer will UTF-8 encode it).
    """
    if USE_RUST:
        return accel.pretokenize_to_string(text, prepare_mapping(mapping))
    return "".join(_python_cute_split_text(text, mapping))


def pretokenize_batch(texts: list[str], mapping: PUAMapping) -> list[str]:
    """Batched pre-tokenization. Processes all texts in one FFI hop with
    Rayon-parallel work-stealing under the hood; eliminates per-item
    Python dispatch cost on `_batch_encode_plus`.

    Falls back to a list comprehension over the single-text path when the
    Rust extension isn't loaded.
    """
    if USE_RUST and hasattr(accel, "pretokenize_batch"):
        return list(accel.pretokenize_batch(texts, prepare_mapping(mapping)))
    return [pretokenize_to_string(t, mapping) for t in texts]


# ---------------------------------------------------------------------------
# Helpers for analysis / debugging
# ---------------------------------------------------------------------------


def count_pua_substitutions(text: str, mapping: PUAMapping) -> int:
    """Count how many characters in the pre-tokenized output are PUA chars.

    Useful for measuring compression on a sample without invoking BPE.
    """
    return sum(1 for ch in pretokenize_to_string(text, mapping) if is_pua_char(ch))


__all__ = [
    "build_pua_automaton",
    "count_pua_substitutions",
    "cute_split_text",
    "pretokenize_to_string",
]
