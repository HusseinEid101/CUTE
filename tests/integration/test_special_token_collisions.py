"""Regression test for the v0.1.3 roundtrip-loss bug.

The original DEFAULT_SPECIAL_TOKENS list included `<s>`, `</s>`, `<unk>`,
`<pad>` — which collide with natural text in code corpora (NLP code
prints these as literal strings, docstrings reference `<unk>` etc.).
Once they were registered as special tokens, the tokenizer would auto-
match them as boundaries, and `decode(skip_special_tokens=True)` would
silently strip them.

This test trains a tiny tokenizer with the current default specials and
asserts that text containing the formerly-problematic substrings round-
trips byte-exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cute_tokenizer import CUTEConfig, CUTETokenizerFast, build_cute

pytestmark = pytest.mark.integration


_PROBLEM_STRINGS = [
    "x = '<unk>'",
    'token = "<s>"',
    "boundary = '</s>'",
    "pad = '<pad>'",
    "['<s>', '</s>', '<unk>'] # NLP boundaries",
    'unk_token="<unk>", bos_token="<s>"',
    # The exact failure pattern observed in the holdout/python set:
    "whether to add <s> and </s> symbols (e.g., for NMT)",
    "context = ['<unk>']*(max_context_size-len(...))",
]


def test_natural_text_with_pseudo_special_tokens_roundtrips(tmp_path: Path) -> None:
    """Train a tokenizer and assert text containing `<s>`, `</s>`, `<unk>`,
    `<pad>` round-trips byte-exactly (i.e., these are NOT special tokens)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.py").write_text("def foo():\n    return 1\n" * 100, encoding="utf-8")
    (corpus / "b.py").write_text("class Bar:\n    pass\n" * 100, encoding="utf-8")

    out = tmp_path / "out"
    build_cute(
        corpus,
        out,
        CUTEConfig(
            vocab_size=2_000,
            pua_budget=200,
            min_bpe_budget=1_500,
            min_frequency=1,
            workers=1,
            seed=42,
        ),
    )

    tok = CUTETokenizerFast(
        tokenizer_file=out / "tokenizer.json",
        cute_mapping_file=out / "cute_mapping.json",
    )

    for s in _PROBLEM_STRINGS:
        ids = tok(s, add_special_tokens=False).input_ids
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == s, (
            f"roundtrip lost characters in {s!r} -> {decoded!r}; "
            "did <s>/</s>/<unk>/<pad> get registered as special tokens?"
        )


def test_default_specials_only_pipe_style() -> None:
    """No default special token should be a substring likely to appear in
    real code. All defaults must use the `<|...|>` convention."""
    from cute_tokenizer.config import DEFAULT_SPECIAL_TOKENS

    for tok in DEFAULT_SPECIAL_TOKENS:
        assert tok.startswith("<|") and tok.endswith("|>"), (
            f"{tok!r} is not pipe-style; it may collide with natural text."
        )
