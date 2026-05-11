"""Regression test for the v0.1.3 BMP-PUA collision bug.

If we assign BMP PUA codepoints (U+E000-U+F8FF) to mapped words, real
source text containing literal BMP PUA chars (e.g., TypeScript Unicode
mapping tables, CJK fonts) will roundtrip incorrectly: the input PUA
char gets reverse-substituted into the mapped word.

This test trains a tokenizer with `pua_skip_bmp=True` (the production
default) and asserts that text containing literal BMP PUA chars
roundtrips byte-exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cute_tokenizer import CUTEConfig, CUTETokenizerFast, build_cute
from cute_tokenizer.pua import PUA_BMP_END, PUA_BMP_START

pytestmark = pytest.mark.integration


def test_bmp_pua_in_input_roundtrips_when_skip_bmp_true(tmp_path: Path) -> None:
    """A trained tokenizer with pua_skip_bmp=True must NOT touch literal
    BMP PUA chars during decode."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.py").write_text("def parser():\n    return 1\n" * 100, encoding="utf-8")

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
            pua_skip_bmp=True,  # production default — this is what we're testing
        ),
    )

    tok = CUTETokenizerFast(
        tokenizer_file=out / "tokenizer.json",
        cute_mapping_file=out / "cute_mapping.json",
    )

    # Source text containing literal BMP PUA chars — common in
    # TypeScript Unicode mapping tables and CJK font configs.
    samples = [
        # Mid-BMP PUA
        "const map = ['', '', ''];\n",
        # Low BMP PUA
        f"x = '{chr(PUA_BMP_START)}'\n",
        # High BMP PUA
        f"y = '{chr(PUA_BMP_END)}'\n",
        # Mixed
        "data = ' hello world'\n",
    ]
    for s in samples:
        ids = tok(s, add_special_tokens=False).input_ids
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == s, (
            f"BMP PUA chars in input got reverse-substituted: "
            f"{s!r} -> {decoded!r}. "
            f"pua_skip_bmp=True should preserve them as-is."
        )


def test_pua_skip_bmp_default_is_safe() -> None:
    """The default config must skip BMP PUA — real-world correctness over
    1-byte-per-PUA savings."""
    cfg = CUTEConfig()
    assert cfg.pua_skip_bmp is True, (
        "production default must be pua_skip_bmp=True to avoid real-world "
        "BMP PUA collisions in user content"
    )
