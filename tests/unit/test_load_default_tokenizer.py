"""Smoke test for the bundled default tokenizer (requires ``model/`` in checkout)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cute_tokenizer import load_default_tokenizer


def _bundled_model_present() -> bool:
    """True if the development model/ directory or wheel-embedded data dir exists."""
    pkg_root = Path(__file__).resolve().parents[2] / "src" / "cute_tokenizer"
    embedded = pkg_root / "data" / "tokenizer.json"
    repo_dev = Path(__file__).resolve().parents[2] / "model" / "tokenizer.json"
    return embedded.is_file() or repo_dev.is_file()


pytestmark = pytest.mark.skipif(
    not _bundled_model_present(),
    reason="Bundled model/ artifacts not present (CI checkouts skip this smoke test).",
)


def test_load_default_tokenizer_encodes() -> None:
    tok = load_default_tokenizer()
    text = "def hello(): return 42"
    ids = tok(text, add_special_tokens=False).input_ids
    assert len(ids) >= 1
    assert tok.decode(ids, skip_special_tokens=True) == text
