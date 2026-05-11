"""Byte-level determinism for the trained tokenizer.

Same-OS, same-python, same-tokenizers-version: building twice on identical
inputs must produce byte-identical `tokenizer.json` and `cute_mapping.json`.

Cross-platform byte-identity is **not** guaranteed (the Rust BPE trainer's
float comparisons + serialization order may differ across OS / Python
versions / tokenizers versions), so this test runs only on the host where
the smoke corpus was last hashed.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from cute_tokenizer import CUTEConfig, build_cute

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="BPE training is not byte-deterministic on Windows; see determinism contract",
    ),
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _smoke_config() -> CUTEConfig:
    return CUTEConfig(
        vocab_size=2_000,
        pua_budget=200,
        coverage_target=0.85,
        min_bpe_budget=1_500,
        min_frequency=1,
        workers=1,
        seed=42,
    )


def test_tokenizer_json_byte_identical(
    tiny_corpus_factory,
    tmp_path: Path,
) -> None:
    """Two builds with identical inputs produce a byte-identical tokenizer.json."""
    c1 = tiny_corpus_factory("c1")
    c2 = tiny_corpus_factory("c2")
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    build_cute(c1, out1, _smoke_config())
    build_cute(c2, out2, _smoke_config())

    h1 = _digest(out1 / "tokenizer.json")
    h2 = _digest(out2 / "tokenizer.json")
    assert h1 == h2, f"tokenizer.json differs across builds: {h1} vs {h2}"


def test_mapping_json_byte_identical(
    tiny_corpus_factory,
    tmp_path: Path,
) -> None:
    c1 = tiny_corpus_factory("c1")
    c2 = tiny_corpus_factory("c2")
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    build_cute(c1, out1, _smoke_config())
    build_cute(c2, out2, _smoke_config())

    h1 = _digest(out1 / "cute_mapping.json")
    h2 = _digest(out2 / "cute_mapping.json")
    assert h1 == h2
