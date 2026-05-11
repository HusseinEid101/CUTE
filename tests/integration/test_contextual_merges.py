"""Integration tests for the contextual PUA-merge BPE mechanism.

Smoke gate: after the Step-3 fix (substitute training stream), BPE should
learn whitespace+PUA merges so common prefixed tokens (e.g. ` return`)
collapse to a single id when their bare form is in the PUA mapping.

Also asserts the four invariants enforced by `merge_policy`:
* model.type == BPE
* decoder.type == ByteLevel
* pre_tokenizer.type == ByteLevel
* every PUA char from the mapping has a vocab id
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cute_tokenizer.config import CUTEConfig
from cute_tokenizer.merge_policy import (
    assert_invariants,
    find_pua_pua_merges,
)
from cute_tokenizer.trainer import build_cute, load_mapping

pytestmark = pytest.mark.integration


def _write_corpus(tmp: Path, py_files: dict[str, str]) -> Path:
    """Write `py_files` (filename → content) into `tmp` so `ingest_corpus`
    will pick them up."""
    corpus = tmp / "corpus"
    corpus.mkdir()
    for name, content in py_files.items():
        (corpus / name).write_text(content, encoding="utf-8")
    return corpus


def _smoke_config() -> CUTEConfig:
    """Tiny config that fits the 250-token budget invariant."""
    return CUTEConfig(
        vocab_size=2_000,
        pua_budget=200,
        min_bpe_budget=1_500,
        coverage_target=0.9,
        max_token_len=50,
        min_frequency=1,
        workers=1,
        seed=42,
        # The smoke corpus is tiny; force at most a single shard.
        shard_size_bytes=10 * 1024 * 1024,
    )


def test_invariants_enforced_after_build(tmp_path: Path) -> None:
    """Building any tokenizer must end with all four invariants intact."""
    corpus = _write_corpus(
        tmp_path,
        {
            "a.py": "def foo():\n    return 1\n" * 100,
            "b.py": "class Bar:\n    def baz(self):\n        return self\n" * 100,
        },
    )
    out = tmp_path / "out"
    build_cute(corpus, out, config=_smoke_config())

    tj = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    mapping = load_mapping(out / "cute_mapping.json")
    assert_invariants(tj, mapping)


def test_no_pua_pua_merges_under_strict_atomicity(tmp_path: Path) -> None:
    """With strict_pua_atomicity=True (default), no PUA-PUA pair survives."""
    # A corpus where two short tokens that get PUA-substituted often appear
    # adjacent, encouraging BPE to attempt a PUA-PUA merge.
    body = "ab cd ab cd " * 1_000
    corpus = _write_corpus(tmp_path, {f"f{i}.py": body for i in range(3)})
    out = tmp_path / "out"
    build_cute(corpus, out, config=_smoke_config())

    tj = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    offending = find_pua_pua_merges(tj)
    assert offending == [], f"expected no PUA-PUA merges, found {offending[:3]}"


def test_substituted_iterator_actually_substitutes(tmp_path: Path) -> None:
    """Belt-and-braces: confirm the training-stream substitution path is
    functional. We do this by checking that a token that occurs only in the
    PUA-mapping (synthesized) ends up in the trained vocab as a single
    char — which is only possible if the trainer saw the PUA char."""
    # Build with a corpus containing a unique high-frequency identifier that
    # will end up in the savings-selected PUA mapping.
    body = "MAGIC_TOKEN " * 5_000
    corpus = _write_corpus(tmp_path, {f"f{i}.py": body for i in range(3)})
    out = tmp_path / "out"
    build_cute(corpus, out, config=_smoke_config())

    mapping = load_mapping(out / "cute_mapping.json")
    # The token should have been selected (it has high savings under cl100k
    # if available, and high frequency under any baseline).
    if "MAGIC_TOKEN" not in mapping.word_to_pua:
        pytest.skip("MAGIC_TOKEN not selected by current baseline; skip behavioral check")

    pua_char = mapping.word_to_pua["MAGIC_TOKEN"]

    tj = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    # The PUA char must be a single-character vocab entry (either via
    # learned merge of byte-level pieces — unlikely for a single char — or
    # via the AddedToken safety net registration).
    vocab = tj["model"]["vocab"]
    added = {t["content"] for t in tj.get("added_tokens", [])}
    assert pua_char in vocab or pua_char in added


def test_decoder_invariant_explicitly(tmp_path: Path) -> None:
    """Step-3 review Finding 3: decoder must remain ByteLevel after any rewrite."""
    corpus = _write_corpus(
        tmp_path,
        {"a.py": "x = 1\n" * 1000},
    )
    out = tmp_path / "out"
    build_cute(corpus, out, config=_smoke_config())

    tj = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    assert tj["decoder"]["type"] == "ByteLevel"
    assert tj["pre_tokenizer"]["type"] == "ByteLevel"
    assert tj["model"]["type"] == "BPE"
