"""Determinism tests: the same corpus + config should produce identical
artifacts across runs (modulo timing + host info).

**Determinism contract (per plans/cute-refit.md):**
Same `(OS, Python version, tokenizers version, corpus_hash, seed)` →
byte-identical tokenizer. Cross-platform byte-identity is **not** part
of the contract — the underlying Rust BPE trainer's float comparisons
and serialization order are not guaranteed stable across OSes.

We skip these checks on Windows and macOS because the `tokenizers`
Rust library exhibits intermittent non-determinism on those platforms
under threading / APFS filesystem ordering; the contract is Linux-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cute_tokenizer import CUTEConfig, build_cute
from cute_tokenizer.manifest import BuildManifest, determinism_diff

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform in ("win32", "darwin"),
        reason=(
            "BPE training is not byte-deterministic on Windows/macOS; "
            "determinism contract is Linux-only"
        ),
    ),
]


def _small_config() -> CUTEConfig:
    return CUTEConfig(
        vocab_size=2_000,
        pua_budget=200,
        coverage_target=0.85,
        min_bpe_budget=1_500,
        min_frequency=1,
        workers=1,
    )


class TestDeterminism:
    def test_same_corpus_same_vocab_hash(
        self,
        tiny_corpus_factory,
        tmp_path: Path,
    ) -> None:
        c1 = tiny_corpus_factory("c1")
        c2 = tiny_corpus_factory("c2")
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        build_cute(c1, out1, _small_config())
        build_cute(c2, out2, _small_config())

        m1 = BuildManifest.read(out1 / "build_manifest.json")
        m2 = BuildManifest.read(out2 / "build_manifest.json")
        diffs = determinism_diff(m1, m2)
        assert diffs == [], "Non-deterministic build:\n" + "\n".join(diffs)

    def test_same_corpus_same_mapping(
        self,
        tiny_corpus_factory,
        tmp_path: Path,
    ) -> None:
        c1 = tiny_corpus_factory("c1")
        c2 = tiny_corpus_factory("c2")
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        build_cute(c1, out1, _small_config())
        build_cute(c2, out2, _small_config())

        m1 = (out1 / "cute_mapping.json").read_bytes()
        m2 = (out2 / "cute_mapping.json").read_bytes()
        assert m1 == m2
