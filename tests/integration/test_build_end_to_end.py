"""End-to-end integration tests: build a real CUTE tokenizer.

These are slower (each runs the full pipeline) but small enough to be part
of the standard CI run on the tiny corpus fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cute_tokenizer import CUTEConfig, CUTETokenizerFast, build_cute
from cute_tokenizer.manifest import BuildManifest

# All E2E tests in this file need the full pipeline, so mark accordingly.
pytestmark = pytest.mark.integration


def _small_config() -> CUTEConfig:
    """Tiny config for fast E2E runs on the fixture corpus."""
    return CUTEConfig(
        vocab_size=2_000,
        pua_budget=200,
        coverage_target=0.85,
        min_bpe_budget=1_500,
        min_frequency=1,
        workers=1,  # avoid multiprocessing in tests
    )


class TestBuildE2E:
    def test_build_produces_artifacts(self, tiny_corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        manifest_path = build_cute(tiny_corpus, out, _small_config())

        assert (out / "tokenizer.json").exists()
        assert (out / "cute_mapping.json").exists()
        assert (out / "tokenizer_config.json").exists()
        assert (out / "build_manifest.json").exists()
        assert manifest_path == out / "build_manifest.json"

    def test_mapping_is_nonempty(self, tiny_corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        build_cute(tiny_corpus, out, _small_config())
        mapping = json.loads((out / "cute_mapping.json").read_text(encoding="utf-8"))
        assert mapping["size"] > 0
        # Under savings-based selection, every selected token must encode to
        # MORE than 1 baseline (cl100k) token — otherwise the savings is 0.
        # We assert the mapping isn't dominated by single-byte ASCII junk
        # by sanity-checking the typical lengths.
        words = list(mapping["word_to_codepoint"].keys())
        # At least one multi-character identifier should be present (e.g. a
        # camelCase or snake_case identifier from the fixture).
        assert any(len(w) >= 5 for w in words), (
            f"Selection produced no long identifiers (top 5: {words[:5]})"
        )

    def test_manifest_has_coverage(self, tiny_corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        build_cute(tiny_corpus, out, _small_config())
        m = BuildManifest.read(out / "build_manifest.json")
        assert 0.0 < m.coverage_achieved <= 1.0
        assert m.pua_mapping_size > 0
        assert m.cute_version
        assert "tokenizers" in m.library_versions

    def test_load_via_wrapper_class(self, tiny_corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        build_cute(tiny_corpus, out, _small_config())

        tok = CUTETokenizerFast(
            tokenizer_file=out / "tokenizer.json",
            cute_mapping_file=out / "cute_mapping.json",
        )
        assert tok.cute_mapping.size > 0

    @pytest.mark.parametrize(
        "text",
        [
            "def hello(): return 42",
            "class Foo: pass",
            "user_id = 1",
            "x = 'hello world'",
        ],
    )
    def test_round_trip_on_simple_strings(
        self, tiny_corpus: Path, tmp_path: Path, text: str
    ) -> None:
        out = tmp_path / "out"
        build_cute(tiny_corpus, out, _small_config())
        tok = CUTETokenizerFast(
            tokenizer_file=out / "tokenizer.json",
            cute_mapping_file=out / "cute_mapping.json",
        )
        ids = tok(text, add_special_tokens=False).input_ids
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == text, f"Round-trip failed: {text!r} → {decoded!r}"

    def test_compression_observed(self, tiny_corpus: Path, tmp_path: Path) -> None:
        """CUTE should produce SHORTER sequences than a vanilla byte-level
        BPE on the same corpus. We compare token count to character count
        as a sanity proxy (a real benchmark vs tiktoken lives in benchmarks/)."""
        out = tmp_path / "out"
        build_cute(tiny_corpus, out, _small_config())
        tok = CUTETokenizerFast(
            tokenizer_file=out / "tokenizer.json",
            cute_mapping_file=out / "cute_mapping.json",
        )
        text = "def calculate(self, user_id): return self.history"
        ids = tok(text, add_special_tokens=False).input_ids
        # On code text with frequent tokens mapped to PUA, we should see
        # significantly fewer tokens than characters.
        assert len(ids) < len(text), f"No compression: {len(ids)} ids for {len(text)} chars"

    def test_save_pretrained_roundtrip(self, tiny_corpus: Path, tmp_path: Path) -> None:
        """save_pretrained → from_pretrained should round-trip."""
        out = tmp_path / "out"
        build_cute(tiny_corpus, out, _small_config())
        tok = CUTETokenizerFast(
            tokenizer_file=out / "tokenizer.json",
            cute_mapping_file=out / "cute_mapping.json",
        )
        save_dir = tmp_path / "saved"
        tok.save_pretrained(str(save_dir))

        # Reload
        tok2 = CUTETokenizerFast(
            tokenizer_file=save_dir / "tokenizer.json",
            cute_mapping_file=save_dir / "cute_mapping.json",
        )
        text = "def x(): return 1"
        ids1 = tok(text, add_special_tokens=False).input_ids
        ids2 = tok2(text, add_special_tokens=False).input_ids
        assert ids1 == ids2

    def test_batch_encoding(self, tiny_corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        build_cute(tiny_corpus, out, _small_config())
        tok = CUTETokenizerFast(
            tokenizer_file=out / "tokenizer.json",
            cute_mapping_file=out / "cute_mapping.json",
        )
        batch = ["def x(): pass", "class Y: pass", "return 42"]
        encoded = tok(batch, add_special_tokens=False)
        assert len(encoded.input_ids) == len(batch)
        for text, ids in zip(batch, encoded.input_ids, strict=True):
            decoded = tok.decode(ids, skip_special_tokens=True)
            assert decoded == text


class TestEmptyCorpus:
    def test_empty_corpus_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError, match="zero tokens"):
            build_cute(empty, tmp_path / "out", _small_config())


class TestPUAAtomicity:
    """The PUA chars must end up as single-ID vocab entries — no merging,
    no byte-splitting, no AddedToken misconfiguration."""

    def test_pua_chars_are_atomic_vocab_entries(self, tiny_corpus: Path, tmp_path: Path) -> None:
        """Every PUA char in the mapping must end up as exactly one vocab ID.

        Catches a regression where BPE merges or AddedToken misconfiguration
        would split PUA chars into multiple tokens at encode time.
        """
        out = tmp_path / "out"
        build_cute(tiny_corpus, out, _small_config())
        tok = CUTETokenizerFast(
            tokenizer_file=out / "tokenizer.json",
            cute_mapping_file=out / "cute_mapping.json",
        )

        # Every PUA char must encode to exactly one ID. We bypass the wrapper's
        # PUA substitution by feeding the PUA char directly to the underlying
        # tokenizer — this is the pure atomicity check.
        for pua_char in tok.cute_mapping.pua_chars:
            ids = tok.backend_tokenizer.encode(pua_char, add_special_tokens=False).ids
            assert len(ids) == 1, (
                f"PUA char {pua_char!r} (U+{ord(pua_char):04X}) tokenized to "
                f"{len(ids)} IDs, expected 1"
            )

    def test_pua_chars_have_dedicated_vocab_ids(self, tiny_corpus: Path, tmp_path: Path) -> None:
        """Every PUA char must appear in the vocab map."""
        out = tmp_path / "out"
        build_cute(tiny_corpus, out, _small_config())
        tok = CUTETokenizerFast(
            tokenizer_file=out / "tokenizer.json",
            cute_mapping_file=out / "cute_mapping.json",
        )
        vocab = tok.get_vocab()
        for pua_char in tok.cute_mapping.pua_chars:
            assert pua_char in vocab, (
                f"PUA char {pua_char!r} (U+{ord(pua_char):04X}) missing from vocab"
            )
