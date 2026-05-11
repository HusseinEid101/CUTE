"""Unit tests for cute_tokenizer.manifest."""

from __future__ import annotations

import json
from pathlib import Path

from cute_tokenizer.manifest import (
    BuildManifest,
    determinism_diff,
    hash_corpus_shards,
    hash_vocab,
    make_manifest,
)


class TestHashVocab:
    def test_empty(self) -> None:
        assert hash_vocab({}) == hash_vocab({})

    def test_order_independent(self) -> None:
        a = {"a": 0, "b": 1, "c": 2}
        b = {"c": 2, "a": 0, "b": 1}
        assert hash_vocab(a) == hash_vocab(b)

    def test_different_vocabs_differ(self) -> None:
        assert hash_vocab({"a": 0}) != hash_vocab({"a": 1})
        assert hash_vocab({"a": 0}) != hash_vocab({"b": 0})


class TestHashCorpusShards:
    def test_no_shards(self, tmp_path: Path) -> None:
        d = tmp_path / "shards"
        d.mkdir()
        h1 = hash_corpus_shards(d)
        h2 = hash_corpus_shards(d)
        assert h1 == h2

    def test_with_shards(self, tmp_path: Path) -> None:
        import gzip

        d = tmp_path / "shards"
        d.mkdir()
        with gzip.open(d / "shard_00000.jsonl.gz", "wb") as f:
            f.write(b'{"x":1}\n')
        with gzip.open(d / "shard_00001.jsonl.gz", "wb") as f:
            f.write(b'{"x":2}\n')
        h1 = hash_corpus_shards(d)
        # Re-hash to confirm the function is deterministic on the same FS state.
        assert hash_corpus_shards(d) == h1
        # Note: gzip headers may include mtime; if so, this test is meant
        # to reveal it. We hash file contents (raw bytes) so changes there
        # would shift the hash.
        # We accept that two writes at different times may differ; we just
        # confirm the function is total.
        assert isinstance(h1, str)
        assert len(h1) == 64


class TestBuildManifest:
    def test_round_trip(self, tmp_path: Path) -> None:
        m1 = make_manifest(
            config={"vocab_size": 1000, "coverage_target": 0.9},
            corpus_hash="abc",
            vocab_hash="def",
            pua_mapping_size=42,
            pua_codepoints_in_corpus=[0xE000, 0xE001],
            coverage_achieved=0.91,
        )
        path = tmp_path / "manifest.json"
        m1.write(path)

        m2 = BuildManifest.read(path)
        assert m2.corpus_hash == "abc"
        assert m2.vocab_hash == "def"
        assert m2.pua_mapping_size == 42
        assert m2.pua_codepoints_in_corpus == [0xE000, 0xE001]
        assert m2.coverage_achieved == 0.91

    def test_to_dict_is_json_serializable(self) -> None:
        m = make_manifest(
            config={"vocab_size": 1000},
            corpus_hash="x",
            vocab_hash="y",
            pua_mapping_size=0,
            pua_codepoints_in_corpus=[],
            coverage_achieved=0.0,
        )
        # Round-trip through json to confirm pure-data structure.
        s = json.dumps(m.to_dict())
        d = json.loads(s)
        assert d["cute_version"] == m.cute_version


class TestDeterminismDiff:
    def test_identical_no_diff(self) -> None:
        kwargs = {
            "config": {"vocab_size": 1000},
            "corpus_hash": "x",
            "vocab_hash": "y",
            "pua_mapping_size": 42,
            "pua_codepoints_in_corpus": [],
            "coverage_achieved": 0.9,
        }
        a = make_manifest(**kwargs)
        b = make_manifest(**kwargs)
        # build_host_info / timing differ but should be ignored.
        assert determinism_diff(a, b) == []

    def test_vocab_hash_difference_reported(self) -> None:
        a = make_manifest(
            config={"vocab_size": 1000},
            corpus_hash="x",
            vocab_hash="y",
            pua_mapping_size=42,
            pua_codepoints_in_corpus=[],
            coverage_achieved=0.9,
        )
        b = make_manifest(
            config={"vocab_size": 1000},
            corpus_hash="x",
            vocab_hash="DIFFERENT",
            pua_mapping_size=42,
            pua_codepoints_in_corpus=[],
            coverage_achieved=0.9,
        )
        diffs = determinism_diff(a, b)
        assert any("vocab_hash" in d for d in diffs)
