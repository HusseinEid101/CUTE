"""Unit tests for cute_tokenizer.corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from cute_tokenizer.corpus import (
    detect_license,
    has_secret,
    ingest_corpus,
    is_license_allowed,
    iter_corpus_files,
    iter_shard_texts,
    iter_shards,
    read_shard,
)


class TestSecretScrub:
    @pytest.mark.parametrize(
        "text",
        [
            "ACCESS_KEY = AKIAIOSFODNN7EXAMPLE",
            "OPENAI_KEY=sk-abc123def456ghi789jklmnopqrstuvwxyz12345",
            "GH_TOKEN=ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789",
            "AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R",
            "-----BEGIN RSA PRIVATE KEY-----",
            "header.eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturedata12345",
        ],
    )
    def test_detects_secrets(self, text: str) -> None:
        assert has_secret(text) is not None

    def test_normal_code_passes(self) -> None:
        assert has_secret("def hello(): return 42") is None
        assert has_secret("user_id = get_user_by_email(email)") is None

    def test_partial_match_does_not_trigger(self) -> None:
        # "AKIA" alone is not enough — full pattern requires 16 trailing chars.
        assert has_secret("variable AKIA = 'something'") is None


class TestIterCorpusFiles:
    def test_finds_matching_extensions(self, tiny_corpus: Path) -> None:
        files = list(iter_corpus_files(tiny_corpus, [".py", ".js", ".ts"]))
        names = {p.name for p in files}
        assert "math.py" in names
        assert "user.py" in names
        assert "utils.js" in names
        assert "service.ts" in names

    def test_deterministic_order(self, tiny_corpus: Path) -> None:
        a = list(iter_corpus_files(tiny_corpus, [".py", ".js", ".ts", ".md"]))
        b = list(iter_corpus_files(tiny_corpus, [".py", ".js", ".ts", ".md"]))
        assert a == b

    def test_skips_oversized_files(self, tmp_path: Path) -> None:
        big = tmp_path / "big.py"
        big.write_text("x" * 10_000_000)
        files = list(iter_corpus_files(tmp_path, [".py"], max_bytes=1_000_000))
        assert big not in files


class TestIngestCorpus:
    def test_creates_shards(self, tiny_corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        stats = ingest_corpus(
            corpus_dir=tiny_corpus,
            out_dir=out,
            extensions=[".py", ".js", ".ts", ".md"],
            shard_size_bytes=1_000_000,
        )
        assert stats.files_kept > 0
        shards = list(iter_shards(out / "shards"))
        assert len(shards) >= 1

    def test_dedup(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # Two files with identical content
        (corpus / "a.py").write_text("def foo(): pass\n")
        (corpus / "b.py").write_text("def foo(): pass\n")
        out = tmp_path / "out"
        stats = ingest_corpus(
            corpus_dir=corpus,
            out_dir=out,
            extensions=[".py"],
        )
        assert stats.files_seen == 2
        assert stats.files_kept == 1
        assert stats.files_dropped_dedup == 1

    def test_secret_drop(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "good.py").write_text("def foo(): pass\n")
        (corpus / "bad.py").write_text("KEY = 'AKIAIOSFODNN7EXAMPLE'\n")
        out = tmp_path / "out"
        stats = ingest_corpus(
            corpus_dir=corpus,
            out_dir=out,
            extensions=[".py"],
        )
        assert stats.files_kept == 1
        assert stats.files_dropped_secret == 1

    def test_secret_scrub_disabled(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "bad.py").write_text("KEY = 'AKIAIOSFODNN7EXAMPLE'\n")
        out = tmp_path / "out"
        stats = ingest_corpus(
            corpus_dir=corpus,
            out_dir=out,
            extensions=[".py"],
            enable_secret_scrub=False,
        )
        assert stats.files_kept == 1
        assert stats.files_dropped_secret == 0

    def test_pua_in_corpus_recorded(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # Use a PUA char in the corpus
        from cute_tokenizer.pua import PUA_BMP_START

        (corpus / "weird.py").write_text(
            f"def foo(): return '{chr(PUA_BMP_START)}'\n", encoding="utf-8"
        )
        out = tmp_path / "out"
        stats = ingest_corpus(
            corpus_dir=corpus,
            out_dir=out,
            extensions=[".py"],
        )
        assert PUA_BMP_START in stats.pua_codepoints_in_corpus

    def test_round_trip_records(self, tiny_corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        ingest_corpus(
            corpus_dir=tiny_corpus,
            out_dir=out,
            extensions=[".py", ".js", ".ts", ".md"],
        )
        # Confirm we can stream records back. The iter_shard_texts call
        # exercises the streaming path; the inner loop checks per-record
        # invariants.
        assert list(iter_shard_texts(out / "shards"))  # non-empty
        for shard in iter_shards(out / "shards"):
            for rec in read_shard(shard):
                assert rec.path
                assert rec.text
                assert len(rec.sha256) == 64  # sha256 hex

    def test_deterministic_shards(self, tiny_corpus_factory, tmp_path: Path) -> None:
        """Two ingest runs on identical inputs produce byte-equal shards."""
        c1 = tiny_corpus_factory("c1")
        c2 = tiny_corpus_factory("c2")
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        ingest_corpus(
            corpus_dir=c1,
            out_dir=out1,
            extensions=[".py", ".js", ".ts", ".md"],
        )
        ingest_corpus(
            corpus_dir=c2,
            out_dir=out2,
            extensions=[".py", ".js", ".ts", ".md"],
        )
        shards1 = sorted((out1 / "shards").glob("shard_*.jsonl.gz"))
        shards2 = sorted((out2 / "shards").glob("shard_*.jsonl.gz"))
        assert len(shards1) == len(shards2)
        # Compare uncompressed content (gzip headers may include timestamps)
        for s1, s2 in zip(shards1, shards2, strict=True):
            import gzip

            with gzip.open(s1, "rb") as f1, gzip.open(s2, "rb") as f2:
                assert f1.read() == f2.read()


class TestLicenseFilter:
    def test_detect_spdx(self) -> None:
        text = "# SPDX-License-Identifier: MIT\nimport os\n"
        assert detect_license(text) == "MIT"

    def test_detect_apache(self) -> None:
        text = "// SPDX-License-Identifier: Apache-2.0\nfunction x(){}\n"
        assert detect_license(text) == "Apache-2.0"

    def test_detect_none(self) -> None:
        assert detect_license("just code, no header\n") is None

    def test_allowed_with_matching_spdx(self) -> None:
        text = "# SPDX-License-Identifier: MIT\nx=1\n"
        assert is_license_allowed(text, ["MIT", "Apache-2.0"])

    def test_rejected_with_unmatched_spdx(self) -> None:
        text = "# SPDX-License-Identifier: GPL-3.0\nx=1\n"
        assert not is_license_allowed(text, ["MIT", "Apache-2.0"])

    def test_rejected_proprietary_marker(self) -> None:
        text = "# Copyright Acme Corp. All Rights Reserved.\nx=1\n"
        assert not is_license_allowed(text, ["MIT"])

    def test_rejected_unlicensed(self) -> None:
        text = '"""License: UNLICENSED"""\nx=1\n'
        assert not is_license_allowed(text, ["MIT"])

    def test_no_header_no_marker_allowed(self) -> None:
        # Plain code with no headers is allowed (corpus owner's responsibility).
        assert is_license_allowed("def foo(): pass\n", ["MIT"])

    def test_ingest_license_filter_drops_files(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "ok.py").write_text(
            "# SPDX-License-Identifier: MIT\ndef foo(): pass\n", encoding="utf-8"
        )
        (corpus / "bad.py").write_text(
            "# SPDX-License-Identifier: GPL-3.0\ndef bar(): pass\n", encoding="utf-8"
        )
        (corpus / "proprietary.py").write_text(
            "# All Rights Reserved\ndef baz(): pass\n", encoding="utf-8"
        )
        out = tmp_path / "out"
        stats = ingest_corpus(
            corpus_dir=corpus,
            out_dir=out,
            extensions=[".py"],
            enable_license_filter=True,
            license_allowlist=("MIT", "Apache-2.0"),
        )
        assert stats.files_kept == 1
        assert stats.files_dropped_license == 2

    def test_ingest_license_filter_disabled_keeps_all(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "gpl.py").write_text(
            "# SPDX-License-Identifier: GPL-3.0\ndef bar(): pass\n", encoding="utf-8"
        )
        out = tmp_path / "out"
        stats = ingest_corpus(
            corpus_dir=corpus,
            out_dir=out,
            extensions=[".py"],
            enable_license_filter=False,
        )
        assert stats.files_kept == 1
        assert stats.files_dropped_license == 0
