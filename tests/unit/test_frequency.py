"""Unit tests for cute_tokenizer.frequency."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from cute_tokenizer.corpus import ingest_corpus
from cute_tokenizer.frequency import count_frequencies, count_in_text


class TestCountInText:
    def test_basic_counts(self) -> None:
        c: Counter[str] = Counter()
        b: defaultdict[str, float] = defaultdict(float)
        count_in_text(
            "def foo(): return foo() + foo()",
            c,
            b,
            boost_weight=0.0,
            max_token_len=50,
        )
        assert c["def"] == 1
        assert c["foo"] == 3
        assert c["return"] == 1
        assert c["("] == 3
        assert c[")"] == 3

    def test_identifier_boost(self) -> None:
        c: Counter[str] = Counter()
        b: defaultdict[str, float] = defaultdict(float)
        count_in_text("getUserId getUserId", c, b, boost_weight=0.5, max_token_len=50)
        assert c["getUserId"] == 2
        # Sub-parts get boosted
        assert b["get"] >= 1.0
        assert b["User"] >= 1.0
        assert b["Id"] >= 1.0

    def test_no_boost_for_zero_weight(self) -> None:
        c: Counter[str] = Counter()
        b: defaultdict[str, float] = defaultdict(float)
        count_in_text("getUserId", c, b, boost_weight=0.0, max_token_len=50)
        assert dict(b) == {}

    def test_max_len_filter(self) -> None:
        c: Counter[str] = Counter()
        b: defaultdict[str, float] = defaultdict(float)
        long_token = "x" * 100
        count_in_text(f"def {long_token}", c, b, boost_weight=0.0, max_token_len=50)
        assert long_token not in c
        assert "def" in c


class TestCountFrequencies:
    def test_on_tiny_corpus(self, tiny_corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        ingest_corpus(
            corpus_dir=tiny_corpus,
            out_dir=out,
            extensions=[".py", ".js", ".ts"],
        )
        freq = count_frequencies(
            shards_dir=out / "shards",
            boost_weight=0.3,
            workers=1,
        )
        # Common Python tokens should be present
        assert freq["def"] > 0
        assert freq["return"] > 0
        assert freq["self"] > 0
        assert freq["("] > 0
        assert freq[":"] > 0

    def test_empty_shards(self, tmp_path: Path) -> None:
        shards = tmp_path / "shards"
        shards.mkdir()
        freq = count_frequencies(shards_dir=shards, workers=1)
        assert freq == Counter()

    def test_boost_actually_increases_count(
        self,
        tiny_corpus: Path,
        tmp_path: Path,
    ) -> None:
        """Identifier sub-parts should get nonzero counts even if they
        never appear as standalone tokens (the v2.1 'int(0.3) == 0' bug)."""
        out = tmp_path / "out"
        ingest_corpus(
            corpus_dir=tiny_corpus,
            out_dir=out,
            extensions=[".py", ".js", ".ts"],
        )
        freq_no_boost = count_frequencies(
            shards_dir=out / "shards",
            boost_weight=0.0,
            workers=1,
        )
        freq_boost = count_frequencies(
            shards_dir=out / "shards",
            boost_weight=0.5,
            workers=1,
        )
        # 'Total' is a sub-part of 'calculateTotal' and 'formatCurrency' has
        # 'format' / 'Currency' sub-parts.
        # At least one sub-part token must have a higher count with boosting.
        any_increased = any(
            freq_boost[k] > freq_no_boost[k]
            for k in ["Total", "calculate", "User", "Service", "Currency"]
        )
        assert any_increased, "boost weight had no observable effect"

    def test_determinism(self, tiny_corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        ingest_corpus(
            corpus_dir=tiny_corpus,
            out_dir=out,
            extensions=[".py", ".js", ".ts"],
        )
        f1 = count_frequencies(shards_dir=out / "shards", workers=1)
        f2 = count_frequencies(shards_dir=out / "shards", workers=1)
        assert f1 == f2
