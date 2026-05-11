"""Parallel frequency counting over corpus shards.

The output is a Counter mapping token → integer count, where:
- Each whole token from `iter_tokens` contributes its raw count.
- For ASCII identifiers, sub-parts contribute fractional counts via
  `boost_weight`, accumulated as floats and ceiled at the end so the boost
  is actually nonzero (fix #4 from the build plan: the v2.1 draft used
  `int(0.3) == 0`, silently disabling the boost).
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ._accel_loader import USE_RUST, accel
from .corpus import iter_shards, read_shard
from .patterns import is_identifier, iter_tokens, split_identifier


def count_in_text(
    text: str,
    counter: Counter[str],
    boost_acc: defaultdict[str, float],
    boost_weight: float,
    max_token_len: int,
) -> None:
    """Accumulate token counts from `text` into `counter` + `boost_acc`."""
    if USE_RUST:
        c_delta, b_delta = accel.count_in_text(text, boost_weight, max_token_len)
        for k, v in c_delta.items():
            counter[k] += v
        for k, v in b_delta.items():
            boost_acc[k] += v
        return
    for tok, _, _ in iter_tokens(text):
        if len(tok) > max_token_len:
            continue
        counter[tok] += 1
        if boost_weight > 0 and is_identifier(tok):
            for part in split_identifier(tok):
                if part != tok:
                    boost_acc[part] += boost_weight


def _count_shard(
    args: tuple[Path, float, int],
) -> tuple[Counter[str], dict[str, float]]:
    """Worker: count one shard, return (raw counter, boost accumulator)."""
    shard_path, boost_weight, max_token_len = args
    counter: Counter[str] = Counter()
    boost_acc: defaultdict[str, float] = defaultdict(float)
    for rec in read_shard(shard_path):
        count_in_text(rec.text, counter, boost_acc, boost_weight, max_token_len)
    return counter, dict(boost_acc)


def count_frequencies(
    shards_dir: Path,
    boost_weight: float = 0.3,
    max_token_len: int = 50,
    workers: int = 0,
) -> Counter[str]:
    """Count token frequencies across all shards.

    Parameters
    ----------
    shards_dir
        Directory containing `shard_*.jsonl.gz` produced by `ingest_corpus`.
    boost_weight
        Fractional weight added per identifier sub-part occurrence.
        Sub-parts that never appear as standalone tokens still get nonzero
        counts and become eligible for PUA assignment.
    max_token_len
        Tokens longer than this are ignored entirely.
    workers
        0 → use os.cpu_count(); 1 → run sequentially in this process;
        N>1 → use a ProcessPoolExecutor with N workers.
    """
    shards = list(iter_shards(shards_dir))
    if not shards:
        return Counter()

    if USE_RUST:
        # Rust path: Rayon-parallel shard processing inside the extension.
        # `workers` is ignored — Rayon manages its own thread pool sized to
        # available CPUs. Returns a flat dict already merged with the boost.
        rust_dict = accel.count_frequencies([str(s) for s in shards], boost_weight, max_token_len)
        return Counter(rust_dict)

    if workers == 0:
        workers = os.cpu_count() or 1

    args_list = [(s, boost_weight, max_token_len) for s in shards]

    total_counter: Counter[str] = Counter()
    boost_total: defaultdict[str, float] = defaultdict(float)

    if workers <= 1:
        results: Iterator[tuple[Counter[str], dict[str, float]]] = (
            _count_shard(a) for a in args_list
        )
    else:
        # imap-style streaming via map; ordered for determinism.
        # `with` guarantees pool shutdown even on exception.
        def _run() -> Iterator[tuple[Counter[str], dict[str, float]]]:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                yield from ex.map(_count_shard, args_list)

        results = _run()

    for c, b in results:
        total_counter.update(c)
        for k, v in b.items():
            boost_total[k] += v

    # Merge boost into the counter, ceiling fractions so any nonzero boost
    # produces at least 1 count. This guarantees the boost is observable.
    for tok, frac in boost_total.items():
        bonus = math.ceil(frac)
        if bonus > 0:
            total_counter[tok] += bonus

    return total_counter


__all__ = ["count_frequencies", "count_in_text"]
