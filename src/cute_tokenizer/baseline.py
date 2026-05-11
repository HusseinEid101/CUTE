"""Baseline tokenizer adapters for savings-based candidate scoring.

The selection step asks "how many baseline tokens would this string cost?"
to compute net token savings per PUA candidate. We provide a thin protocol
plus a `tiktoken`-backed cl100k implementation and a fallback.

`tiktoken` is an *optional* dependency — install via
`pip install 'cute-tokenizer[baseline]'`. The CUTE inference path does not
need it.
"""

from __future__ import annotations

import functools
import warnings
from typing import Protocol


class BaselineTokenizer(Protocol):
    """Counts baseline tokens for a string. Implementations must be deterministic
    and side-effect-free with respect to identical inputs."""

    name: str

    def count_tokens(self, text: str) -> int:
        """Return the number of baseline tokens for `text`. Must be ≥ 0."""
        ...


class Cl100kBaseline:
    """OpenAI cl100k baseline (GPT-4 / cl100k_base via `tiktoken`).

    Caches per-token counts in an in-process LRU. Thread-safe for read-after-init.
    """

    name: str = "cl100k_base"

    def __init__(self, cache_size: int = 1_000_000) -> None:
        try:
            import tiktoken  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Cl100kBaseline requires tiktoken. "
                "Install with: pip install 'cute-tokenizer[baseline]'"
            ) from e
        self._enc = tiktoken.get_encoding("cl100k_base")
        # Bind a cached encode-and-count over `self`. We don't cache on the
        # encoding object directly because tiktoken's BPE handles are not hashable.
        self._count = functools.lru_cache(maxsize=cache_size)(self._count_uncached)

    def _count_uncached(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return self._count(text)


class NullBaseline:
    """Fallback that scores every token as costing 1 baseline token.

    Under this baseline, `compute_candidate_score` collapses to `0` for any
    single-token input — i.e. every PUA substitution is a wash. Useful only
    as a degenerate fallback that effectively disables savings-based ranking.
    Emits a one-time warning when used so callers know they're getting
    frequency-based ranking, not savings-based.
    """

    name: str = "null"
    _warned = False

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if not NullBaseline._warned:
            warnings.warn(
                "NullBaseline in use — savings scores will all be 0. "
                "Install 'cute-tokenizer[baseline]' for cl100k-aware ranking.",
                stacklevel=2,
            )
            NullBaseline._warned = True
        return 1


def get_default_baseline() -> BaselineTokenizer:
    """Return `Cl100kBaseline` if tiktoken is available, else `NullBaseline`.

    Callers that *require* cl100k-aware ranking (e.g. `cute build`) should
    construct `Cl100kBaseline()` directly so they get a clear ImportError
    instead of silently degraded scoring.
    """
    try:
        return Cl100kBaseline()
    except ImportError:
        return NullBaseline()


__all__ = [
    "BaselineTokenizer",
    "Cl100kBaseline",
    "NullBaseline",
    "get_default_baseline",
]
