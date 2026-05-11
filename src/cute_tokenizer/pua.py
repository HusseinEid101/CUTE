"""Private Use Area codepoint allocation.

Available space (after subtracting non-character codepoints):
- BMP (U+E000..U+F8FF):           6_400 slots
- Plane 15 (U+F0000..U+FFFFD):   65_534 slots  (FFFFE/FFFFF are non-chars)
- Plane 16 (U+100000..U+10FFFD): 65_534 slots  (10FFFE/10FFFF are non-chars)
Total: 137_468 slots - comfortably exceeds the 30k-70k tokens needed for
90% coverage on real code corpora.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# Range bounds. Inclusive on both ends; non-character codepoints excluded.
PUA_BMP_START = 0xE000
PUA_BMP_END = 0xF8FF  # inclusive

PUA_P15_START = 0xF0000
PUA_P15_END = 0xFFFFD  # FFFFE & FFFFF are non-characters by Unicode

PUA_P16_START = 0x100000
PUA_P16_END = 0x10FFFD  # 10FFFE & 10FFFF are non-characters

PUA_BMP_SIZE = PUA_BMP_END - PUA_BMP_START + 1  # 6_400
PUA_P15_SIZE = PUA_P15_END - PUA_P15_START + 1  # 65_534
PUA_P16_SIZE = PUA_P16_END - PUA_P16_START + 1  # 65_534
PUA_TOTAL = PUA_BMP_SIZE + PUA_P15_SIZE + PUA_P16_SIZE  # 137_468


def _iter_pua_codepoints(
    skip: frozenset[int] = frozenset(),
    *,
    skip_bmp: bool = False,
) -> Iterable[int]:
    """Yield PUA codepoints in priority order.

    Default (skip_bmp=False): BMP first (3-byte UTF-8), then P15, then P16.
    BMP order minimizes UTF-8 byte cost but BMP PUAs occasionally appear
    in real source text (Asian fonts, Unicode mapping tables) — using
    them risks decode-time collisions where the user's literal PUA char
    gets reverse-substituted into a mapped word.

    With skip_bmp=True: assigns only supplementary-plane PUAs (P15 + P16,
    131,068 slots). These are virtually never used in real text, making
    roundtrip byte-exact on any input. Costs 1 extra UTF-8 byte per
    PUA char (4 vs 3) — minor compression hit for guaranteed correctness.
    """
    if not skip_bmp:
        for cp in range(PUA_BMP_START, PUA_BMP_END + 1):
            if cp not in skip:
                yield cp
    for cp in range(PUA_P15_START, PUA_P15_END + 1):
        if cp not in skip:
            yield cp
    for cp in range(PUA_P16_START, PUA_P16_END + 1):
        if cp not in skip:
            yield cp


def is_pua_char(ch: str) -> bool:
    """True iff `ch` is a single character in any of the 3 PUA ranges."""
    if len(ch) != 1:
        return False
    cp = ord(ch)
    return (
        PUA_BMP_START <= cp <= PUA_BMP_END
        or PUA_P15_START <= cp <= PUA_P15_END
        or PUA_P16_START <= cp <= PUA_P16_END
    )


@dataclass(frozen=True)
class PUAMapping:
    """An assignment of words → PUA characters.

    Round-trip invariant: ``pua_to_word[word_to_pua[w]] == w`` for all w.
    """

    word_to_pua: dict[str, str]
    pua_to_word: dict[str, str]
    skipped_codepoints: tuple[int, ...]  # PUA codepoints found in corpus, NOT used

    def __post_init__(self) -> None:
        if len(self.word_to_pua) != len(self.pua_to_word):
            raise ValueError(
                f"Mapping is not bijective: {len(self.word_to_pua)} words vs "
                f"{len(self.pua_to_word)} chars"
            )
        for w, c in self.word_to_pua.items():
            if not is_pua_char(c):
                raise ValueError(f"Mapping contains non-PUA char for word {w!r}: {c!r}")
            if self.pua_to_word.get(c) != w:
                raise ValueError(f"Mapping is not bijective at {w!r} <-> {c!r}")

    @property
    def size(self) -> int:
        return len(self.word_to_pua)

    @property
    def pua_chars(self) -> list[str]:
        """All PUA chars in assignment order (sorted by codepoint for determinism)."""
        return sorted(self.pua_to_word.keys(), key=ord)


def assign_pua_mapping(
    tokens: list[str],
    corpus_pua_codepoints: frozenset[int] = frozenset(),
    *,
    skip_bmp: bool = False,
) -> PUAMapping:
    """Assign each token in `tokens` to a unique PUA codepoint.

    Parameters
    ----------
    tokens
        Tokens in priority order (most frequent first).
    corpus_pua_codepoints
        PUA codepoints observed in the training corpus. These are SKIPPED
        during assignment to avoid ambiguity (a corpus PUA char could be
        confused with our injected substitution).
    skip_bmp
        If True, do not use BMP (`U+E000`-`U+F8FF`) for assignments.
        Recommended for production use because BMP PUAs occasionally
        appear in real source text (Asian fonts, Unicode mapping tables),
        which would cause decode-time collisions on user input. Cost: 1
        extra UTF-8 byte per PUA char (4 vs 3). Supplementary planes
        (P15+P16) provide 131,068 slots — far more than typical
        `pua_budget` settings.

    Raises
    ------
    ValueError
        If `tokens` contains duplicates or exceeds available PUA space.
    """
    if len(tokens) != len(set(tokens)):
        raise ValueError("assign_pua_mapping: input contains duplicate tokens")

    total_available = PUA_TOTAL - (PUA_BMP_SIZE if skip_bmp else 0)
    available = total_available - len(corpus_pua_codepoints)
    if len(tokens) > available:
        raise ValueError(
            f"Need {len(tokens)} PUA slots but only {available} available "
            f"(skip_bmp={skip_bmp}, "
            f"corpus codepoints skipped={len(corpus_pua_codepoints)})"
        )

    word_to_pua: dict[str, str] = {}
    pua_to_word: dict[str, str] = {}

    cp_iter = iter(_iter_pua_codepoints(skip=corpus_pua_codepoints, skip_bmp=skip_bmp))
    for tok in tokens:
        cp = next(cp_iter)
        ch = chr(cp)
        word_to_pua[tok] = ch
        pua_to_word[ch] = tok

    return PUAMapping(
        word_to_pua=word_to_pua,
        pua_to_word=pua_to_word,
        skipped_codepoints=tuple(sorted(corpus_pua_codepoints)),
    )


def find_pua_codepoints(text: str) -> set[int]:
    """Return the set of PUA codepoints that appear in `text`."""
    return {ord(c) for c in text if is_pua_char(c)}


__all__ = [
    "PUA_BMP_END",
    "PUA_BMP_START",
    "PUA_P15_END",
    "PUA_P15_START",
    "PUA_P16_END",
    "PUA_P16_START",
    "PUA_TOTAL",
    "PUAMapping",
    "assign_pua_mapping",
    "find_pua_codepoints",
    "is_pua_char",
]
