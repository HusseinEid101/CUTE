"""PUA candidate selection.

Two strategies are provided:

* `select_by_savings` — the production path. Ranks candidates by
  `frequency * (baseline_token_cost - 1)` so PUA slots go to tokens that
  actually save tokens vs the baseline (cl100k). Tokens whose baseline
  cost is 1 (single byte/char) score 0 — byte fallback already handles
  them optimally.

* `select_by_coverage` — the legacy v0 strategy. Ranks by raw frequency.
  Kept as a deprecated shim for callers that haven't migrated; emits a
  `DeprecationWarning` on use.

The savings strategy also applies a tier-aware penalty: tokens beyond
`PUA_BMP_SIZE` would be assigned a 4-byte supplementary-plane PUA char
(vs 3 bytes for BMP). For tokens whose token-savings == 1, the byte cost
of substitution can erase the saving, so by default we cap the budget at
`PUA_BMP_SIZE` unless the caller opts into supplementary planes.
"""

from __future__ import annotations

import warnings
from collections import Counter

import regex as _re

from .baseline import BaselineTokenizer
from .pua import PUA_BMP_SIZE

DEFAULT_MAX_LEN = 50

# Heuristic regex for hash / UUID / base64-blob shapes — these waste PUA slots.
# Keep conservative: false positives just demote a token from PUA to byte
# fallback, which is fine.
_HEX_HASH_RE = _re.compile(r"^[0-9a-f]{16,}$", _re.IGNORECASE)
_UUID_RE = _re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", _re.IGNORECASE
)
_BASE64_BLOB_RE = _re.compile(r"^[A-Za-z0-9+/=_\-]{24,}$")
# A "blob" must be mostly base64 alphabet with no recognizable English-word
# substrings. We approximate: if the token is long, has high alphabet entropy,
# and at least 1/3 digits or 1/3 mixed-case alpha runs, treat it as a blob.


def _looks_like_secret_or_hash(tok: str) -> bool:
    """Heuristic: True if `tok` looks like a hash, UUID, or base64 blob."""
    if _UUID_RE.match(tok):
        return True
    if _HEX_HASH_RE.match(tok):
        return True
    if len(tok) >= 24 and _BASE64_BLOB_RE.match(tok):
        # Has the shape — confirm it lacks lowercase-word substructure.
        # A real identifier has long-ish lowercase runs; a base64 blob is
        # interleaved upper/lower/digit. Require at least 3 short runs of
        # contiguous lowercase letters before we accept it as "wordy".
        lower_runs = _re.findall(r"[a-z]{3,}", tok)
        if len(lower_runs) < 2:
            return True
    return False


def is_good_token(tok: str, max_len: int = DEFAULT_MAX_LEN) -> bool:
    """Filter for PUA-eligible tokens.

    Rejects:
    - Empty / whitespace-only / over-length tokens.
    - Tokens that encode to a single UTF-8 byte (byte fallback handles them
      optimally, so a PUA assignment is at best a wash).
    - Hashes, UUIDs, long base64 blobs (waste PUA slots; near-zero
      cross-document reuse).
    """
    if not tok:
        return False
    if tok.isspace():
        return False
    if len(tok) > max_len:
        return False
    if len(tok.encode("utf-8")) <= 1:
        return False
    return not _looks_like_secret_or_hash(tok)


def compute_candidate_score(
    token: str,
    frequency: int,
    baseline: BaselineTokenizer,
) -> float:
    """Net token savings per occurrence times frequency.

    `savings = max(0, baseline.count_tokens(token) - 1)`.

    A token with `baseline_count == 1` scores 0 — substituting it costs a
    PUA slot but doesn't shorten the token sequence. A token with
    `baseline_count == 5` and frequency 1000 scores 4000.

    Note: this scoring is intentionally tier-agnostic. The tier-aware
    penalty (BMP vs supplementary plane) is applied in `select_by_savings`
    after ranking, not here, because the tier of any given candidate
    depends on how many candidates outrank it.
    """
    baseline_cost = baseline.count_tokens(token)
    savings = max(0, baseline_cost - 1)
    return frequency * savings


def select_by_savings(
    freq: Counter[str],
    baseline: BaselineTokenizer,
    *,
    vocab_budget: int,
    max_len: int = DEFAULT_MAX_LEN,
    min_score: float = 1.0,
    allow_supplementary_pua: bool = False,
) -> list[str]:
    """Select PUA candidates by net token savings.

    Parameters
    ----------
    freq
        Frequency counter from `count_frequencies`.
    baseline
        Baseline tokenizer used to score `(baseline_cost - 1)` per token.
    vocab_budget
        Hard upper bound on the number of selected tokens. Capped at
        `PUA_BMP_SIZE` unless `allow_supplementary_pua` is True.
    max_len
        Reject tokens longer than this.
    min_score
        Reject tokens whose score is below this threshold. Default 1.0
        (i.e., at least one token saved across the entire corpus).
    allow_supplementary_pua
        If True, the budget is uncapped (limited only by `PUA_TOTAL`).
        Off by default because supplementary-plane PUA chars are 4 bytes
        each and net savings can go negative for `baseline_count==2`
        candidates.

    Returns
    -------
    Tokens in deterministic order: descending score, ascending lex for ties.
    """
    if vocab_budget <= 0:
        return []

    effective_budget = vocab_budget
    if not allow_supplementary_pua:
        effective_budget = min(effective_budget, PUA_BMP_SIZE)

    scored: list[tuple[float, str]] = []
    for token, count in freq.items():
        if not is_good_token(token, max_len=max_len):
            continue
        score = compute_candidate_score(token, count, baseline)
        if score < min_score:
            continue
        scored.append((score, token))

    # Sort by (-score, token) for deterministic, savings-priority order.
    scored.sort(key=lambda st: (-st[0], st[1]))

    return [tok for _, tok in scored[:effective_budget]]


def select_by_coverage(
    freq: Counter[str],
    coverage_target: float = 0.90,
    max_len: int = DEFAULT_MAX_LEN,
    max_tokens: int | None = None,
) -> list[str]:
    """Frequency-based selection (deprecated, retained for backward compat).

    Stops at the first of: cumulative coverage ≥ `coverage_target`,
    `max_tokens` selected, or input exhausted. New code should use
    `select_by_savings`.
    """
    if not 0.0 < coverage_target <= 1.0:
        raise ValueError(f"coverage_target must be in (0,1], got {coverage_target}")

    total = sum(freq.values())
    if total == 0:
        return []

    sorted_items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))

    selected: list[str] = []
    cumulative = 0
    threshold = coverage_target * total

    for token, count in sorted_items:
        if not is_good_token(token, max_len=max_len):
            continue
        selected.append(token)
        cumulative += count
        if max_tokens is not None and len(selected) >= max_tokens:
            break
        if cumulative >= threshold:
            break

    return selected


def coverage_of(freq: Counter[str], tokens: list[str]) -> float:
    """Fraction of total frequency covered by `tokens`. Used for diagnostics."""
    total = sum(freq.values())
    if total == 0:
        return 0.0
    covered = sum(freq.get(t, 0) for t in tokens)
    return covered / total


def select_by_coverage_deprecated(
    freq: Counter[str],
    coverage_target: float = 0.90,
    max_len: int = DEFAULT_MAX_LEN,
    max_tokens: int | None = None,
) -> list[str]:
    """Shim that emits a DeprecationWarning then delegates to `select_by_coverage`."""
    warnings.warn(
        "select_by_coverage is deprecated; use select_by_savings with a "
        "BaselineTokenizer for production builds.",
        DeprecationWarning,
        stacklevel=2,
    )
    return select_by_coverage(freq, coverage_target, max_len, max_tokens)


__all__ = [
    "DEFAULT_MAX_LEN",
    "compute_candidate_score",
    "coverage_of",
    "is_good_token",
    "select_by_coverage",
    "select_by_coverage_deprecated",
    "select_by_savings",
]
