"""Post-train safety net for PUA atomicity.

The primary mechanism that lets BPE learn whitespace+PUA merges is
*pre-substitution of the training corpus* (see `trainer._train_bpe`). With
substitution applied to the training stream, two PUA chars from different
word-internal substitutions rarely end up adjacent in the symbol stream, so
BPE has no co-occurrence statistic to merge them. But "rarely" is not
"never" — adjacent PUAs can occur if two mapped tokens are lexically
contiguous in the corpus (e.g. mapped `"self"` immediately followed by
mapped `"."`).

This module audits the trained `tokenizer.json` for PUA-PUA merges and,
when `strict_pua_atomicity` is on, removes them and re-saves. We also
assert that the four invariants required for a valid CUTE tokenizer
survive any rewrite:

* `model.type == "BPE"`
* `decoder.type == "ByteLevel"`
* `pre_tokenizer.type == "ByteLevel"`
* every PUA char in the mapping is reachable as an atomic token id
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pua import PUAMapping, is_pua_char


def _split_merge(merge: Any) -> tuple[str, str]:
    """Normalize a merges entry: tokenizers stores either a space-joined
    string or a 2-tuple, depending on version."""
    if isinstance(merge, str):
        a, _, b = merge.partition(" ")
        return a, b
    if isinstance(merge, list | tuple) and len(merge) == 2:
        return str(merge[0]), str(merge[1])
    raise ValueError(f"unexpected merge entry shape: {merge!r}")


def _is_pua_str(s: str) -> bool:
    return len(s) == 1 and is_pua_char(s)


def find_pua_pua_merges(tokenizer_json: dict[str, Any]) -> list[tuple[int, str, str]]:
    """Return `(index, a, b)` for every merge whose both sides are PUA chars."""
    out: list[tuple[int, str, str]] = []
    merges = tokenizer_json.get("model", {}).get("merges", [])
    for i, m in enumerate(merges):
        try:
            a, b = _split_merge(m)
        except ValueError:
            continue
        if _is_pua_str(a) and _is_pua_str(b):
            out.append((i, a, b))
    return out


def _vocab_id_for_token(tokenizer_json: dict[str, Any], token: str) -> int | None:
    vocab = tokenizer_json.get("model", {}).get("vocab", {})
    val = vocab.get(token)
    if isinstance(val, int):
        return val
    return None


def assert_invariants(tokenizer_json: dict[str, Any], mapping: PUAMapping) -> None:
    """Raise `AssertionError` if the tokenizer JSON has lost any required invariant.

    Cheap to call — runs in O(|added_tokens| + |mapping|).
    """
    model = tokenizer_json.get("model", {})
    if model.get("type") != "BPE":
        raise AssertionError(f"model.type must be 'BPE', got {model.get('type')!r}")

    decoder = tokenizer_json.get("decoder", {})
    if decoder.get("type") != "ByteLevel":
        raise AssertionError(f"decoder.type must be 'ByteLevel', got {decoder.get('type')!r}")

    pre = tokenizer_json.get("pre_tokenizer", {})
    if pre.get("type") != "ByteLevel":
        raise AssertionError(f"pre_tokenizer.type must be 'ByteLevel', got {pre.get('type')!r}")

    # Every PUA char in the mapping must have a vocab id, either as a regular
    # vocab entry or as an added token.
    added_token_strs = {t.get("content") for t in tokenizer_json.get("added_tokens", [])}
    vocab = model.get("vocab", {})
    missing: list[str] = []
    for ch in mapping.pua_chars:
        if ch in added_token_strs:
            continue
        if ch in vocab:
            continue
        missing.append(ch)
    if missing:
        raise AssertionError(
            f"{len(missing)} PUA char(s) from mapping lack a vocab id "
            f"(first 5: {[hex(ord(c)) for c in missing[:5]]})"
        )


def filter_pua_pua_merges(tokenizer_json: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Remove every PUA-PUA merge from the tokenizer JSON. Returns
    `(new_json, n_removed)`. The vocab is left untouched — orphan composed
    PUA-PUA strings remain as inert vocab entries that BPE will simply
    never produce since the merge sequence no longer reaches them.

    This is intentionally conservative: deleting vocab entries can shift
    ids and break downstream model embeddings. Leaving inert entries is
    safe for the loader and harmless for behavior.
    """
    offending = find_pua_pua_merges(tokenizer_json)
    if not offending:
        return tokenizer_json, 0

    bad_indices = {i for i, _, _ in offending}
    merges = tokenizer_json.get("model", {}).get("merges", [])
    new_merges = [m for i, m in enumerate(merges) if i not in bad_indices]
    new_json = dict(tokenizer_json)
    new_model = dict(new_json["model"])
    new_model["merges"] = new_merges
    new_json["model"] = new_model
    return new_json, len(offending)


def audit_and_filter_tokenizer_file(
    tokenizer_path: Path,
    mapping: PUAMapping,
    *,
    strict: bool = True,
) -> dict[str, int]:
    """Read `tokenizer.json`, optionally filter PUA-PUA merges, re-save.

    Returns a small stats dict for the build manifest:
        {"pua_pua_merges_found": N, "pua_pua_merges_removed": N | 0}
    """
    tokenizer_json: dict[str, Any] = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    found = find_pua_pua_merges(tokenizer_json)
    removed = 0
    if strict and found:
        tokenizer_json, removed = filter_pua_pua_merges(tokenizer_json)
        assert_invariants(tokenizer_json, mapping)
        # Atomic write: write to temp then rename.
        tmp = tokenizer_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(tokenizer_json, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(tokenizer_path)
    else:
        # Always assert invariants even if we didn't rewrite.
        assert_invariants(tokenizer_json, mapping)
    return {"pua_pua_merges_found": len(found), "pua_pua_merges_removed": removed}


__all__ = [
    "assert_invariants",
    "audit_and_filter_tokenizer_file",
    "filter_pua_pua_merges",
    "find_pua_pua_merges",
]
