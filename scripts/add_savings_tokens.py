"""Add the highest-savings cl100k tokens to the existing CUTE tokenizer.

Reads ``reports/cute-savings-analysis.json`` (frequency × (cute_cost - 1)
per cl100k token string), filters minimally, and adds the top N as
AddedTokens. Designed to push CUTE past cl100k on Python compression.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tokenizers import AddedToken, Tokenizer

REPO = Path(__file__).resolve().parent.parent

_BYTE_LEVEL_TABLE = str.maketrans({" ": "Ġ", "\n": "Ċ", "\t": "ĉ"})


def to_bytelevel(s: str) -> str:
    return s.translate(_BYTE_LEVEL_TABLE)


def is_addable(s: str, *, max_len: int = 80) -> bool:
    """Reject only obviously-bad candidates.

    Rejects:
    - Empty / single-char strings (no compression possible).
    - Pure whitespace runs (BPE handles these via the whitespace arm).
    - Strings longer than max_len (extreme outliers waste vocab).
    - Strings containing non-ASCII chars or U+FFFD: HF AddedTokens
      match on byte-level representation, and partial-codepoint
      candidates corrupt input on encode (e.g. cl100k's split UTF-8).
      Conservative ASCII-only filter eliminates this class of bug.
    """
    if not s or len(s) < 2:
        return False
    if s.isspace():
        return False
    if len(s) > max_len:
        return False
    if "�" in s:
        return False
    if any(ord(c) >= 0x80 for c in s):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="model")
    ap.add_argument("--target", default="model_v9")
    ap.add_argument("--analysis", default="reports/cute-savings-analysis.json")
    ap.add_argument("--top", type=int, default=20000)
    ap.add_argument("--min-savings", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=80)
    args = ap.parse_args()

    src = REPO / args.source
    dst = REPO / args.target
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    print(f"Loading tokenizer from {src} ...")
    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    vocab = tok.get_vocab()
    print(f"Existing vocab: {len(vocab):,}")

    analysis = json.loads((REPO / args.analysis).read_text(encoding="utf-8"))
    candidates = analysis["candidates"]
    print(f"Candidates in analysis: {len(candidates):,}")
    print(f"Total potential savings: {analysis['total_potential_savings']:,} tokens")

    selected: list[tuple[str, int]] = []
    skipped_known = 0
    skipped_filter = 0
    skipped_savings = 0
    for c in candidates[: args.top]:
        s = c["string"]
        sav = c["savings"]
        if sav < args.min_savings:
            skipped_savings += 1
            continue
        if not is_addable(s, max_len=args.max_len):
            skipped_filter += 1
            continue
        bl = to_bytelevel(s)
        if bl in vocab:
            skipped_known += 1
            continue
        selected.append((s, sav))

    print(f"Skipped (already in vocab): {skipped_known}")
    print(f"Skipped (filter): {skipped_filter}")
    print(f"Skipped (low savings): {skipped_savings}")
    print(f"Selected to add: {len(selected)}")
    captured_savings = sum(sav for _, sav in selected)
    print(f"Captured savings: {captured_savings:,} tokens "
          f"({100 * captured_savings / max(1, analysis['total_potential_savings']):.1f}% "
          f"of theoretical max)")

    if selected[:15]:
        print("\nFirst 15 selected:")
        for s, sav in selected[:15]:
            r = (
                s.replace(" ", "_")
                 .replace("\n", "\\n")
                 .replace("\t", "\\t")
                 .encode("ascii", "replace")
                 .decode("ascii")
            )
            print(f"  {sav:>8,}  {r!r}")

    added = [
        AddedToken(
            s,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=False,
        )
        for s, _ in selected
    ]
    n = tok.add_tokens(added)
    print(f"\nAdded {n} tokens (HF dedup may have skipped a few).")

    tok.save(str(dst / "tokenizer.json"))
    shutil.copyfile(src / "cute_mapping.json", dst / "cute_mapping.json")
    if (src / "tokenizer_config.json").is_file():
        shutil.copyfile(src / "tokenizer_config.json", dst / "tokenizer_config.json")

    new_vocab_size = len(tok.get_vocab())
    print(f"New vocab size: {new_vocab_size:,} (delta {new_vocab_size - len(vocab)})")
    print(f"Saved to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
