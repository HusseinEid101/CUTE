#!/usr/bin/env python3
"""Find files where the CUTE tokenizer fails byte-exact roundtrip.

We promise byte-equal lossless decode(encode(text)) == text. Any file
that doesn't satisfy that is a correctness bug. This script enumerates
the failures so we can fix them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cute_tokenizer import CUTETokenizerFast


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-files", type=int, default=2000)
    ap.add_argument("--show-first-diffs", type=int, default=5,
                    help="Show first-diff context for the first N failures")
    args = ap.parse_args()

    tok_dir = Path(args.tokenizer)
    tok = CUTETokenizerFast(
        tokenizer_file=tok_dir / "tokenizer.json",
        cute_mapping_file=tok_dir / "cute_mapping.json",
    )

    failures: list[tuple[Path, str, str]] = []
    n_checked = 0
    extensions = (".py", ".js", ".ts", ".rs", ".go", ".rb", ".java")
    for p in sorted(Path(args.corpus).rglob("*")):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue
        if n_checked >= args.max_files:
            break
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        n_checked += 1
        ids = tok(text, add_special_tokens=False).input_ids
        decoded = tok.decode(ids, skip_special_tokens=True)
        if decoded != text:
            failures.append((p, text, decoded))

    print(f"checked {n_checked:,} files; {len(failures)} failed byte-exact roundtrip")

    for i, (path, expected, got) in enumerate(failures[: args.show_first_diffs]):
        print(f"\n--- failure {i + 1}: {path} ---")
        print(f"  len(expected) = {len(expected):,}, len(got) = {len(got):,}")
        # Find first diff
        first_diff = None
        for j, (e, g) in enumerate(zip(expected, got, strict=False)):
            if e != g:
                first_diff = j
                break
        if first_diff is None:
            first_diff = min(len(expected), len(got))
        ctx_start = max(0, first_diff - 30)
        ctx_end = first_diff + 30
        print(f"  first diff @ char {first_diff}")
        print(f"    expected[{ctx_start}:{ctx_end}] = {expected[ctx_start:ctx_end]!r}")
        print(f"    got     [{ctx_start}:{ctx_end}] = {got[ctx_start:ctx_end]!r}")
        # Show codepoints around the diff
        if first_diff < len(expected):
            print(f"    expected[{first_diff}] = U+{ord(expected[first_diff]):04X}")
        if first_diff < len(got):
            print(f"    got     [{first_diff}] = U+{ord(got[first_diff]):04X}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
