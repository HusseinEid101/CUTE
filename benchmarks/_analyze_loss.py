"""Where does CUTE waste tokens vs cl100k?

Encodes the same files with both tokenizers, lists cl100k tokens that
CUTE pre-tokenization can't capture (multi-word, cross-whitespace,
identifier+punctuation, indent runs). Counts cross-corpus frequency
of these patterns. The output is the candidate list of multi-token PUAs
to add to a v6 build.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tiktoken
from cute_tokenizer import CUTETokenizerFast


def is_multi_pattern(tok: str) -> bool:
    """A 'multi pattern' is a token that crosses what CUTE pre-tokenization
    would split into multiple regex tokens — i.e., it spans whitespace or
    mixes word + punctuation."""
    if not tok:
        return False
    # Indent runs are pure whitespace.
    if tok.isspace():
        return True
    # Contains both word-char and non-word/whitespace.
    has_word = any(c.isalnum() or c == "_" for c in tok)
    has_non_word = any(not (c.isalnum() or c == "_") for c in tok)
    if has_word and has_non_word:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--max-files", type=int, default=300)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--out", default="reports/cute-loss-analysis.json")
    args = ap.parse_args()

    print("loading tokenizers...", flush=True)
    cute = CUTETokenizerFast(
        tokenizer_file="model/tokenizer.json",
        cute_mapping_file="model/cute_mapping.json",
    )
    cl100k = tiktoken.get_encoding("cl100k_base")

    holdout = Path(args.holdout)
    files = sorted(p for p in holdout.rglob("*") if p.is_file() and p.suffix == ".py")[
        : args.max_files
    ]
    print(f"analyzing {len(files)} files", flush=True)

    multi_pattern_freq: Counter[str] = Counter()
    cl100k_total = 0
    cute_total = 0

    for i, f in enumerate(files):
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        cl_ids = cl100k.encode(text, disallowed_special=())
        cute_ids = cute(text, add_special_tokens=False).input_ids
        cl100k_total += len(cl_ids)
        cute_total += len(cute_ids)

        # Collect cl100k token strings that span whitespace/punctuation.
        for tid in cl_ids:
            tok = cl100k.decode([tid])
            if len(tok) < args.min_len:
                continue
            if is_multi_pattern(tok):
                multi_pattern_freq[tok] += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)

    delta = cute_total - cl100k_total
    pct = (delta / cute_total) * 100 if cute_total else 0
    print(
        f"\nTOTAL: CUTE={cute_total:,} tokens, cl100k={cl100k_total:,} tokens, "
        f"CUTE wastes {delta:,} ({pct:.2f}%) more"
    )
    print(f"Distinct multi-patterns in cl100k: {len(multi_pattern_freq):,}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "cl100k_tokens_total": cl100k_total,
                "cute_tokens_total": cute_total,
                "files_analyzed": len(files),
                "top_multi_patterns": [
                    {"pattern": pat, "count": n}
                    for pat, n in multi_pattern_freq.most_common(args.top)
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
