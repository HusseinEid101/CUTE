"""For every cl100k token used in the holdout, compute the savings
of making it a single CUTE token: ``frequency * (cute_count - 1)``.

This is a cleaner version of ``_analyze_loss.py`` — it ranks ALL
candidate strings (not just multi-pattern) by their corpus-wide token
savings vs the current CUTE tokenizer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tiktoken
from tokenizers import Tokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--cute-dir", default="model")
    ap.add_argument("--max-files", type=int, default=1500)
    ap.add_argument("--out", default="reports/cute-savings-analysis.json")
    args = ap.parse_args()

    print("loading tokenizers...", flush=True)
    cute = Tokenizer.from_file(str(Path(args.cute_dir) / "tokenizer.json"))
    cl = tiktoken.get_encoding("cl100k_base")

    holdout = Path(args.holdout)
    files = sorted(p for p in holdout.rglob("*") if p.is_file() and p.suffix == ".py")[
        : args.max_files
    ]
    print(f"analyzing {len(files)} files", flush=True)

    cl_token_counts: Counter[int] = Counter()
    total_text_bytes = 0

    # Pass 1: count cl100k token id frequencies.
    for i, f in enumerate(files):
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        cl_ids = cl.encode(text, disallowed_special=())
        for tid in cl_ids:
            cl_token_counts[tid] += 1
        total_text_bytes += len(text.encode("utf-8"))
        if (i + 1) % 200 == 0:
            print(f"  pass1 {i + 1}/{len(files)}", flush=True)

    print(f"distinct cl100k token ids: {len(cl_token_counts):,}", flush=True)

    # Pass 2: for each cl100k token id, decode it and compute CUTE cost.
    candidates: list[tuple[float, str, int, int]] = []  # savings, string, freq, cute_cost
    skipped_partial_utf8 = 0
    skipped_no_savings = 0
    for tid, freq in cl_token_counts.items():
        try:
            s = cl.decode([tid])
        except Exception:
            continue
        if not s or len(s) < 1:
            continue
        # Reject tokens that decode to a partial UTF-8 codepoint (U+FFFD).
        # cl100k often splits a non-ASCII codepoint across multiple byte
        # tokens; decoding ONE byte gives U+FFFD which would corrupt input
        # on encode if added to CUTE's vocab.
        if "�" in s:
            skipped_partial_utf8 += 1
            continue
        # Count CUTE token cost for this string.
        cute_ids = cute.encode(s, add_special_tokens=False).ids
        cute_cost = len(cute_ids)
        if cute_cost <= 1:
            skipped_no_savings += 1
            continue
        savings = freq * (cute_cost - 1)
        candidates.append((savings, s, freq, cute_cost))

    print(f"skipped (partial-utf8 / U+FFFD): {skipped_partial_utf8:,}")
    print(f"skipped (no savings): {skipped_no_savings:,}")

    candidates.sort(key=lambda x: -x[0])

    print(f"\ntotal candidates with savings > 0: {len(candidates):,}")
    total_savings = sum(c[0] for c in candidates)
    print(f"total potential savings: {total_savings:,} tokens")

    # Render top 30 (ASCII-safe).
    print(f"\n{'savings':>10}  {'freq':>8}  {'cute':>4}  string")
    print("-" * 78)
    for sav, s, fr, cc in candidates[:30]:
        rendered = (
            s.replace(" ", "_")
             .replace("\n", "\\n")
             .replace("\t", "\\t")
             .encode("ascii", "replace")
             .decode("ascii")
        )
        print(f"{sav:>10,}  {fr:>8,}  {cc:>4}  {rendered!r}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "files_analyzed": len(files),
                "total_potential_savings": int(total_savings),
                "candidates": [
                    {"string": s, "freq": fr, "cute_cost": cc, "savings": sav}
                    for sav, s, fr, cc in candidates[:30000]
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path} ({len(candidates)} total, top 30000 saved).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
