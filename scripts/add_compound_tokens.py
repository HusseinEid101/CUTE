"""Build a v6+ model by adding compound tokens cl100k captures but
CUTE's GPT-2 ByteLevel regex splits.

Source: a cl100k loss-analysis JSON (see benchmarks/_analyze_loss.py).
We add the missing patterns as AddedTokens which match atomically before
the regex pre-tokenizer runs.

Usage::

    python scripts/add_compound_tokens.py \
        --source model --target model_v7 \
        --analysis reports/cute-loss-analysis-deep.json \
        --top 20000 --max-add 10000 --min-count 3
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


def is_useful_compound(s: str) -> bool:
    """Patterns CUTE's GPT-2 ByteLevel regex splits, but cl100k merges.

    Includes:
    - Punct-led + word: ".get", "(self", "='".
    - Word + punct (no leading whitespace): "self)", "value,".
    - Mixed kinds inside: "n=", "x=1".
    - Pure punct runs of length >= 2 not in CUTE vocab: "):", "}}".

    Excludes:
    - Whitespace-led patterns (BPE already merges these).
    - Single-char tokens.
    - Pure word/digit runs.
    """
    if not s or len(s) < 2:
        return False
    if s.isspace():
        return False
    # Whitespace-led — BPE already merges via " ?\\p{L}+".
    if s[0] in " \t\n":
        return False
    has_word = False
    has_punct = False
    for c in s:
        if c.isalnum() or c == "_":
            has_word = True
        elif not c.isspace():
            has_punct = True
        if has_word and has_punct:
            return True
    return has_punct and not has_word and len(s) >= 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="model")
    ap.add_argument("--target", default="model_v6")
    ap.add_argument("--analysis", default="reports/cute-loss-analysis.json")
    ap.add_argument("--top", type=int, default=20000)
    ap.add_argument("--max-add", type=int, default=10000)
    ap.add_argument("--min-count", type=int, default=3)
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
    patterns = analysis["top_multi_patterns"]
    print(f"Patterns in analysis: {len(patterns):,}")

    candidates: list[tuple[str, int]] = []
    skipped_known = 0
    skipped_filter = 0
    for entry in patterns[: args.top]:
        pat = entry["pattern"]
        cnt = entry["count"]
        if cnt < args.min_count:
            continue
        if not is_useful_compound(pat):
            skipped_filter += 1
            continue
        bl = to_bytelevel(pat)
        if bl in vocab:
            skipped_known += 1
            continue
        candidates.append((pat, cnt))

    candidates.sort(key=lambda kv: -kv[1])
    selected = candidates[: args.max_add]

    print(f"Already in vocab: {skipped_known}")
    print(f"Filtered out: {skipped_filter}")
    print(f"Selected to add: {len(selected)}")

    if selected[:15]:
        print("First 15 candidates:")
        for pat, cnt in selected[:15]:
            rendered = pat.replace(" ", "_").replace("\n", "\\n").replace("\t", "\\t")
            safe = rendered.encode("ascii", "replace").decode("ascii")
            print(f"  {cnt:>7,}  {safe!r}")

    added = [
        AddedToken(
            pat,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=False,
        )
        for pat, _ in selected
    ]
    n = tok.add_tokens(added)
    print(f"Added {n} tokens.")

    tok.save(str(dst / "tokenizer.json"))
    shutil.copyfile(src / "cute_mapping.json", dst / "cute_mapping.json")
    if (src / "tokenizer_config.json").is_file():
        shutil.copyfile(src / "tokenizer_config.json", dst / "tokenizer_config.json")

    new_vocab_size = len(tok.get_vocab())
    print(f"\nNew vocab size: {new_vocab_size:,} (delta {new_vocab_size - len(vocab)})")
    print(f"Saved to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
