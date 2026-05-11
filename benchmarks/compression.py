"""Compression benchmark: CUTE vs baseline tokenizers.

Reports:
  - mean / p50 / p95 / p99 token count per file
  - mean bytes-per-token
  - aggregate compression ratio (CUTE / baseline)

Usage:
    python -m benchmarks.compression \
        --tokenizer ./output \
        --holdout ./holdout

The holdout corpus should be code that was NOT in the training set.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

from cute_tokenizer import CUTETokenizerFast
from cute_tokenizer.corpus import iter_corpus_files


class TokStats(NamedTuple):
    name: str
    mean: float
    p50: float
    p95: float
    p99: float
    bytes_per_token: float
    files_evaluated: int


def _percentile(vals: list[int], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return float(s[k])


def evaluate(
    name: str,
    encode_fn: Callable[[str], list[int]],
    files: Iterable[Path],
) -> TokStats:
    counts: list[int] = []
    total_bytes = 0
    total_tokens = 0
    n = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        ids = encode_fn(text)
        counts.append(len(ids))
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(ids)
        n += 1
    if not counts:
        return TokStats(name, 0, 0, 0, 0, 0, 0)
    return TokStats(
        name=name,
        mean=statistics.mean(counts),
        p50=_percentile(counts, 50),
        p95=_percentile(counts, 95),
        p99=_percentile(counts, 99),
        bytes_per_token=total_bytes / max(1, total_tokens),
        files_evaluated=n,
    )


def _build_tokenizers(tokenizer_dir: Path) -> dict[str, Callable[[str], list[int]]]:
    """Return { name: encode_fn } for CUTE plus available baselines."""
    encoders: dict[str, Callable[[str], list[int]]] = {}

    cute = CUTETokenizerFast(
        tokenizer_file=tokenizer_dir / "tokenizer.json",
        cute_mapping_file=tokenizer_dir / "cute_mapping.json",
    )
    encoders["CUTE"] = lambda t: cute(t, add_special_tokens=False).input_ids

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        encoders["tiktoken/cl100k"] = enc.encode
    except Exception as e:
        print(f"  (skipping tiktoken: {e})")

    try:
        from transformers import AutoTokenizer
        gpt2 = AutoTokenizer.from_pretrained("gpt2")
        encoders["GPT-2 (HF)"] = lambda t: gpt2(t, add_special_tokens=False)["input_ids"]
    except Exception as e:
        print(f"  (skipping GPT-2: {e})")

    return encoders


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="CUTE tokenizer dir")
    ap.add_argument("--holdout", required=True, help="Held-out corpus dir")
    ap.add_argument("--max-files", type=int, default=2_000)
    args = ap.parse_args()

    tokenizer_dir = Path(args.tokenizer)
    holdout = Path(args.holdout)

    files = list(iter_corpus_files(
        holdout,
        extensions=(".py", ".js", ".ts", ".java", ".c", ".cpp", ".rs", ".go", ".rb", ".php"),
    ))[: args.max_files]
    if not files:
        print(f"No files found under {holdout}")
        return 1

    encoders = _build_tokenizers(tokenizer_dir)

    print(f"\nEvaluating {len(files)} files from {holdout}\n")
    results: list[TokStats] = []
    for name, enc in encoders.items():
        print(f"  -> {name} ...", flush=True)
        results.append(evaluate(name, enc, files))

    # Header
    print(f"\n{'Tokenizer':<20} {'mean':>10} {'p50':>10} {'p95':>10} {'p99':>10} {'B/tok':>8}")
    print("-" * 72)
    cute_mean = next(r.mean for r in results if r.name == "CUTE")
    for r in results:
        ratio = r.mean / cute_mean if cute_mean else 1.0
        marker = "" if r.name == "CUTE" else f"  ({ratio:.2f}x CUTE)"
        print(
            f"{r.name:<20} {r.mean:>10.1f} {r.p50:>10.1f} {r.p95:>10.1f} "
            f"{r.p99:>10.1f} {r.bytes_per_token:>8.2f}{marker}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
