"""Focused encode/decode latency benchmark with single-text and batch paths.

Measures:
- Single-text encode/decode latency (p50, p95, p99 in microseconds)
- Batch encoding throughput (texts/sec)
- Compares two CUTE models if --compare is passed.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cute_tokenizer import CUTETokenizerFast


def percentile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


def collect_files(holdout: Path, n: int) -> list[str]:
    files = sorted(p for p in holdout.rglob("*") if p.is_file() and p.suffix == ".py")[:n]
    out = []
    for f in files:
        try:
            out.append(f.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
    return out


def bench_single(tok: CUTETokenizerFast, texts: list[str]) -> dict:
    # Warm-up
    for t in texts[:5]:
        tok(t, add_special_tokens=False)

    encode_us = []
    decode_us = []
    for text in texts:
        t0 = time.perf_counter_ns()
        ids = tok(text, add_special_tokens=False).input_ids
        encode_us.append((time.perf_counter_ns() - t0) / 1000.0)

        t0 = time.perf_counter_ns()
        tok.decode(ids, skip_special_tokens=True)
        decode_us.append((time.perf_counter_ns() - t0) / 1000.0)

    return {
        "encode_p50_us": percentile(encode_us, 50),
        "encode_p95_us": percentile(encode_us, 95),
        "encode_p99_us": percentile(encode_us, 99),
        "encode_mean_us": statistics.mean(encode_us),
        "decode_p50_us": percentile(decode_us, 50),
        "decode_p95_us": percentile(decode_us, 95),
        "decode_mean_us": statistics.mean(decode_us),
        "n": len(texts),
    }


def bench_batch(tok: CUTETokenizerFast, texts: list[str], batch_size: int = 64) -> dict:
    # Warm-up
    tok(texts[: min(batch_size, len(texts))], add_special_tokens=False)

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    total_texts = sum(len(b) for b in batches)

    t0 = time.perf_counter()
    for b in batches:
        tok(b, add_special_tokens=False)
    elapsed = time.perf_counter() - t0

    return {
        "batch_size": batch_size,
        "n_texts": total_texts,
        "elapsed_s": elapsed,
        "throughput_texts_per_s": total_texts / elapsed,
        "us_per_text": elapsed * 1e6 / total_texts,
    }


def report(label: str, single: dict, batch: dict) -> str:
    return (
        f"{label}\n"
        f"  encode  p50={single['encode_p50_us']:>8,.0f} us  "
        f"p95={single['encode_p95_us']:>8,.0f} us  "
        f"p99={single['encode_p99_us']:>8,.0f} us  "
        f"mean={single['encode_mean_us']:>8,.0f}\n"
        f"  decode  p50={single['decode_p50_us']:>8,.0f} us  "
        f"p95={single['decode_p95_us']:>8,.0f} us  "
        f"mean={single['decode_mean_us']:>8,.0f}\n"
        f"  batch  {batch['n_texts']} texts in {batch['elapsed_s']:.2f}s = "
        f"{batch['throughput_texts_per_s']:>6,.0f} texts/s "
        f"({batch['us_per_text']:,.0f} us/text @ batch={batch['batch_size']})"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--compare", default=None, help="second tokenizer dir for A/B")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    print(f"loading texts from {args.holdout} ...")
    texts = collect_files(Path(args.holdout), args.n)
    print(f"  {len(texts)} texts loaded")

    print(f"\nloading tokenizer {args.tokenizer} ...")
    tok_a = CUTETokenizerFast(
        tokenizer_file=f"{args.tokenizer}/tokenizer.json",
        cute_mapping_file=f"{args.tokenizer}/cute_mapping.json",
    )
    print("running single-text bench ...")
    single_a = bench_single(tok_a, texts)
    print("running batch bench ...")
    batch_a = bench_batch(tok_a, texts, args.batch_size)
    print()
    print(report(f"=== {args.tokenizer} ===", single_a, batch_a))

    if args.compare:
        print(f"\nloading {args.compare} ...")
        tok_b = CUTETokenizerFast(
            tokenizer_file=f"{args.compare}/tokenizer.json",
            cute_mapping_file=f"{args.compare}/cute_mapping.json",
        )
        single_b = bench_single(tok_b, texts)
        batch_b = bench_batch(tok_b, texts, args.batch_size)
        print()
        print(report(f"=== {args.compare} ===", single_b, batch_b))

        # Speedup table
        print("\n=== speedup (B vs A) ===")
        s = single_a["encode_p50_us"] / single_b["encode_p50_us"]
        print(f"  encode p50: {s:.2f}x")
        s = single_a["decode_p50_us"] / single_b["decode_p50_us"]
        print(f"  decode p50: {s:.2f}x")
        s = batch_b["throughput_texts_per_s"] / batch_a["throughput_texts_per_s"]
        print(f"  batch throughput: {s:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
