"""Latency microbenchmark.

Reports encode/decode time per 1KB of code text. CUTE pre-tokenization runs
in Python (Aho-Corasick) so we expect ~1.5-2x slower than tiktoken; this
benchmark is meant to keep that overhead in a known range.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from cute_tokenizer import CUTETokenizerFast


SAMPLE_TEXT = """
def calculate_total(items, tax_rate=0.1):
    \"\"\"Compute the total price including tax.\"\"\"
    if not items:
        return 0.0
    subtotal = sum(item.price * item.quantity for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax


class ShoppingCart:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.items = []

    def add_item(self, item):
        self.items.append(item)
        return self

    def remove_item(self, item_id):
        self.items = [i for i in self.items if i.id != item_id]
        return self

    def total(self):
        return calculate_total(self.items)
""" * 50  # ~5KB of text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--iterations", type=int, default=100)
    args = ap.parse_args()

    tokenizer_dir = Path(args.tokenizer)
    tok = CUTETokenizerFast(
        tokenizer_file=tokenizer_dir / "tokenizer.json",
        cute_mapping_file=tokenizer_dir / "cute_mapping.json",
    )

    text = SAMPLE_TEXT
    text_kb = len(text.encode("utf-8")) / 1024

    # Warm up
    for _ in range(5):
        tok(text)

    # Encode
    encode_times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        out = tok(text, add_special_tokens=False)
        encode_times.append(time.perf_counter() - t0)
    ids = out.input_ids

    # Decode
    decode_times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        tok.decode(ids, skip_special_tokens=True)
        decode_times.append(time.perf_counter() - t0)

    print(f"\nText size: {text_kb:.1f} KB ({len(ids)} tokens)\n")
    print(f"Encode: {statistics.mean(encode_times)*1000:.2f} ms ± "
          f"{statistics.stdev(encode_times)*1000:.2f} (per call) ; "
          f"{statistics.mean(encode_times)*1000/text_kb:.2f} ms/KB")
    print(f"Decode: {statistics.mean(decode_times)*1000:.2f} ms ± "
          f"{statistics.stdev(decode_times)*1000:.2f} (per call) ; "
          f"{statistics.mean(decode_times)*1000/text_kb:.2f} ms/KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
