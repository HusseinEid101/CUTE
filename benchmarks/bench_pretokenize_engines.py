"""Head-to-head Python vs Rust pre-tokenizer benchmark.

Runs the same input through both engines on the same Python interpreter
in the same process — no separate runs, no apples-to-oranges. The Python
path is reached via the explicit fallback functions; the Rust path goes
through ``_accel`` directly. This avoids the ``CUTE_USE_PYTHON_PRETOKENIZER``
env-var dance which only switches the wrapper-level dispatch.

Measures:
- ``pretokenize_to_string``: full string-form output (the encode hot path).
- ``cute_split_text``: list-of-pieces output (debug / inspection path).
- ``reverse_pua_substitute``: decode-side PUA reversal.
- ``count_in_text``: training-time frequency accumulation.

Usage::

    python benchmarks/bench_pretokenize_engines.py [--iterations N]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Ensure the worktree's src/ is importable when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cute_tokenizer._accel_loader import USE_RUST, accel  # noqa: E402
from cute_tokenizer.pretokenizer import _python_cute_split_text  # noqa: E402
from cute_tokenizer.pua import PUAMapping  # noqa: E402
from cute_tokenizer.trainer import load_mapping  # noqa: E402

SAMPLE_TEXT = (
    """
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
""".strip()
    + "\n"
) * 50


def _python_count_in_text(
    text: str, boost: float, max_len: int
) -> tuple[dict[str, int], dict[str, float]]:
    from collections import Counter, defaultdict

    from cute_tokenizer.patterns import iter_tokens, is_identifier, split_identifier

    counter: Counter[str] = Counter()
    boost_acc: defaultdict[str, float] = defaultdict(float)
    for tok, _, _ in iter_tokens(text):
        if len(tok) > max_len:
            continue
        counter[tok] += 1
        if boost > 0 and is_identifier(tok):
            for part in split_identifier(tok):
                if part != tok:
                    boost_acc[part] += boost
    return dict(counter), dict(boost_acc)


def _bench(label: str, fn, iterations: int) -> tuple[float, float]:
    # Warm-up.
    for _ in range(min(5, iterations // 10 or 1)):
        fn()
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, stdev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--mapping", type=str, default="model/cute_mapping.json")
    args = ap.parse_args()

    if not USE_RUST:
        print("ERROR: Rust _accel extension is not loaded.", file=sys.stderr)
        print(
            "Run `maturin develop --release` from the repo root and ensure "
            "CUTE_USE_PYTHON_PRETOKENIZER is unset.",
            file=sys.stderr,
        )
        return 1

    mapping_path = Path(args.mapping)
    if mapping_path.is_file():
        mapping: PUAMapping = load_mapping(mapping_path)
        print(f"Loaded PUAMapping with {mapping.size} entries from {mapping_path}.")
    else:
        mapping = PUAMapping(word_to_pua={}, pua_to_word={}, skipped_codepoints=())
        print("WARNING: no mapping at path, using empty mapping.")

    prepared = accel.PreparedMapping(mapping.word_to_pua, mapping.pua_to_word)
    text_kb = len(SAMPLE_TEXT.encode("utf-8")) / 1024
    print(f"Sample text: {text_kb:.1f} KB.")
    print(f"Iterations: {args.iterations}.\n")

    def _py_pretokenize_to_string() -> str:
        return "".join(_python_cute_split_text(SAMPLE_TEXT, mapping))

    cases: list[tuple[str, callable, callable]] = [
        (
            "pretokenize_to_string",
            _py_pretokenize_to_string,
            lambda: accel.pretokenize_to_string(SAMPLE_TEXT, prepared),
        ),
        (
            "cute_split_text",
            lambda: _python_cute_split_text(SAMPLE_TEXT, mapping),
            lambda: accel.cute_split_text(SAMPLE_TEXT, prepared),
        ),
    ]

    # Generate a PUA-rich body for the decode bench.
    pre = accel.pretokenize_to_string(SAMPLE_TEXT, prepared)

    def _py_reverse() -> str:
        from cute_tokenizer.pua import is_pua_char as py_is_pua

        out: list[str] = []
        ptw = mapping.pua_to_word
        if not ptw:
            return pre
        for ch in pre:
            if py_is_pua(ch):
                out.append(ptw.get(ch, ch))
            else:
                out.append(ch)
        return "".join(out)

    cases.append(
        (
            "reverse_pua_substitute",
            _py_reverse,
            lambda: accel.reverse_pua_substitute(pre, prepared),
        )
    )

    cases.append(
        (
            "count_in_text",
            lambda: _python_count_in_text(SAMPLE_TEXT, 0.3, 50),
            lambda: accel.count_in_text(SAMPLE_TEXT, 0.3, 50),
        )
    )

    print(f"{'Function':<28} {'Python (ms)':>14} {'Rust (ms)':>12} {'Speedup':>10}")
    print("-" * 68)
    for name, py_fn, rs_fn in cases:
        py_mean, py_std = _bench(f"py:{name}", py_fn, args.iterations)
        rs_mean, rs_std = _bench(f"rs:{name}", rs_fn, args.iterations)
        speedup = py_mean / rs_mean if rs_mean > 0 else float("inf")
        print(
            f"{name:<28} {py_mean * 1000:>10.2f} ± {py_std * 1000:>4.1f}"
            f" {rs_mean * 1000:>8.2f} ± {rs_std * 1000:>4.1f} {speedup:>9.2f}×"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
