"""Compare cute-bpe vs HF Tokenizer vs tiktoken on encode latency."""
from __future__ import annotations
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tiktoken
from cute_tokenizer import CUTETokenizerFast
from cute_tokenizer._accel import BPEEncoder
from tokenizers import Tokenizer

bpe = BPEEncoder("model/tokenizer.json", "model/cute_mapping.json")
ctok = CUTETokenizerFast(
    tokenizer_file="model/tokenizer.json", cute_mapping_file="model/cute_mapping.json"
)
hf = Tokenizer.from_file("model/tokenizer.json")
cl = tiktoken.get_encoding("cl100k_base")

text = Path("rust/cute-core/benches/sample_code.txt").read_text(encoding="utf-8")
nb = len(text.encode("utf-8"))
print(f"sample: {nb} bytes")
print()


def time_fn(fn, n=300):
    for _ in range(10):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
    s = sorted(samples)
    return {
        "p50": s[len(s) // 2],
        "p95": s[int(len(s) * 0.95)],
        "p99": s[int(len(s) * 0.99)],
        "mean": statistics.mean(samples),
    }


cases = [
    ("CUTE __call__ (full HF wrapper)", lambda: ctok(text, add_special_tokens=False).input_ids),
    ("CUTE fast_encode (HF BPE)      ", lambda: ctok.fast_encode(text)),
    ("cute-bpe (new Rust BPE)        ", lambda: bpe.encode(text)),
    ("tiktoken cl100k                ", lambda: cl.encode(text, disallowed_special=())),
]
print(f"{'name':38}  {'p50 µs':>9}  {'p95 µs':>9}  {'p99 µs':>9}  {'mean µs':>9}")
print("-" * 80)
for name, fn in cases:
    s = time_fn(fn)
    print(f"{name}  {s['p50']:>9,.0f}  {s['p95']:>9,.0f}  {s['p99']:>9,.0f}  {s['mean']:>9,.0f}")
