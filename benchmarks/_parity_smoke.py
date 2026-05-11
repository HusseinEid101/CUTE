"""Quick parity smoke-check: cute-bpe vs HF Tokenizer on a few inputs."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cute_tokenizer._accel import BPEEncoder
from tokenizers import Tokenizer

bpe = BPEEncoder("model/tokenizer.json", "model/cute_mapping.json")
hf = Tokenizer.from_file("model/tokenizer.json")

print(f"vocab_size: {bpe.vocab_size()}")
print()

samples = [
    "def hello(): return 42",
    "class Foo:\n    pass",
    "from typing import List, Dict",
    "if x == y: return None",
    "self.cache[key] = value",
    "import numpy as np",
    "for i in range(10):\n    print(i)",
    "x: int = 0",
]
ok = 0
for s in samples:
    cb = bpe.encode(s)
    hf_ids = hf.encode(s, add_special_tokens=False).ids
    match = list(cb) == hf_ids
    ok += match
    s_safe = repr(s.replace("\n", "\\n")).encode("ascii", "replace").decode("ascii")
    flag = "OK  " if match else "FAIL"
    print(f"  [{flag}] {s_safe:50}  cute={len(cb):3}  hf={len(hf_ids):3}")
    if not match:
        print(f"           cute: {list(cb)[:20]}")
        print(f"           hf:   {hf_ids[:20]}")

print()
print(f"Parity: {ok}/{len(samples)}")
sys.exit(0 if ok == len(samples) else 1)
