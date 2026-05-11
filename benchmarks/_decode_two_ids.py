"""Decode the two divergent IDs to understand what's in each."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokenizers import Tokenizer

hf = Tokenizer.from_file("model/tokenizer.json")
vocab = hf.get_vocab()
inv = {v: k for k, v in vocab.items()}

for tid in [41651, 192736]:
    text = inv.get(tid, "<missing>")
    bytes_repr = text.encode("utf-8")
    print(f"id {tid}:")
    print(f"  vocab key string (byte-level): {text!r}")
    print(f"  UTF-8 bytes:                   {bytes_repr!r}")
    print(f"  codepoints:                    {[f'U+{ord(c):04X}' for c in text]}")
    decoded = hf.decode([tid], skip_special_tokens=False)
    print(f"  HF decode → str:               {decoded!r} (len {len(decoded)})")
    print()
