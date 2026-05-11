"""Debug parity failures: decode each token ID + show what's different."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cute_tokenizer._accel import BPEEncoder
from tokenizers import Tokenizer

bpe = BPEEncoder("model/tokenizer.json", "model/cute_mapping.json")
hf = Tokenizer.from_file("model/tokenizer.json")
vocab = hf.get_vocab()
inv = {v: k for k, v in vocab.items()}

text = "class Foo:\n    pass"
cb = list(bpe.encode(text))
hf_ids = hf.encode(text, add_special_tokens=False).ids

print(f"input: {text!r}")
print(f"input bytes: {text.encode('utf-8')!r}")
print()
print(f"{'pos':>3}  {'cute':>6}  {'cute_str':30} {'hf':>6}  {'hf_str'}")
for i in range(max(len(cb), len(hf_ids))):
    c = cb[i] if i < len(cb) else None
    h = hf_ids[i] if i < len(hf_ids) else None
    cs = repr(inv.get(c, "?")).encode("ascii", "replace").decode("ascii") if c is not None else ""
    hs = repr(inv.get(h, "?")).encode("ascii", "replace").decode("ascii") if h is not None else ""
    print(f"{i:>3}  {c if c else '':>6}  {cs:30} {h if h else '':>6}  {hs}")
