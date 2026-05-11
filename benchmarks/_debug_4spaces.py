"""Probe what each tokenizer does on tricky boundary inputs."""
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

probes = [
    "    ",
    "    pass",
    "   pass",
    "  pass",
    " pass",
    "pass",
    "    p",
    "    pa",
    "    pas",
]
for s in probes:
    cb = list(bpe.encode(s))
    h = hf.encode(s, add_special_tokens=False).ids
    cb_s = [inv.get(i, "?").encode("ascii", "replace").decode("ascii") for i in cb]
    h_s = [inv.get(i, "?").encode("ascii", "replace").decode("ascii") for i in h]
    flag = "OK  " if list(cb) == list(h) else "DIFF"
    s_safe = repr(s).encode("ascii", "replace").decode("ascii")
    print(f"{flag}  {s_safe:14}  cute={cb} {cb_s}")
    print(f"      {' ':14}  hf  ={list(h)} {h_s}")
    print()
