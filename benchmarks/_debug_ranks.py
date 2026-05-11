"""Look up vocab IDs for key merge candidates to understand HF's tie-breaking."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokenizers import Tokenizer

hf = Tokenizer.from_file("model/tokenizer.json")
vocab = hf.get_vocab()

# Look up specific merge results.
candidates = [
    "Ġ",      # single space byte-level
    "ĠĠ",     # 2 spaces
    "ĠĠĠ",   # 3 spaces
    "ĠĠĠĠ",  # 4 spaces
    "Ġp",     # space+p
    "p",
    "pa",
    "pass",
    "Ġpass",
    "Ġpa",
    "ĠĠp",   # ?
    "ĠĠĠp",  # ?
]
for c in candidates:
    rid = vocab.get(c)
    safe = repr(c).encode("ascii", "replace").decode("ascii")
    print(f"  {safe:15}  id = {rid}")
