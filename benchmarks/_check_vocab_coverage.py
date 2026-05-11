"""Check whether CUTE's current vocab already contains common cl100k patterns
(like ` self`, ` def`, indent runs) — and if not, why BPE didn't learn them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tiktoken
from tokenizers import Tokenizer

CHECK = [
    "    ", "        ", "            ", "                ",
    "\n    ", "\n        ", "\n\n",
    " self", "(self", " self.", "self.", " return", " def",
    " for", " if", " in", " import", " from", " not", " is",
    " None", " True", " False", " and", " or",
    "):", " ->", " ==",
    ".get", ".set", ".admin", ".append", ".items", ".keys",
    "__init__", "__main__", "__name__",
]


def main() -> int:
    cute = Tokenizer.from_file("model/tokenizer.json")
    cl = tiktoken.get_encoding("cl100k_base")
    vocab = cute.get_vocab()  # str -> id

    # cl100k uses GPT-2 byte-level encoding: " " -> "Ġ", "\n" -> "Ċ", "\t" -> "ĉ".
    # CUTE also uses ByteLevel pre-tokenizer so its vocab is in the same encoding.
    SPACE = "Ġ"  # Ġ
    NEWLINE = "Ċ"  # Ċ
    TAB = "ĉ"  # ĉ
    def to_bytelevel(s: str) -> str:
        return s.replace(" ", SPACE).replace("\n", NEWLINE).replace("\t", TAB)

    print(f"CUTE vocab size: {len(vocab):,}")
    print(f"{'pattern':40} {'CUTE? IDs':12} {'cl100k IDs'}")
    print("-" * 78)
    for pat in CHECK:
        bl = to_bytelevel(pat)
        # Try direct lookup in vocab
        if bl in vocab:
            cute_repr = f"1 (id={vocab[bl]})"
        else:
            # Encode via the tokenizer to see how many tokens it really uses
            enc = cute.encode(pat, add_special_tokens=False)
            cute_repr = f"{len(enc.ids)} ({enc.tokens})"
        cl_ids = cl.encode(pat, disallowed_special=())
        cl_repr = f"{len(cl_ids)}"
        rendered = pat.replace(" ", "_").replace("\n", "\\n").replace("\t", "\\t")
        cute_safe = cute_repr.encode("ascii", "replace").decode("ascii")
        print(f"{rendered!r:40} {cute_safe:50} {cl_repr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
