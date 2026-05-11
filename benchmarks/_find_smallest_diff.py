"""Find the smallest-diff parity failure and show surrounding context."""
from __future__ import annotations
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cute_tokenizer import CUTETokenizerFast
from cute_tokenizer._accel import BPEEncoder

bpe = BPEEncoder("model/tokenizer.json", "model/cute_mapping.json")
ctok = CUTETokenizerFast(
    tokenizer_file="model/tokenizer.json", cute_mapping_file="model/cute_mapping.json"
)

HOLDOUT = Path(r"C:\Users\husse\Downloads\CUTE\holdout\python")
files = sorted(HOLDOUT.rglob("*.py"))
rng = random.Random(42)
rng.shuffle(files)
files = files[:200]

# Find the file with the smallest non-zero differing count
best: tuple[Path, list[int], list[int], list[int]] | None = None
for path in files:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    cb = list(bpe.encode(text))
    ref = list(ctok.fast_encode(text))
    if cb == ref:
        continue
    differs = [i for i in range(min(len(cb), len(ref))) if cb[i] != ref[i]]
    if differs and (best is None or len(differs) < len(best[3])):
        best = (path, cb, ref, differs)

if best is None:
    print("no divergences found"); sys.exit(0)

path, cb, ref, differs = best
print(f"file: {path.name}")
print(f"cute: {len(cb)} tokens, ref: {len(ref)} tokens, {len(differs)} differing positions")
print()

# Inspect the first divergence
first = differs[0]
print(f"First divergence at index {first}:")
print(f"  cute[{first}] = {cb[first]}")
print(f"  ref [{first}] = {ref[first]}")

# Get the byte position in the input text by walking through with fast_encode
text = path.read_text(encoding="utf-8")
# Try to reconstruct: decode the first `first` tokens to find where it goes wrong
prefix_text = ctok.decode(ref[:first], skip_special_tokens=True)
suffix_text = ctok.decode(ref[first:first+10], skip_special_tokens=True)
print()
print(f"context just before divergence (decoded ref[:{first}]):")
safe = prefix_text[-80:].encode("ascii", "replace").decode("ascii")
print(f"  ...{safe!r}")
print(f"context AT divergence (decoded ref[{first}:{first+10}]):")
safe = suffix_text.encode("ascii", "replace").decode("ascii")
print(f"  {safe!r}")
print()
print(f"around the divergence:")
for i in range(max(0, first-3), min(len(cb), len(ref), first+5)):
    flag = "  " if cb[i] == ref[i] else "<>"
    print(f"  {flag} [{i:>4}]  cute={cb[i]:>7}  ref={ref[i]:>7}")
