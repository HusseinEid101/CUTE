"""Probe what happens to a PUA char through cute-bpe vs fast_encode."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cute_tokenizer import CUTETokenizerFast
from cute_tokenizer._accel import BPEEncoder
from cute_tokenizer.trainer import load_mapping
from tokenizers import Tokenizer

bpe = BPEEncoder("model/tokenizer.json", "model/cute_mapping.json")
ctok = CUTETokenizerFast(
    tokenizer_file="model/tokenizer.json", cute_mapping_file="model/cute_mapping.json"
)
m = load_mapping(Path("model/cute_mapping.json"))
hf = Tokenizer.from_file("model/tokenizer.json")
vocab = hf.get_vocab()
inv = {v: k for k, v in vocab.items()}

# Check whether "Rounds" is in the PUA mapping.
if "Rounds" in m.word_to_pua:
    print(f'"Rounds" -> PUA U+{ord(m.word_to_pua["Rounds"]):05X}')
else:
    print('"Rounds" NOT in PUA mapping')

# Encode a single PUA char directly via cute-bpe (bypass substitution).
pua_char = m.word_to_pua.get("Rounds")
if pua_char:
    cb_ids = list(bpe.encode(pua_char))
    ref_ids = ctok.fast_encode(pua_char)
    print(f"\nencode PUA char alone:")
    print(f"  cute = {cb_ids}")
    print(f"  ref  = {ref_ids}")
    print(f"\nencode 'Rounds' (will be substituted to PUA):")
    cb_ids = list(bpe.encode("Rounds"))
    ref_ids = ctok.fast_encode("Rounds")
    print(f"  cute = {cb_ids}  decoded: {[inv.get(i, '?').encode('ascii','replace').decode('ascii') for i in cb_ids]}")
    print(f"  ref  = {ref_ids}  decoded: {[inv.get(i, '?').encode('ascii','replace').decode('ascii') for i in ref_ids]}")
