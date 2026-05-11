"""Debug one specific roundtrip failure: encode a small substring, decode,
inspect what changed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cute_tokenizer import CUTETokenizerFast


def main() -> int:
    tok = CUTETokenizerFast(
        tokenizer_file="model_v10/tokenizer.json",
        cute_mapping_file="model_v10/cute_mapping.json",
    )
    samples = [
        "l'éditeur",
        "Español",
        "brasileirão",
        "Poids de l'économie",
        "español",
    ]
    for sample in samples:
        ids = tok(sample, add_special_tokens=False).input_ids
        decoded = tok.decode(ids, skip_special_tokens=True)
        ok = decoded == sample
        sample_safe = sample.encode("ascii", "replace").decode("ascii")
        decoded_safe = decoded.encode("ascii", "replace").decode("ascii")
        print(f"{'OK' if ok else 'FAIL'}: {sample_safe!r} -> {decoded_safe!r}")
        if not ok:
            print(f"  ids = {ids}")
            for i in ids:
                t = tok.convert_ids_to_tokens([i])[0]
                t_safe = t.encode("ascii", "replace").decode("ascii")
                print(f"    {i:>6}  {t_safe!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
