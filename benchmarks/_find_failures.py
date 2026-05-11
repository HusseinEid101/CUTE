"""Find which holdout files fail roundtrip on a given CUTE model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cute_tokenizer import CUTETokenizerFast


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cute-dir", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--max-files", type=int, default=1500)
    args = ap.parse_args()

    tok = CUTETokenizerFast(
        tokenizer_file=f"{args.cute_dir}/tokenizer.json",
        cute_mapping_file=f"{args.cute_dir}/cute_mapping.json",
    )
    files = sorted(p for p in Path(args.holdout).rglob("*") if p.is_file() and p.suffix == ".py")[
        : args.max_files
    ]

    fails = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        ids = tok(text, add_special_tokens=False).input_ids
        decoded = tok.decode(ids, skip_special_tokens=True)
        if decoded != text:
            fails.append((f, text, decoded))

    print(f"Failed: {len(fails)}/{len(files)}\n")
    for f, orig, dec in fails:
        print(f"=== {f} ===")
        # Find the first difference
        for i, (a, b) in enumerate(zip(orig, dec)):
            if a != b:
                ctx_start = max(0, i - 30)
                ctx_end = min(len(orig), i + 30)
                def safe(s: str) -> str:
                    return (
                        s.replace("\n", "\\n")
                         .replace("\t", "\\t")
                         .encode("ascii", "replace")
                         .decode("ascii")
                    )
                ctx_o = safe(orig[ctx_start:ctx_end])
                dec_end = min(len(dec), i + 30)
                ctx_d = safe(dec[ctx_start:dec_end])
                # Show the actual differing chars by codepoint.
                a_cp = f"U+{ord(a):04X}"
                b_cp = f"U+{ord(b):04X}" if i < len(dec) else "EOF"
                print(f"  first diff at char {i}: orig={ctx_o!r}")
                print(f"                          dec ={ctx_d!r}")
                print(f"  diff char: orig {a_cp} ({a!r}) vs dec {b_cp}")
                break
        else:
            if len(orig) != len(dec):
                print(f"  length diff: orig={len(orig)} dec={len(dec)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
