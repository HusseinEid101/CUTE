#!/usr/bin/env python3
"""Push the bundled CUTE tokenizer + wrapper code to HuggingFace Hub.

Creates/updates the `HusseinEid/cute-tokenizer` repo with:
- model/{tokenizer,cute_mapping,tokenizer_config,build_manifest}.json
- the wrapper modules needed for `trust_remote_code=True`
- a model-card README.md with metadata + usage + benchmarks

After the upload, the script calls `super_squash_history` so the repo
shows a single revision matching the GitHub V1.0 release.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


HF_REPO = "HusseinEid/cute-tokenizer"


MODEL_CARD = """---
license: mit
library_name: tokenizers
tags:
- code
- tokenizer
- byte-level-bpe
- private-use-area
- lossless-roundtrip
- the-stack
language:
- code
---

# CUTE

**Compact Unicode Token Encoding via Semantic-Anchored Byte-level BPE**

CUTE is a code-aware tokenizer built on a single architectural idea:
substitute high-savings multi-byte patterns to atomic Unicode codepoints
*before* byte-level BPE sees them. On 1,500 held-out Python files from
The Stack, CUTE produces fewer tokens per file than nine widely-used
baselines — including OpenAI's `cl100k_base` and `o200k_base`, LLaMA-3's
SentencePiece BPE, and three SentencePiece Unigram variants — and is
the only tokenizer in this comparison that re-encodes every file to
byte-identical source.

## Compression (1,500 held-out Python files, The Stack)

| Tokenizer                            | mean tok | bytes/tok | vs CUTE | roundtrip   |
|--------------------------------------|---------:|----------:|--------:|-------------|
| **CUTE**                             |    1,767 |      4.42 |       — | 1500 / 1500 |
| OpenAI cl100k_base                   |    1,874 |      4.17 |   +6.0% | 1500 / 1500 |
| OpenAI o200k_base                    |    1,886 |      4.14 |   +6.7% | 1500 / 1500 |
| LLaMA-3 (SentencePiece BPE)          |    1,872 |      4.17 |   +5.9% |  686 / 1500 |
| StarCoder2                           |    2,210 |      3.53 |  +25.1% |  685 / 1500 |
| XLM-RoBERTa (SentencePiece Unigram)  |    2,438 |      3.20 |  +38.0% |    0 / 1500 |
| CodeLlama                            |    2,573 |      3.03 |  +45.6% | 1493 / 1500 |
| T5 (SentencePiece Unigram)           |    2,706 |      2.89 |  +53.2% |    0 / 1500 |
| GPT-2                                |    3,581 |      2.18 | +102.7% | 1500 / 1500 |

`vs CUTE` is the extra cost the baseline pays per file. LLM API spend
is linear in this number.

## Latency (p50, 1.7 KB Python sample)

| Tokenizer                            | encode p50 | decode p50 |
|--------------------------------------|-----------:|-----------:|
| OpenAI cl100k_base                   |     552 µs |      56 µs |
| OpenAI o200k_base                    |     746 µs |      63 µs |
| LLaMA-3 (SentencePiece BPE)          |   1,427 µs |     326 µs |
| StarCoder2                           |   1,461 µs |     258 µs |
| **CUTE**                             | **1,526 µs** | **146 µs** |
| T5 (SentencePiece Unigram)           |   1,803 µs |     273 µs |
| XLM-RoBERTa (SentencePiece Unigram)  |   1,988 µs |     262 µs |
| GPT-2                                |   2,043 µs |     396 µs |
| CodeLlama                            |   5,120 µs |   2,417 µs |

Decode is **third-fastest** in the field (behind only OpenAI's cl100k
and o200k). Encode is competitive with open-source code tokenizers
(within ~7 % of LLaMA-3 / StarCoder2) but **does not beat tiktoken's
`cl100k_base`** — end-to-end encode is ~2.8 × slower. The `cute-bpe`
core encoder runs in ~259 µs; the rest is the PUA pre-substitution
Aho-Corasick pass plus the Python FFI boundary. If your bottleneck is
encoder throughput on short prompts, `cl100k_base` is the better
choice; if it's context-window budget or roundtrip safety on code,
CUTE wins.

## How it works

1. A frequency-weighted, savings-ranked selection pass mines
   high-value multi-byte patterns (identifiers, common slices like
   `(self`, `=None`, `:\\n`) from a code corpus.
2. Selected patterns are mapped one-to-one to **supplementary-plane
   Private-Use-Area (PUA) codepoints** (`U+F0000+`). The BMP-PUA range
   is deliberately skipped to avoid colliding with literal PUA
   characters that appear in real source code.
3. A byte-level BPE trainer runs on the **PUA-pre-substituted stream**,
   so semantic anchors are visible to the merge algorithm and can
   compose freely with whitespace and punctuation (e.g. `Ġ + ⟦def⟧`).
4. A second savings pass adds the top-6,000 high-frequency compound
   patterns as atomic `AddedToken`s.
5. At encode time, an Aho-Corasick (leftmost-longest) Rust pass
   substitutes PUA codepoints; a purpose-built Rust BPE encoder
   (`cute-bpe`, modeled on tiktoken's linear-scan-min-rank merge loop)
   then performs the byte-level BPE pass.
6. At decode time, the inverse PUA map restores the original source
   text — byte-for-byte identical.

## Use it

### Via the standalone package

```bash
pip install cute-tokenizer
```

```python
from cute_tokenizer import load_default_tokenizer

tok = load_default_tokenizer()
ids = tok("def hello(): return 42", add_special_tokens=False).input_ids
text = tok.decode(ids, skip_special_tokens=True)
assert text == "def hello(): return 42"
```

For tight inference loops where `BatchEncoding` machinery is overhead,
use `fast_encode` / `fast_decode` — these go straight to the Rust
`cute-bpe` encoder/decoder:

```python
ids = tok.fast_encode("def hello(): return 42")
text = tok.fast_decode(ids)
```

### Via Hugging Face AutoTokenizer

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "HusseinEid/cute-tokenizer",
    trust_remote_code=True,
)
ids = tok("class Foo: pass", add_special_tokens=False).input_ids
text = tok.decode(ids, skip_special_tokens=True)
```

`trust_remote_code=True` is required because the wrapper class
(`CUTETokenizerFast`) runs PUA pre-substitution before delegating to
the byte-level BPE encoder.

## Properties

- **Byte-equal roundtrip** on 1,500 / 1,500 Python holdout files.
- **Deterministic `tokenizer.json`** within a fixed
  `(OS, python, tokenizers, _accel, corpus_hash, seed)` host triple.
  Cross-platform byte-identity of trained artifacts is not part of
  the contract.
- **Atomicity invariants** asserted on every save: model is `BPE`,
  decoder is `ByteLevel`, pre-tokenizer is `ByteLevel`, every mapping
  PUA codepoint has a vocab id.
- **No BMP-PUA collisions** — mappings live in the supplementary
  planes only, so literal BMP-PUA characters in real source code
  (TypeScript Unicode tables, CJK fonts) roundtrip unchanged.

## Citation

```bibtex
@software{cute_tokenizer_2026,
  author  = {Eid, Hussein},
  title   = {CUTE: Compact Unicode Token Encoding via Semantic-Anchored Byte-level BPE},
  year    = {2026},
  url     = {https://github.com/HusseinEid101/CUTE},
  version = {1.0.1}
}
```

## License

MIT. Source, training scripts, benchmark suite, and full reproduction
instructions live at <https://github.com/HusseinEid101/CUTE>.
"""


def main() -> int:
    from huggingface_hub import HfApi

    api = HfApi()

    print(f"creating/checking {HF_REPO} ...", flush=True)
    api.create_repo(
        repo_id=HF_REPO,
        repo_type="model",
        exist_ok=True,
        private=False,
    )

    # Files to upload (relative to repo root locally -> path in HF repo)
    uploads: list[tuple[str, str]] = [
        # Tokenizer artifacts (the wheel-bundled production model)
        ("model/tokenizer.json", "tokenizer.json"),
        ("model/cute_mapping.json", "cute_mapping.json"),
        ("model/tokenizer_config.json", "tokenizer_config.json"),
        ("model/build_manifest.json", "build_manifest.json"),
        # Wrapper modules (required for trust_remote_code=True)
        ("src/cute_tokenizer/__init__.py", "cute_tokenizer/__init__.py"),
        ("src/cute_tokenizer/_version.py", "cute_tokenizer/_version.py"),
        ("src/cute_tokenizer/tokenizer.py", "cute_tokenizer/tokenizer.py"),
        ("src/cute_tokenizer/decode.py", "cute_tokenizer/decode.py"),
        ("src/cute_tokenizer/pretokenizer.py", "cute_tokenizer/pretokenizer.py"),
        ("src/cute_tokenizer/patterns.py", "cute_tokenizer/patterns.py"),
        ("src/cute_tokenizer/pua.py", "cute_tokenizer/pua.py"),
        ("src/cute_tokenizer/trainer.py", "cute_tokenizer/trainer.py"),
        ("src/cute_tokenizer/baseline.py", "cute_tokenizer/baseline.py"),
        ("src/cute_tokenizer/config.py", "cute_tokenizer/config.py"),
        ("src/cute_tokenizer/corpus.py", "cute_tokenizer/corpus.py"),
        ("src/cute_tokenizer/frequency.py", "cute_tokenizer/frequency.py"),
        ("src/cute_tokenizer/manifest.py", "cute_tokenizer/manifest.py"),
        ("src/cute_tokenizer/merge_policy.py", "cute_tokenizer/merge_policy.py"),
        ("src/cute_tokenizer/selection.py", "cute_tokenizer/selection.py"),
        ("src/cute_tokenizer/_accel_loader.py", "cute_tokenizer/_accel_loader.py"),
    ]

    for local, remote in uploads:
        local_path = REPO_ROOT / local
        if not local_path.exists():
            print(f"  skip (missing): {local}", flush=True)
            continue
        print(f"  upload: {local} -> {remote}", flush=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote,
            repo_id=HF_REPO,
            repo_type="model",
        )

    # Model card
    card_path = REPO_ROOT / "scripts" / "_hf_readme.md"
    card_path.write_text(MODEL_CARD, encoding="utf-8")
    print("  upload: model card -> README.md", flush=True)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="model",
    )
    card_path.unlink()

    # Squash repo history to a single commit (matches the GitHub V1.0 release).
    print("squashing repo history ...", flush=True)
    api.super_squash_history(repo_id=HF_REPO, repo_type="model")

    print(f"\ndone: https://huggingface.co/{HF_REPO}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
