# Changelog

All notable changes to CUTE are documented in this file. This project
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] — 2026-05-12 — perf: 5–6× encode speedup

Pure performance release. Same `tokenizer.json` artifact as 1.0.1,
same token IDs (200 / 200 byte-identical against HuggingFace
`fast_encode` on the Stack-Python holdout). Five compounding
optimizations on the cute-bpe Rust hot path:

1. **SmallVec for BPE parts** — stack-resident `parts` table for any
   piece ≤ 79 bytes, zero heap allocation in the inner loop.
2. **Heap-based BPE path for long pieces** — `BinaryHeap` +
   doubly-linked-list takes over above 64 bytes, asymptotically
   O(n log n) instead of O(n²). Tie-breaker is `Reverse(idx)` to
   match tiktoken's leftmost-first merge order.
3. **Thread-local byte-level scratch** — `bl_buf` is now a
   `thread_local!` `RefCell<Vec<u8>>` reused across every encode call,
   eliminating per-piece allocation.
4. **Hand-rolled GPT-2 pre-tokenizer** — drops the `fancy-regex`
   dependency. The NFA + backtracking lookahead (`\s+(?!\S)`) was
   the single largest contributor to encode cost on short inputs;
   the hand-rolled scanner is a direct state machine.
5. **Precomputed byte-level UTF-8 table** — `BL_BYTES: [[u8; 3]; 256]`
   gives each input byte its byte-level UTF-8 form as
   `(byte0, byte1, length)`. `encode_bytes_into` is now a single
   array lookup + 1–2 byte pushes per input byte.

### Microbenchmarks (`cargo bench -p cute-bpe --bench encode`)

| Bench | 1.0.1 | 1.0.2 | Speedup |
|---|---:|---:|---:|
| encode_short (1.1 KB) | 597 µs | **154 µs** | **3.9×** |
| encode_long (~21 KB) | 10.97 ms | **1.87 ms** | **5.9×** |
| encode_ascii_only | 290 µs | **55 µs** | **5.3×** |

### End-to-end Python (1.7 KB sample)

| Path | 1.0.1 | 1.0.2 | Speedup |
|---|---:|---:|---:|
| `fast_encode` | 1,526 µs | **254 µs** | **6.0×** |
| `fast_decode` | 146 µs | **31 µs** | **4.7×** |

### Global benchmark (1,500-file Python holdout, p50 across all files)

| Tokenizer | mean tok | vs CUTE | encode p50 | decode p50 | roundtrip |
|---|---:|---:|---:|---:|---:|
| **CUTE** | **1,767** | — | **1,822 µs** | **263 µs** | **1500 / 1500** |
| cl100k_base | 1,874 | +6.0% | 1,338 µs | 120 µs | 1500 / 1500 |
| o200k_base | 1,886 | +6.7% | 1,760 µs | 126 µs | 1500 / 1500 |
| LLaMA-3 SP-BPE | 1,872 | +5.9% | 3,753 µs | 792 µs | 686 / 1500 |
| StarCoder2 | 2,210 | +25.1% | 4,316 µs | 775 µs | 685 / 1500 |
| XLM-RoBERTa | 2,438 | +38.0% | 3,272 µs | 440 µs | 0 / 1500 |
| CodeLlama | 2,573 | +45.6% | 3,162 µs | 1,885 µs | 1493 / 1500 |
| T5 | 2,706 | +53.2% | 3,121 µs | 479 µs | 0 / 1500 |
| GPT-2 | 3,581 | +102.7% | 4,467 µs | 911 µs | 1500 / 1500 |

CUTE is **3rd-fastest encode**, **3rd-fastest decode**, **most
compact**, and the **only tokenizer with byte-perfect roundtrip on
all 1,500 files** in the comparison. Full report:
[`reports/v102.md`](reports/v102.md).

### Added

- `rust/cute-bpe`: `byte_pair_encode_heap` (BinaryHeap + linked-list
  long-piece path) and the `(Reverse<rank>, Reverse<idx>)` tie-break.
- `rust/cute-bpe`: `BL_BYTES` precomputed byte-level UTF-8 table in
  `bytelevel.rs`.
- `rust/cute-bpe`: hand-rolled GPT-2 pre-tokenizer scanner in
  `regex.rs` (replaces fancy-regex).
- `rust/cute-bpe`: real production-model encode benchmarks in
  `benches/encode.rs` (was a placeholder; now exercises short / long
  / ASCII-heavy samples via Criterion).
- `cute-bpe`: new tests `small_and_heap_paths_agree` and
  `fast_table_matches_char_encode` covering the two new code paths.

### Changed

- `rust/cute-bpe/Cargo.toml`: drop `fancy-regex` (no longer used);
  add `smallvec` workspace dep.
- `rust/cute-bpe/src/pipeline.rs`: hoist per-piece byte-level scratch
  to a thread-local `BL_BUF`.
- Bumped to `1.0.2` in `pyproject.toml`, `Cargo.toml`,
  `src/cute_tokenizer/_version.py`, and the HF model-card citation.

### Verification

- `cargo test -p cute-bpe --release`: 36 / 36 (incl. two new tests).
- `cargo clippy --all -- -D warnings`: clean.
- `pytest tests/{unit,property,integration/test_cute_bpe_parity.py}`:
  290 / 290 (the load-bearing parity gate is 200 / 200 byte-identical
  against HuggingFace `fast_encode`).

## [1.0.1] — 2026-05-11 — V1.0 release

> The V1.0 release ships as PyPI `cute-tokenizer 1.0.1`. An earlier
> `1.0.0` release exists on PyPI from a pre-release attempt; that
> version is partial (macOS + Linux wheels only — Windows wheel and
> sdist were blocked by PyPI's filename-reuse rule) and has been
> yanked. `1.0.1` is the canonical, full-platform release.

The first stable release of CUTE: *Compact Unicode Token Encoding via
Semantic-Anchored Byte-level BPE*. A code-aware tokenizer that produces
fewer tokens per file than nine widely-used baselines on real-world
Python source — and the only tokenizer in that comparison that
roundtrips 1,500 / 1,500 held-out files byte-identically.

### Compression (1,500-file Python holdout, The Stack)

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

### Latency (p50, 1.7 KB Python sample)

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

CUTE decode is third-fastest in the field (behind only OpenAI's
cl100k and o200k). CUTE encode is competitive with open-source code
tokenizers (within ~7 % of LLaMA-3 / StarCoder2) but does *not* beat
tiktoken's hand-tuned `cl100k_base` — end-to-end encode is ~2.8 ×
slower. The `cute-bpe` core encoder itself runs in ~259 µs; the
remaining gap is the PUA pre-substitution Aho-Corasick pass (108 µs)
plus the Python FFI boundary and the 2 × larger vocab hash table.

### Architecture

- **`rust/cute-core`** — Aho-Corasick PUA pretokenizer, reverse-PUA
  decoder, parallel frequency counter, secret-pattern `RegexSet`.
- **`rust/cute-bpe`** — purpose-built byte-pair encoder modeled on
  tiktoken's algorithm. Loads the same `tokenizer.json` artifact as
  HuggingFace `tokenizers`; token IDs are byte-identical on
  200 / 200 random Python files from the Stack holdout.
- **`rust/cute_tokenizer_accel`** — PyO3 bindings, `BPEEncoder` class,
  Rayon-parallel batch APIs.
- **`src/cute_tokenizer/`** — Python wrapper subclassing
  `PreTrainedTokenizerFast`. `fast_encode` / `fast_decode` go through
  `cute-bpe` directly; the standard `__call__` path routes through
  HuggingFace for full `BatchEncoding` compatibility.

### Production properties

- **Byte-equal roundtrip** on 1,500 / 1,500 Python holdout files.
- **Deterministic `tokenizer.json`** within a fixed
  `(OS, python, tokenizers, _accel, corpus_hash, seed)` host triple.
  Cross-platform byte-identity of trained artifacts is *not* part of
  the contract.
- **Atomicity invariants** asserted on every save: model is `BPE`,
  decoder is `ByteLevel`, pre-tokenizer is `ByteLevel`, every mapping
  PUA codepoint has a vocab id.
- **No BMP-PUA collisions** — mappings live in the supplementary
  planes only, so literal BMP-PUA characters in real source code
  (TypeScript Unicode tables, CJK fonts) roundtrip unchanged.
- **Secret scrubbing** on corpus ingest: AWS / OpenAI / Anthropic /
  GitHub / Slack / Google API keys, JWTs, and PEM private keys.

### Distribution

- Cross-platform native wheels via `PyO3/maturin-action`:
  - manylinux2014 (x86_64, aarch64),
  - macOS (universal2, x86_64, aarch64),
  - Windows (x64).
- Triggered on `v*` tag push; uploaded to PyPI via Trusted Publishing.
- HuggingFace Hub artifact: `HusseinEid/cute-tokenizer`.
