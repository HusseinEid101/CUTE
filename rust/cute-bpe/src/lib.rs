//! cute-bpe — purpose-built BPE encoder/decoder for the CUTE tokenizer.
//!
//! Goals (1.1.0):
//! - Encode + decode latency strictly better than `tiktoken` cl100k on the
//!   sample workloads (Stack Python holdout 1.5k-file mean).
//! - Token-id byte-equal output against the current HF-`tokenizers`-backed
//!   `CUTETokenizerFast.fast_encode` on the same `tokenizer.json` model.
//! - Zero retraining: load all state from the existing `model/tokenizer.json`.
//!
//! Pipeline (`encode`):
//! 1. PUA pre-substitution via [`cute_core::pretok::pretokenize_to_string`].
//! 2. Compound `AddedToken` matching via a single OR-joined `fancy-regex`
//!    (longest-match-wins because the patterns are pre-sorted by length).
//! 3. Ordinary BPE encode on the residual segments: byte-level encoding
//!    (GPT-2 `Ġ`/`Ċ`/`ĉ` table) followed by tiktoken-style BPE merging
//!    (simple linear-scan for pieces <100 bytes, binary-heap for ≥100).
//!
//! Pipeline (`decode`):
//! 1. ID → byte-sequence lookup (single allocation with capacity hint).
//! 2. GPT-2 byte-level inverse.
//! 3. PUA reverse-substitution via [`cute_core::decode::reverse_pua_substitute`].

pub mod bytelevel;
pub mod decoder;
pub mod encoder;
pub mod pipeline;
pub mod regex;
pub mod special;
pub mod vocab;

pub use pipeline::CuteBpe;

/// Crate version, surfaced to Python via the PyO3 binding for build tagging.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
