//! cute-core: pure-Rust primitives for the CUTE tokenizer.
//!
//! No Python dependency. Tested and benchmarked independently. The
//! `cute_tokenizer_accel` crate provides a thin PyO3 facade over this one.
//!
//! # Module map
//!
//! - [`tokens`]   — `TOKEN_REGEX` equivalent + `iter_tokens`.
//! - [`identifier`] — `is_identifier`, `split_identifier`.
//! - [`pua`]      — Private-Use-Area range checks (mirrors `cute_tokenizer.pua`).
//! - [`pretok`]   — `cute_split_text`, `pretokenize_to_string`.
//! - [`decode`]   — `reverse_pua_substitute`.
//! - [`frequency`] — `count_in_text` and Rayon-parallel `count_frequencies`.
//! - [`secrets`]  — Aho-Corasick / `RegexSet` over `SECRET_PATTERNS`.

pub mod decode;
pub mod frequency;
pub mod identifier;
pub mod pretok;
pub mod pua;
pub mod secrets;
pub mod tokens;

/// Build identifier shown by the Python wrapper to confirm Rust accel is loaded.
pub const ACCEL_BUILD_TAG: &str = concat!("cute-core ", env!("CARGO_PKG_VERSION"));
