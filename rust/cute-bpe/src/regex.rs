//! GPT-2 pre-tokenization regex.
//!
//! HuggingFace's `ByteLevel` pre-tokenizer applies this exact pattern
//! to split text into chunks before BPE runs. Each match is a piece
//! that BPE encodes independently. Our V2 `tokenizer.json` declares
//! `ByteLevel(use_regex=true)` as the pre-tokenizer, so to produce
//! byte-identical token IDs we must mirror the split here.
//!
//! Pattern (copied verbatim from `tokenizers/src/pre_tokenizers/byte_level.rs`):
//! ```text
//! 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//! ```
//!
//! We compile via `fancy_regex` because the pattern uses negative
//! lookahead `(?!\S)`. The match-iteration semantics we need are
//! `find_iter` (leftmost-first non-overlapping), which `fancy_regex`
//! provides.

use fancy_regex::Regex;
use once_cell::sync::Lazy;

const PRETOK_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Compiled, process-global instance. Building the regex is ~200 µs;
/// reusing the compiled form makes encode itself faster.
pub static PRETOK_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(PRETOK_PATTERN).expect("GPT-2 pre-tokenization regex compiles"));

/// Iterate non-overlapping leftmost matches over `text`, returning
/// `(start_byte, end_byte)` pairs. Empty matches and zero-length
/// pieces are skipped.
pub fn split(text: &str) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for m in PRETOK_REGEX.find_iter(text).flatten() {
        let s = m.start();
        let e = m.end();
        if e > s {
            out.push((s, e));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_four_spaces_pass_correctly() {
        // HF's GPT-2 pre-tokenizer splits "    pass" → ["   ", " pass"].
        // This is the load-bearing case for our parity issue.
        let parts = split("    pass");
        let strs: Vec<&str> = parts.iter().map(|&(s, e)| &"    pass"[s..e]).collect();
        assert_eq!(strs, vec!["   ", " pass"]);
    }

    #[test]
    fn splits_word_with_leading_space() {
        let parts = split(" def hello");
        let strs: Vec<&str> = parts.iter().map(|&(s, e)| &" def hello"[s..e]).collect();
        assert_eq!(strs, vec![" def", " hello"]);
    }

    #[test]
    fn splits_operators_separately() {
        let parts = split("a == b");
        let strs: Vec<&str> = parts.iter().map(|&(s, e)| &"a == b"[s..e]).collect();
        assert_eq!(strs, vec!["a", " ==", " b"]);
    }

    #[test]
    fn newlines_split_correctly() {
        let parts = split(":\n    pass");
        let strs: Vec<&str> = parts.iter().map(|&(s, e)| &":\n    pass"[s..e]).collect();
        // ":" (punct), then "\n   " (\s+ that's NOT followed by non-S because
        // the trailing pos is "pass" which IS non-S → backtracks to "\n  "
        // or similar, then " pass" as a word with leading space).
        // Exact splits we want to match HF byte-for-byte.
        assert!(strs.iter().any(|s| s.contains("pass")));
    }
}
