//! Token-level regex scanner.
//!
//! Mirrors the Python `TOKEN_REGEX` at `patterns.py:22`. The pattern is
//! compiled once via `once_cell::sync::Lazy`. We expose two iterators:
//!
//! - [`iter_token_strings`] — just the matched substring per token.
//! - [`iter_token_byte_spans`] — `(token, byte_start, byte_end)` for callers
//!   (like [`crate::pretok::cute_split_text`]) that need to fill the gaps
//!   between matches.
//!
//! Note: Rust string slices are byte-indexed (UTF-8). We expose BYTE offsets
//! and slice in Rust internally; we never hand byte offsets back to Python,
//! to avoid Python codepoint-vs-byte confusion. Python keeps its own
//! `iter_tokens` for the rare cases where its callers want the offsets.

use once_cell::sync::Lazy;
use regex::Regex;

/// The CUTE token regex. Matches in priority order:
///
/// 1. Emoji sequences (with ZWJ, VS16, keycap, emoji modifiers).
/// 2. Word / identifier (letters, digits, underscores, internal hyphens).
/// 3. Multi-char operators (==, !=, <=, >=, <<, >>, +=, -=, *=, /=, %=, &&, ||, ->, =>, ::, :=, ..., ..).
/// 4. Single non-space punctuation.
///
/// Whitespace is intentionally NOT matched; gap-fill is the consumer's job.
pub static TOKEN_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(TOKEN_PATTERN).expect("CUTE TOKEN_PATTERN compiles"));

const TOKEN_PATTERN: &str = r"(?x)
(?:
    [\p{Emoji_Presentation}\p{Extended_Pictographic}]
    (?:\u{200D}[\p{Emoji_Presentation}\p{Extended_Pictographic}])*
    [\u{FE0F}\u{20E3}\p{Emoji_Modifier}]*
)+
| [\p{L}\p{N}_](?:[\p{L}\p{N}_\-]*[\p{L}\p{N}_])?
| (?:!=|==|<=|>=|<<|>>|\+=|-=|\*=|/=|%=|&&|\|\||->|=>|::|:=|\.\.\.|\.\.)
| [^\s\w]
";

/// Iterate matched token substrings only. Used by frequency counting.
#[inline]
pub fn iter_token_strings(text: &str) -> impl Iterator<Item = &str> + '_ {
    TOKEN_REGEX.find_iter(text).map(|m| m.as_str())
}

/// Iterate `(token, byte_start, byte_end)` for callers needing gap context.
#[inline]
pub fn iter_token_byte_spans(text: &str) -> impl Iterator<Item = (&str, usize, usize)> + '_ {
    TOKEN_REGEX
        .find_iter(text)
        .map(|m| (m.as_str(), m.start(), m.end()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(text: &str) -> Vec<&str> {
        iter_token_strings(text).collect()
    }

    #[test]
    fn simple_words_and_punct() {
        assert_eq!(collect("hello world"), vec!["hello", "world"]);
        assert_eq!(collect("foo, bar."), vec!["foo", ",", "bar", "."]);
    }

    #[test]
    fn whitespace_is_not_a_token() {
        let toks = collect("a  b");
        assert_eq!(toks, vec!["a", "b"]);
    }

    #[test]
    fn multi_char_operators() {
        assert_eq!(collect("a == b"), vec!["a", "==", "b"]);
        assert_eq!(collect("x->y"), vec!["x", "->", "y"]);
        assert_eq!(collect("a := 1"), vec!["a", ":=", "1"]);
    }

    #[test]
    fn identifier_with_internal_hyphen() {
        assert_eq!(collect("ab-cd"), vec!["ab-cd"]);
    }

    #[test]
    fn unicode_letters() {
        let toks = collect("héllo wörld");
        assert_eq!(toks, vec!["héllo", "wörld"]);
    }

    #[test]
    fn keeps_byte_offsets_in_bounds() {
        let text = "α + β = γ";
        for (tok, start, end) in iter_token_byte_spans(text) {
            assert_eq!(tok, &text[start..end]);
        }
    }
}
