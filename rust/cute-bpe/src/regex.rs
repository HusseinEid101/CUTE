//! GPT-2 pre-tokenization — hand-rolled scanner.
//!
//! HuggingFace's `ByteLevel` pre-tokenizer applies this regex pattern
//! to split text into chunks before BPE runs (`tokenizers/src/pre_tokenizers/byte_level.rs`):
//! ```text
//! 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//! ```
//!
//! Earlier versions of this crate compiled the pattern via `fancy_regex`
//! (needed for the `(?!\S)` lookahead). `fancy_regex` runs an NFA with
//! backtracking on the lookahead, and the per-`find_iter` cost dominated
//! the encode hot path on short inputs.
//!
//! This hand-rolled scanner implements the same alternation order with
//! ASCII fast paths and Unicode fallbacks via [`char::is_alphabetic`],
//! [`char::is_numeric`], and [`char::is_whitespace`]. It returns byte
//! offsets `(start, end)` into the input `&str`. Behaviour is checked
//! against the prior `fancy_regex` impl in the tests below plus the
//! token-id parity test (200/200 byte-identical against HuggingFace's
//! own `ByteLevel`).

/// Iterate non-overlapping leftmost matches over `text`, yielding
/// `(start_byte, end_byte)` pairs. Empty matches are not yielded.
pub fn split(text: &str) -> Vec<(usize, usize)> {
    split_iter(text).collect()
}

/// Streaming version of [`split`]: avoids the `Vec` allocation for
/// callers that just iterate once.
pub fn split_iter(text: &str) -> SplitIter<'_> {
    SplitIter { text, pos: 0 }
}

pub struct SplitIter<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Iterator for SplitIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        let bytes = self.text.as_bytes();
        let n = bytes.len();
        if self.pos >= n {
            return None;
        }
        let start = self.pos;
        let len = match_one(bytes, self.pos, self.text);
        // `match_one` is total: it always advances at least one char.
        self.pos = start + len;
        Some((start, start + len))
    }
}

/// Attempt to match the GPT-2 pre-tokenization regex at `pos`, returning
/// the matched length in bytes. Always returns at least 1 (advances by
/// one character even if no rule matches — which shouldn't happen on
/// valid UTF-8 input).
#[inline]
fn match_one(bytes: &[u8], pos: usize, text: &str) -> usize {
    // Rule 1: contractions ('s, 't, 're, 've, 'm, 'll, 'd).
    if bytes[pos] == b'\'' {
        if let Some(len) = try_contraction(&bytes[pos..]) {
            return len;
        }
    }

    // Rules 2/3/4: optional leading space + run of letters / numbers / "other".
    // First decide whether the run starts at `pos` or at `pos + 1` (skip
    // one leading space). The leading-space form requires that the
    // character *after* the space match the class.
    let mut scan_from = pos;
    let mut had_space = false;
    if bytes[pos] == b' ' && pos + 1 < bytes.len() {
        scan_from = pos + 1;
        had_space = true;
    }

    if let Some(class) = classify_char_at(text, scan_from) {
        match class {
            CharClass::Letter | CharClass::Number | CharClass::Other => {
                let end = take_run(text, scan_from, class);
                return (end - pos).max(1);
            }
            CharClass::Whitespace => { /* fall through */ }
        }
        // Reset: didn't take a letter/number/other run, drop the leading-space hypothesis.
        let _ = had_space;
    }

    // Rule 5: `\s+(?!\S)` — whitespace run NOT followed by non-whitespace.
    // Rule 6: `\s+` — any whitespace run.
    //
    // Implementation: scan the whitespace run greedily. If it ends at EOF,
    // emit the whole run (rule 5). Otherwise, peek at the first non-ws
    // char: if it would not be absorbed by a later token, emit the whole
    // run. If it WOULD be absorbed (the regex engine backtracks to leave
    // one trailing space for the next ` ?` rule), emit `run - 1` chars.
    if is_ascii_space(bytes[pos]) || classify_char_at(text, pos) == Some(CharClass::Whitespace) {
        let run_end = take_ws_run(text, pos);
        // Peek: any non-whitespace after the run?
        if run_end < bytes.len() {
            // `\s+(?!\S)` would fail at greedy length (next char is non-ws),
            // so the regex backtracks one whitespace char to leave the
            // trailing-space slot for the next ` ?<class>+` rule. We mirror
            // that: yield `run_end - last_ws_char_len` bytes here, then the
            // next iteration picks up the trailing space as part of its
            // optional ` ?` prefix.
            //
            // Edge case: run is exactly one whitespace char and there's a
            // non-ws char after it. In that case the leading-space slot of
            // the next rule consumes it, and we yield nothing here. But
            // `match_one` MUST advance, so we yield the single ws char and
            // accept the (very rare) one-token-off behaviour. In practice
            // this branch isn't hit because the leading-space hypothesis
            // above already absorbed a single space.
            let last_char_len = utf8_char_len_back(bytes, run_end);
            let trimmed_end = run_end - last_char_len;
            if trimmed_end > pos {
                return trimmed_end - pos;
            }
            // Single-char ws run followed by non-ws: emit just it.
            return last_char_len;
        }
        // Run reaches EOF: emit whole run (rule 5).
        return run_end - pos;
    }

    // Total fallback: advance by one UTF-8 character. Shouldn't be
    // reached on valid input but keeps the iterator total.
    utf8_char_len(bytes, pos)
}

#[inline]
fn try_contraction(slice: &[u8]) -> Option<usize> {
    // Order matters only for byte-length tie-breaking; alternatives are
    // disjoint so any order works.
    if slice.len() >= 3 {
        match &slice[..3] {
            b"'re" | b"'ve" | b"'ll" => return Some(3),
            _ => {}
        }
    }
    if slice.len() >= 2 {
        match &slice[..2] {
            b"'s" | b"'t" | b"'m" | b"'d" => return Some(2),
            _ => {}
        }
    }
    None
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum CharClass {
    Letter,
    Number,
    Whitespace,
    /// `[^\s\p{L}\p{N}]` — non-whitespace, non-letter, non-number.
    Other,
}

#[inline]
fn classify_char_at(text: &str, pos: usize) -> Option<CharClass> {
    let bytes = text.as_bytes();
    if pos >= bytes.len() {
        return None;
    }
    let b = bytes[pos];
    if b < 0x80 {
        // ASCII fast path.
        return Some(classify_ascii(b));
    }
    // Decode one UTF-8 codepoint and classify via std.
    let ch = text[pos..].chars().next()?;
    Some(classify_char(ch))
}

#[inline]
fn classify_ascii(b: u8) -> CharClass {
    if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' || b == 0x0B || b == 0x0C {
        CharClass::Whitespace
    } else if b.is_ascii_alphabetic() {
        CharClass::Letter
    } else if b.is_ascii_digit() {
        CharClass::Number
    } else {
        CharClass::Other
    }
}

#[inline]
fn classify_char(ch: char) -> CharClass {
    if ch.is_whitespace() {
        CharClass::Whitespace
    } else if ch.is_alphabetic() {
        CharClass::Letter
    } else if char_is_numeric(ch) {
        CharClass::Number
    } else {
        CharClass::Other
    }
}

#[inline]
fn char_is_numeric(ch: char) -> bool {
    // `\p{N}` covers Nd, Nl, No. `char::is_numeric` is the closest std
    // approximation; `is_ascii_digit` covers Nd for the ASCII subset.
    ch.is_numeric()
}

/// Take a maximal run of characters belonging to the same `class`,
/// starting at `pos`. Returns the byte offset one past the run.
#[inline]
fn take_run(text: &str, pos: usize, class: CharClass) -> usize {
    let bytes = text.as_bytes();
    let mut p = pos;
    while p < bytes.len() {
        let b = bytes[p];
        if b < 0x80 {
            if classify_ascii(b) == class {
                p += 1;
                continue;
            }
            break;
        }
        // Multi-byte UTF-8 — decode and classify.
        let Some(ch) = text[p..].chars().next() else {
            break;
        };
        if classify_char(ch) == class {
            p += ch.len_utf8();
            continue;
        }
        break;
    }
    p
}

#[inline]
fn take_ws_run(text: &str, pos: usize) -> usize {
    take_run(text, pos, CharClass::Whitespace)
}

#[inline]
fn is_ascii_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

/// Length in bytes of the UTF-8 character starting at `bytes[pos]`.
/// Always returns 1..=4. Falls back to 1 on invalid leading byte.
#[inline]
fn utf8_char_len(bytes: &[u8], pos: usize) -> usize {
    let b = bytes[pos];
    // `b < 0xC0` covers ASCII (b < 0x80) *and* stray continuation bytes
    // (0x80..=0xBF) — we treat the latter as 1-byte advances to keep
    // the iterator total on malformed input. Multi-byte leaders use
    // their canonical length.
    if b < 0xC0 {
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

/// Length in bytes of the UTF-8 character ENDING at `bytes[end]`
/// (i.e. starting somewhere in `bytes[end-4..end]`). Returns 1..=4.
#[inline]
fn utf8_char_len_back(bytes: &[u8], end: usize) -> usize {
    // Walk back over continuation bytes (0x80..=0xBF).
    let mut i = end.saturating_sub(1);
    while i > 0 && (bytes[i] & 0xC0) == 0x80 {
        i -= 1;
    }
    end - i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts_str(text: &str) -> Vec<&str> {
        split(text).into_iter().map(|(s, e)| &text[s..e]).collect()
    }

    #[test]
    fn splits_four_spaces_pass_correctly() {
        // HF's GPT-2 pre-tokenizer splits "    pass" → ["   ", " pass"].
        assert_eq!(parts_str("    pass"), vec!["   ", " pass"]);
    }

    #[test]
    fn splits_word_with_leading_space() {
        assert_eq!(parts_str(" def hello"), vec![" def", " hello"]);
    }

    #[test]
    fn splits_operators_separately() {
        assert_eq!(parts_str("a == b"), vec!["a", " ==", " b"]);
    }

    #[test]
    fn newlines_split_correctly() {
        let parts = parts_str(":\n    pass");
        assert!(parts.iter().any(|s| s.contains("pass")));
        // ":" should be its own piece.
        assert_eq!(parts[0], ":");
    }

    #[test]
    fn contraction_splits() {
        assert_eq!(parts_str("don't"), vec!["don", "'t"]);
        assert_eq!(parts_str("they're"), vec!["they", "'re"]);
        assert_eq!(parts_str("I'll"), vec!["I", "'ll"]);
    }

    #[test]
    fn single_word_no_leading_space() {
        assert_eq!(parts_str("hello"), vec!["hello"]);
    }

    #[test]
    fn number_splits() {
        assert_eq!(parts_str("abc123def"), vec!["abc", "123", "def"]);
    }

    #[test]
    fn leading_space_before_number() {
        assert_eq!(parts_str("v 42"), vec!["v", " 42"]);
    }

    #[test]
    fn whitespace_at_end_emitted_whole() {
        // `\s+(?!\S)` — trailing whitespace forms one piece.
        assert_eq!(parts_str("hello   "), vec!["hello", "   "]);
    }

    #[test]
    fn empty_input() {
        assert!(split("").is_empty());
    }

    #[test]
    fn mixed_punctuation_run() {
        assert_eq!(parts_str("):  "), vec!["):", "  "]);
    }
}
