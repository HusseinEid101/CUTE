//! `cute_split_text` and `pretokenize_to_string`.
//!
//! Mirrors `pretokenizer.py:51`. Walks the corpus regex matches, fills
//! whitespace gaps, substitutes mapped tokens with their PUA character,
//! and falls back to identifier sub-splitting (with PUA lookup per part)
//! for ASCII identifier tokens.
//!
//! Round-trip property (preserved):
//! `pretokenize_to_string(text, empty_mapping)` returns `text` byte-for-byte.

use ahash::AHashMap;

use crate::identifier;
use crate::tokens;

/// Word → PUA codepoint mapping. We use `char` (a Unicode scalar value) as
/// the PUA representation to mirror Python's "1-char str" while staying
/// branch-free in Rust.
pub type WordToPua = AHashMap<String, char>;

/// PUA codepoint → original word mapping (for `decode::reverse_pua_substitute`).
pub type PuaToWord = AHashMap<char, String>;

/// Produce the canonical pre-tokenized string used by the underlying BPE.
///
/// Mirrors `pretokenize_to_string(text, mapping)` at `pretokenizer.py:142`.
pub fn pretokenize_to_string(text: &str, word_to_pua: &WordToPua) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_end = 0usize;

    for (tok, start, end) in tokens::iter_token_byte_spans(text) {
        if start > last_end {
            out.push_str(&text[last_end..start]);
        }
        emit_token(tok, word_to_pua, &mut out);
        last_end = end;
    }
    if last_end < text.len() {
        out.push_str(&text[last_end..]);
    }
    out
}

/// Tokenize and split into pieces. `''.join(pieces) == pretokenize_to_string(...)`.
///
/// Returned `Vec<String>` mirrors `cute_split_text(text, mapping) -> list[str]`.
/// Allocates one String per piece to keep the FFI boundary simple; if this
/// becomes hot we can switch to `Vec<Cow<'_, str>>` and convert at the
/// PyO3 boundary instead.
pub fn cute_split_text(text: &str, word_to_pua: &WordToPua) -> Vec<String> {
    let mut pieces: Vec<String> = Vec::new();
    let mut last_end = 0usize;

    for (tok, start, end) in tokens::iter_token_byte_spans(text) {
        if start > last_end {
            pieces.push(text[last_end..start].to_string());
        }
        emit_token_pieces(tok, word_to_pua, &mut pieces);
        last_end = end;
    }
    if last_end < text.len() {
        pieces.push(text[last_end..].to_string());
    }
    pieces
}

#[inline]
fn emit_token(tok: &str, word_to_pua: &WordToPua, out: &mut String) {
    if let Some(&pua) = word_to_pua.get(tok) {
        out.push(pua);
        return;
    }
    if identifier::is_identifier(tok) {
        emit_identifier_pieces_into_string(tok, word_to_pua, out);
    } else {
        out.push_str(tok);
    }
}

#[inline]
fn emit_token_pieces(tok: &str, word_to_pua: &WordToPua, pieces: &mut Vec<String>) {
    if let Some(&pua) = word_to_pua.get(tok) {
        let mut s = String::with_capacity(pua.len_utf8());
        s.push(pua);
        pieces.push(s);
        return;
    }
    if identifier::is_identifier(tok) {
        emit_identifier_pieces_into_vec(tok, word_to_pua, pieces);
    } else {
        pieces.push(tok.to_string());
    }
}

/// Walk `ident` left-to-right, weaving underscores back in, and emit a piece
/// per sub-part (PUA-substituted where possible).
///
/// Mirrors `_emit_identifier_pieces` at `pretokenizer.py:103`.
fn emit_identifier_pieces_into_string(ident: &str, word_to_pua: &WordToPua, out: &mut String) {
    let parts = identifier::split_identifier(ident);
    if parts.is_empty() {
        out.push_str(ident);
        return;
    }

    // Pre-validate by simulating the walk; if it desyncs, bail to atomic
    // emission. Doing this in a temp String keeps the partial work invisible.
    let mut local = String::with_capacity(ident.len());
    let bytes = ident.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;

    for part in &parts {
        while i < n && bytes[i] == b'_' {
            local.push('_');
            i += 1;
        }
        let p_len = part.len();
        if i + p_len > n || &bytes[i..i + p_len] != part.as_bytes() {
            // Defensive desync — emit ident verbatim, mirrors Python bailout.
            out.push_str(ident);
            return;
        }
        if let Some(&pua) = word_to_pua.get(part.as_str()) {
            local.push(pua);
        } else {
            local.push_str(part);
        }
        i += p_len;
    }
    while i < n {
        let ch = ident[i..].chars().next().expect("char boundary");
        local.push(ch);
        i += ch.len_utf8();
    }

    out.push_str(&local);
}

fn emit_identifier_pieces_into_vec(ident: &str, word_to_pua: &WordToPua, pieces: &mut Vec<String>) {
    let parts = identifier::split_identifier(ident);
    if parts.is_empty() {
        pieces.push(ident.to_string());
        return;
    }

    // Stage emissions locally so we can bail without leaving partial work.
    let mut staged: Vec<String> = Vec::with_capacity(parts.len() + 4);
    let bytes = ident.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;

    for part in &parts {
        while i < n && bytes[i] == b'_' {
            staged.push("_".to_string());
            i += 1;
        }
        let p_len = part.len();
        if i + p_len > n || &bytes[i..i + p_len] != part.as_bytes() {
            pieces.push(ident.to_string());
            return;
        }
        if let Some(&pua) = word_to_pua.get(part.as_str()) {
            let mut s = String::with_capacity(pua.len_utf8());
            s.push(pua);
            staged.push(s);
        } else {
            staged.push(part.clone());
        }
        i += p_len;
    }
    while i < n {
        let ch = ident[i..].chars().next().expect("char boundary");
        staged.push(ch.to_string());
        i += ch.len_utf8();
    }

    pieces.extend(staged);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mapping(pairs: &[(&str, char)]) -> WordToPua {
        pairs.iter().map(|(w, c)| (w.to_string(), *c)).collect()
    }

    #[test]
    fn empty_mapping_is_identity() {
        let m = WordToPua::new();
        let text = "def foo(x):\n    return x + 1\n";
        assert_eq!(pretokenize_to_string(text, &m), text);
    }

    #[test]
    fn simple_substitution() {
        let m = make_mapping(&[("def", '\u{F0001}'), ("return", '\u{F0002}')]);
        let out = pretokenize_to_string("def f(): return 1", &m);
        assert!(out.contains('\u{F0001}'));
        assert!(out.contains('\u{F0002}'));
        assert!(!out.contains("def "));
        assert!(out.contains("f"));
    }

    #[test]
    fn whitespace_preserved() {
        let m = WordToPua::new();
        let text = "a   b\n\tc";
        assert_eq!(pretokenize_to_string(text, &m), text);
    }

    #[test]
    fn identifier_subsplit_substitution() {
        // "calculateTotal" -> ["calculate", "Total"] -> map "calculate" -> PUA, keep "Total"
        let m = make_mapping(&[("calculate", '\u{F0010}')]);
        let out = pretokenize_to_string("var calculateTotal = 1", &m);
        // PUA substituted for "calculate", "Total" passes through
        assert!(out.contains('\u{F0010}'));
        assert!(out.contains("Total"));
        assert!(!out.contains("calculate"));
    }

    #[test]
    fn cute_split_text_join_matches_string_form() {
        let m = make_mapping(&[("foo", '\u{F0100}'), ("bar", '\u{F0101}')]);
        let text = "foo + bar - baz";
        let pieces = cute_split_text(text, &m);
        let joined: String = pieces.concat();
        assert_eq!(joined, pretokenize_to_string(text, &m));
    }

    #[test]
    fn underscores_preserved() {
        let m = make_mapping(&[("get", '\u{F0200}'), ("user", '\u{F0201}')]);
        let text = "get_user_id";
        let out = pretokenize_to_string(text, &m);
        assert!(out.contains('\u{F0200}'));
        assert!(out.contains('\u{F0201}'));
        assert!(out.contains("_"));
        assert!(out.contains("id"));
    }
}
