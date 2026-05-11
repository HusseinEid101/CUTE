//! `reverse_pua_substitute` — replace PUA chars with their original words.
//! Mirrors `decode.py:13`.

use crate::pretok::PuaToWord;
use crate::pua::is_pua_char;

/// Replace every PUA character in `text` with its original mapped word.
///
/// Characters not in the mapping pass through unchanged. PUA chars not in
/// the mapping survive the round-trip as themselves.
pub fn reverse_pua_substitute(text: &str, pua_to_word: &PuaToWord) -> String {
    if pua_to_word.is_empty() {
        return text.to_string();
    }

    // Fast path: no PUA chars present.
    if !text.chars().any(is_pua_char) {
        return text.to_string();
    }

    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if is_pua_char(ch) {
            if let Some(word) = pua_to_word.get(&ch) {
                out.push_str(word);
            } else {
                out.push(ch);
            }
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mapping(pairs: &[(char, &str)]) -> PuaToWord {
        pairs.iter().map(|(c, w)| (*c, w.to_string())).collect()
    }

    #[test]
    fn empty_mapping_is_identity() {
        let m = PuaToWord::new();
        assert_eq!(reverse_pua_substitute("hello", &m), "hello");
    }

    #[test]
    fn ascii_unaffected() {
        let m = make_mapping(&[('\u{F0001}', "def")]);
        assert_eq!(reverse_pua_substitute("hello world", &m), "hello world");
    }

    #[test]
    fn substitutes_known_pua() {
        let m = make_mapping(&[('\u{F0001}', "def"), ('\u{F0002}', "return")]);
        let input = format!("\u{F0001} f(): \u{F0002} 1");
        assert_eq!(reverse_pua_substitute(&input, &m), "def f(): return 1");
    }

    #[test]
    fn unknown_pua_passes_through() {
        let m = make_mapping(&[('\u{F0001}', "def")]);
        let input = format!("\u{F0001} \u{F0099}");
        // F0001 -> "def", F0099 not in map -> stays as PUA
        assert_eq!(reverse_pua_substitute(&input, &m), "def \u{F0099}");
    }

    #[test]
    fn round_trip_with_pretokenize() {
        use crate::pretok::{pretokenize_to_string, WordToPua};
        let mut wtp: WordToPua = WordToPua::new();
        wtp.insert("def".to_string(), '\u{F0001}');
        wtp.insert("return".to_string(), '\u{F0002}');

        let mut ptw = PuaToWord::new();
        ptw.insert('\u{F0001}', "def".to_string());
        ptw.insert('\u{F0002}', "return".to_string());

        let original = "def foo(): return 1";
        let pre = pretokenize_to_string(original, &wtp);
        let back = reverse_pua_substitute(&pre, &ptw);
        assert_eq!(back, original);
    }
}
