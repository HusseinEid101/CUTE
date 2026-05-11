//! Token-id → text decode.
//!
//! Three-stage inverse of [`crate::pipeline::CuteBpe::encode`]:
//! 1. Per token id, look up its byte-level UTF-8 bytes (or AddedToken
//!    content) in [`Vocab::decoder`] and concatenate.
//! 2. Reverse the GPT-2 byte-level encoding to recover the raw input
//!    bytes (or PUA-substituted text).
//! 3. PUA reverse-substitute via [`cute_core::decode::reverse_pua_substitute`].

use crate::bytelevel;
use crate::vocab::Vocab;
use cute_core::pretok::PuaToWord;

/// Decode a sequence of token ids into the original text.
pub fn decode(ids: &[u32], vocab: &Vocab, pua_to_word: &PuaToWord) -> String {
    // Concat all token byte sequences (these are byte-level UTF-8 bytes
    // for ordinary tokens, raw UTF-8 of AddedToken content for special /
    // compound tokens).
    let mut bl_buf: Vec<u8> = Vec::with_capacity(ids.len() * 4);
    let mut special_runs: Vec<(usize, usize)> = Vec::new();
    let mut special_content: String = String::new();
    let dec = &vocab.decoder;
    for &id in ids {
        let slot = id as usize;
        if slot >= dec.len() {
            continue;
        }
        if vocab.special_decoder.contains_key(&id) {
            // Special / compound token: its byte sequence is raw text,
            // not byte-level-encoded. We need to splice it into the
            // output AFTER byte-level decoding the surrounding regions.
            // To keep this single-pass, we mark the position in the
            // byte-level buffer and stash the raw text aside; after
            // decoding, we insert.
            special_runs.push((bl_buf.len(), special_content.len()));
            let content = &vocab.special_decoder[&id];
            special_content.push_str(content);
            special_runs.last_mut().unwrap().1 = special_content.len();
        } else {
            bl_buf.extend_from_slice(&dec[slot]);
        }
    }

    let raw_bytes = if special_runs.is_empty() {
        bytelevel::decode_bytes(&String::from_utf8_lossy(&bl_buf))
    } else {
        // Decode byte-level pieces and splice special-token content
        // back in at their insertion points. The runs are in token
        // order; bl_buf positions are byte offsets into the
        // concatenated byte-level buffer.
        decode_with_specials(&bl_buf, &special_runs, &special_content)
    };

    let text = String::from_utf8_lossy(&raw_bytes).into_owned();

    if pua_to_word.is_empty() {
        text
    } else {
        cute_core::decode::reverse_pua_substitute(&text, pua_to_word)
    }
}

/// Decode the byte-level buffer and interleave special-token content.
fn decode_with_specials(bl_buf: &[u8], runs: &[(usize, usize)], special_content: &str) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(bl_buf.len() + special_content.len());
    let mut cursor: usize = 0;
    let mut prev_content_end: usize = 0;
    for &(bl_pos, content_end) in runs {
        // Decode the byte-level region [cursor..bl_pos].
        if bl_pos > cursor {
            let chunk = &bl_buf[cursor..bl_pos];
            // Best-effort: byte-level codepoints are 2-byte UTF-8;
            // String::from_utf8_lossy is safe and rare invalid UTF-8
            // gets replacement chars.
            let decoded = bytelevel::decode_bytes(&String::from_utf8_lossy(chunk));
            out.extend_from_slice(&decoded);
        }
        out.extend_from_slice(&special_content.as_bytes()[prev_content_end..content_end]);
        cursor = bl_pos;
        prev_content_end = content_end;
    }
    if cursor < bl_buf.len() {
        let chunk = &bl_buf[cursor..];
        let decoded = bytelevel::decode_bytes(&String::from_utf8_lossy(chunk));
        out.extend_from_slice(&decoded);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::CuteBpe;
    use std::path::PathBuf;

    fn engine() -> CuteBpe {
        let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../model")
            .canonicalize()
            .expect("model exists");
        CuteBpe::from_paths(
            &model.join("tokenizer.json"),
            &model.join("cute_mapping.json"),
        )
        .expect("loads")
    }

    #[test]
    fn round_trip_ascii() {
        let e = engine();
        for text in [
            "def hello(): return 42",
            "class Foo:\n    pass",
            "from typing import List, Dict",
            "if x == y: return None",
            "",
        ] {
            let ids = e.encode(text);
            let back = decode(&ids, &e.vocab, &e.pua_to_word);
            assert_eq!(back, text, "round-trip failed for {text:?}");
        }
    }

    #[test]
    fn round_trip_pua_substituted_word() {
        // "Rounds" is in the PUA mapping; encode substitutes, decode
        // must reverse correctly.
        let e = engine();
        let text = "Rounds floating point";
        let ids = e.encode(text);
        let back = decode(&ids, &e.vocab, &e.pua_to_word);
        assert_eq!(back, text);
    }

    #[test]
    fn round_trip_with_special_token() {
        let e = engine();
        let text = "hello <|endoftext|> world";
        let ids = e.encode(text);
        let back = decode(&ids, &e.vocab, &e.pua_to_word);
        assert_eq!(back, text);
    }
}
