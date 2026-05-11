//! End-to-end encode + decode pipeline.
//!
//! `CuteBpe` is the top-level engine: load it once from `tokenizer.json`
//! plus `cute_mapping.json`, then call `encode` / `decode` per request.
//! All hot work happens in Rust; the only Python time is the FFI hop.

use std::io;
use std::path::Path;

use cute_core::pretok::{PuaToWord, WordToPua};
use rustc_hash::FxHashMap;
use serde::Deserialize;

use crate::encoder;
use crate::regex as pretok_regex;
use crate::special::SpecialMatcher;
use crate::vocab::Vocab;

/// Run GPT-2 ByteLevel pre-tokenization + BPE encode over a chunk that
/// contains no special tokens. Splits via the cached GPT-2 regex; each
/// match is byte-level-encoded then BPE-encoded independently. Uses a
/// single reusable scratch buffer for the byte-level intermediate to
/// avoid per-piece allocation in the hot loop.
fn encode_pretokenized_bpe(bytes: &[u8], vocab: &Vocab, out: &mut Vec<u32>) {
    let mut bl_buf: Vec<u8> = Vec::with_capacity(64);
    let Ok(text) = std::str::from_utf8(bytes) else {
        encoder::encode_raw_bytes_with_buf(bytes, vocab, &mut bl_buf, out);
        return;
    };
    let mut cursor: usize = 0;
    for (start, end) in pretok_regex::split(text) {
        if start > cursor {
            encoder::encode_raw_bytes_with_buf(&bytes[cursor..start], vocab, &mut bl_buf, out);
        }
        encoder::encode_raw_bytes_with_buf(&bytes[start..end], vocab, &mut bl_buf, out);
        cursor = end;
    }
    if cursor < bytes.len() {
        encoder::encode_raw_bytes_with_buf(&bytes[cursor..], vocab, &mut bl_buf, out);
    }
}

/// Loaded BPE engine. Build once, share across many encode/decode calls.
pub struct CuteBpe {
    pub(crate) vocab: Vocab,
    pub(crate) specials: Option<SpecialMatcher>,
    /// PUA mapping: word → PUA char. Used by encode.
    pub(crate) word_to_pua: WordToPua,
    /// PUA mapping inverse: PUA char → word. Used by decode.
    pub(crate) pua_to_word: PuaToWord,
}

impl CuteBpe {
    pub fn from_paths(tokenizer_json: &Path, cute_mapping_json: &Path) -> io::Result<Self> {
        let vocab = Vocab::from_tokenizer_json(tokenizer_json)?;
        let specials = SpecialMatcher::build(&vocab);
        let mapping_bytes = std::fs::read(cute_mapping_json)?;
        let (word_to_pua, pua_to_word) = parse_cute_mapping(&mapping_bytes)?;
        Ok(Self {
            vocab,
            specials,
            word_to_pua,
            pua_to_word,
        })
    }

    /// Encode `text` into token IDs end-to-end:
    ///   1. PUA pre-substitution (`cute_core::pretok::pretokenize_to_string`).
    ///   2. Compound `AddedToken` matching (`SpecialMatcher`).
    ///   3. Ordinary BPE on segments between matches.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let substituted: String = if self.word_to_pua.is_empty() {
            text.to_string()
        } else {
            cute_core::pretok::pretokenize_to_string(text, &self.word_to_pua)
        };
        self.encode_bytes(substituted.as_bytes())
    }

    /// Encode a raw byte buffer (assumed already PUA-substituted).
    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::with_capacity(bytes.len() / 3 + 1);
        match self.specials.as_ref().filter(|m| m.has_any_match(bytes)) {
            None => encode_pretokenized_bpe(bytes, &self.vocab, &mut out),
            Some(matcher) => {
                let mut cursor: usize = 0;
                for (s, e, id) in matcher.find_all(bytes) {
                    if s > cursor {
                        encode_pretokenized_bpe(&bytes[cursor..s], &self.vocab, &mut out);
                    }
                    out.push(id);
                    cursor = e;
                }
                if cursor < bytes.len() {
                    encode_pretokenized_bpe(&bytes[cursor..], &self.vocab, &mut out);
                }
            }
        }
        out
    }

    /// Decode token ids back to text. Inverse of [`Self::encode`].
    pub fn decode(&self, ids: &[u32]) -> String {
        crate::decoder::decode(ids, &self.vocab, &self.pua_to_word)
    }

    /// Vocab size (largest ID + 1).
    pub fn vocab_size(&self) -> u32 {
        self.vocab.vocab_size
    }
}

#[derive(Deserialize)]
struct CuteMappingJson {
    word_to_codepoint: FxHashMap<String, u32>,
}

fn parse_cute_mapping(json: &[u8]) -> io::Result<(WordToPua, PuaToWord)> {
    let parsed: CuteMappingJson = serde_json::from_slice(json).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cute_mapping.json parse: {e}"),
        )
    })?;
    let mut word_to_pua: WordToPua = ahash::AHashMap::with_capacity(parsed.word_to_codepoint.len());
    let mut pua_to_word: PuaToWord = ahash::AHashMap::with_capacity(parsed.word_to_codepoint.len());
    for (word, cp) in parsed.word_to_codepoint {
        let Some(ch) = char::from_u32(cp) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid codepoint U+{cp:X} for word {word:?}"),
            ));
        };
        word_to_pua.insert(word.clone(), ch);
        pua_to_word.insert(ch, word);
    }
    Ok((word_to_pua, pua_to_word))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn paths() -> (PathBuf, PathBuf) {
        let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../model")
            .canonicalize()
            .expect("model exists");
        (
            model.join("tokenizer.json"),
            model.join("cute_mapping.json"),
        )
    }

    #[test]
    fn loads_engine() {
        let (t, m) = paths();
        let bpe = CuteBpe::from_paths(&t, &m).expect("loads");
        assert!(bpe.vocab_size() > 200_000);
        assert!(bpe.word_to_pua.len() == 50_000);
    }

    #[test]
    fn encodes_simple_def() {
        let (t, m) = paths();
        let bpe = CuteBpe::from_paths(&t, &m).expect("loads");
        let ids = bpe.encode("def");
        assert!(!ids.is_empty());
    }

    #[test]
    fn encodes_with_special_marker() {
        let (t, m) = paths();
        let bpe = CuteBpe::from_paths(&t, &m).expect("loads");
        let ids = bpe.encode("hello <|endoftext|> world");
        // The endoftext id is 0; verify it's in the output.
        assert!(ids.contains(&0), "got: {ids:?}");
    }
}
