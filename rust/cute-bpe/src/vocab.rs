//! Load BPE vocabulary, merges, and added-tokens from a HuggingFace
//! `tokenizer.json` into the data structures the encoder/decoder use.

use std::io;
use std::path::Path;

use rustc_hash::FxHashMap;
use serde::Deserialize;

/// One entry in `added_tokens` at the top level of `tokenizer.json`.
#[derive(Debug, Deserialize)]
struct AddedTokenJson {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
    #[serde(default)]
    #[allow(dead_code)]
    single_word: bool,
    #[serde(default)]
    #[allow(dead_code)]
    lstrip: bool,
    #[serde(default)]
    #[allow(dead_code)]
    rstrip: bool,
    #[serde(default)]
    #[allow(dead_code)]
    normalized: bool,
}

/// The `model` block.
#[derive(Debug, Deserialize)]
struct ModelJson {
    #[serde(rename = "type")]
    _ty: String,
    vocab: FxHashMap<String, u32>,
    merges: Vec<MergePair>,
}

/// Merges are serialized as 2-element JSON arrays.
#[derive(Debug, Deserialize)]
#[serde(from = "(String, String)")]
struct MergePair(String, String);

impl From<(String, String)> for MergePair {
    fn from(t: (String, String)) -> Self {
        MergePair(t.0, t.1)
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    added_tokens: Vec<AddedTokenJson>,
    model: ModelJson,
}

/// All vocabulary state needed by the encoder + decoder.
pub struct Vocab {
    /// Byte sequence → token id (for BPE merges + base byte entries).
    pub encoder: FxHashMap<Vec<u8>, u32>,
    /// Token id → byte sequence. Sparse `Vec` indexed by id; entries
    /// not in the BPE vocab (e.g. compound `AddedToken`s living above
    /// the BPE id range) hold their UTF-8 bytes via the `added_tokens` map.
    pub decoder: Vec<Vec<u8>>,
    /// (left, right) byte-sequence pair → merge rank (lower = earlier merge).
    pub merge_ranks: FxHashMap<(Vec<u8>, Vec<u8>), u32>,
    /// Compound + special `AddedToken`s, content → id. Matched atomically
    /// before BPE during encode.
    pub special_encoder: FxHashMap<String, u32>,
    /// Inverse: id → content string. Used during decode for ids that
    /// belong to special / compound tokens.
    pub special_decoder: FxHashMap<u32, String>,
    /// Total vocab size (highest id + 1). Useful for the decoder Vec.
    pub vocab_size: u32,
}

impl Vocab {
    /// Load from a HuggingFace `tokenizer.json` file.
    pub fn from_tokenizer_json(path: &Path) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_tokenizer_json_bytes(&bytes)
    }

    /// Load from a `tokenizer.json` payload already in memory.
    pub fn from_tokenizer_json_bytes(json: &[u8]) -> io::Result<Self> {
        let parsed: TokenizerJson = serde_json::from_slice(json).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tokenizer.json parse error: {e}"),
            )
        })?;

        // Special tokens — id is whatever the JSON says.
        let mut special_encoder: FxHashMap<String, u32> =
            FxHashMap::with_capacity_and_hasher(parsed.added_tokens.len(), Default::default());
        let mut special_decoder: FxHashMap<u32, String> =
            FxHashMap::with_capacity_and_hasher(parsed.added_tokens.len(), Default::default());
        for at in &parsed.added_tokens {
            special_encoder.insert(at.content.clone(), at.id);
            special_decoder.insert(at.id, at.content.clone());
        }

        // BPE vocab: keys in `tokenizer.json` are *byte-level-encoded*
        // strings (HF ByteLevel pre-tokenizer). E.g. " self" is stored
        // as "Ġself" whose UTF-8 bytes encode the byte-level codepoints.
        // We key our encoder by these UTF-8 bytes DIRECTLY — that's the
        // representation BPE operates on after the per-piece byte-level
        // encoding pass in [`pipeline`]. Trying to decode keys back to
        // raw input bytes (via the GPT-2 inverse table) silently
        // collapses different vocab entries that share the same
        // post-decode bytes (notably PUA codepoints that exist in
        // `initial_alphabet` AND as a learned merge of their UTF-8
        // bytes), so we leave them as-is.
        let mut encoder: FxHashMap<Vec<u8>, u32> =
            FxHashMap::with_capacity_and_hasher(parsed.model.vocab.len(), Default::default());
        let max_id: u32 = parsed
            .model
            .vocab
            .values()
            .copied()
            .max()
            .unwrap_or(0)
            .max(parsed.added_tokens.iter().map(|a| a.id).max().unwrap_or(0));
        let mut decoder: Vec<Vec<u8>> = vec![Vec::new(); (max_id + 1) as usize];
        let special_ids: rustc_hash::FxHashSet<u32> = parsed
            .added_tokens
            .iter()
            .filter(|a| a.special)
            .map(|a| a.id)
            .collect();
        for (key, id) in &parsed.model.vocab {
            let bytes = key.as_bytes().to_vec();
            if special_ids.contains(id) {
                // Special-token entries: their key (e.g. "<|endoftext|>")
                // is literal text, not byte-level-encoded. Don't put in
                // the BPE encoder; the SpecialMatcher handles them.
                decoder[*id as usize] = bytes;
                continue;
            }
            encoder.insert(bytes.clone(), *id);
            decoder[*id as usize] = bytes;
        }

        // Compound AddedTokens (ids past the BPE vocab range): the decoder
        // needs their content's UTF-8 bytes so decode() can stitch text
        // together. Encode side handles them via `special_encoder` /
        // OR-regex (P3), so no entry in `encoder` for these.
        for at in &parsed.added_tokens {
            let slot = at.id as usize;
            if slot < decoder.len() && decoder[slot].is_empty() {
                decoder[slot] = at.content.as_bytes().to_vec();
            }
        }

        // Merge ranks: index in the merges array. Pairs are byte-level
        // strings just like vocab keys; we store their UTF-8 bytes.
        let mut merge_ranks: FxHashMap<(Vec<u8>, Vec<u8>), u32> =
            FxHashMap::with_capacity_and_hasher(parsed.model.merges.len(), Default::default());
        for (rank, MergePair(left, right)) in parsed.model.merges.iter().enumerate() {
            merge_ranks.insert(
                (left.as_bytes().to_vec(), right.as_bytes().to_vec()),
                rank as u32,
            );
        }

        Ok(Self {
            encoder,
            decoder,
            merge_ranks,
            special_encoder,
            special_decoder,
            vocab_size: max_id + 1,
        })
    }

    /// Total number of BPE vocab entries (not counting added tokens that
    /// live above the BPE id range).
    pub fn bpe_vocab_size(&self) -> usize {
        self.encoder.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> std::path::PathBuf {
        let p =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model/tokenizer.json");
        p.canonicalize().expect("model/tokenizer.json exists")
    }

    #[test]
    fn loads_v2_tokenizer_json() {
        let v = Vocab::from_tokenizer_json(&model_path()).expect("loads");
        assert!(v.bpe_vocab_size() > 100_000, "got {}", v.bpe_vocab_size());
        assert!(!v.merge_ranks.is_empty());
        assert!(v.vocab_size > 200_000, "got {}", v.vocab_size);
    }

    #[test]
    fn encoder_decoder_roundtrip_on_byte_level_alphabet() {
        let v = Vocab::from_tokenizer_json(&model_path()).expect("loads");
        // Every byte gets a byte-level codepoint. The vocab key is the
        // UTF-8 bytes of that codepoint. Walk the byte-level table and
        // confirm each codepoint's UTF-8 form is in the encoder.
        for b in 0u8..=255 {
            let bl = crate::bytelevel::BYTE_TO_CHAR[b as usize];
            let mut buf = [0u8; 4];
            let bl_bytes = bl.encode_utf8(&mut buf).as_bytes().to_vec();
            let id = v.encoder.get(&bl_bytes).copied().unwrap_or_else(|| {
                panic!(
                    "missing byte-level entry for byte 0x{b:02X} (codepoint U+{:04X})",
                    bl as u32
                );
            });
            assert_eq!(v.decoder[id as usize], bl_bytes);
        }
    }

    #[test]
    fn special_tokens_have_content_in_decoder() {
        let v = Vocab::from_tokenizer_json(&model_path()).expect("loads");
        let endoftext_id = v.special_encoder["<|endoftext|>"];
        assert_eq!(
            v.special_decoder[&endoftext_id], "<|endoftext|>",
            "special encoder/decoder must mirror"
        );
    }

    #[test]
    fn compound_added_tokens_resolvable_by_id() {
        let v = Vocab::from_tokenizer_json(&model_path()).expect("loads");
        // Pick one of the high-id compound tokens — these are sparse, so
        // we scan the special_decoder for any id >= 200k.
        let any_compound = v
            .special_decoder
            .iter()
            .find(|(id, _)| **id >= 200_000)
            .expect("at least one compound added token");
        assert!(!any_compound.1.is_empty());
    }
}
