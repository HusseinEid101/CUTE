//! tiktoken-style byte-pair encode.
//!
//! Given a piece of raw bytes (post byte-level encoding *into byte form*
//! — see note below) and a `Vocab`, emit the sequence of BPE token IDs
//! that match the canonical HF/tiktoken merge order.
//!
//! Note on representation: in HF/GPT-2-style BPE the "bytes" we work
//! over are UTF-8 bytes of the **byte-level-rendered string** (so a
//! single space byte 0x20 becomes the UTF-8 of `Ġ` which is two bytes:
//! 0xC4 0xA0). The vocab and merges in `tokenizer.json` are likewise
//! keyed by the byte sequences of byte-level-rendered chunks. We
//! therefore feed byte_pair_encode the **UTF-8 bytes of the byte-level
//! string of the input piece**, NOT the raw input bytes.
//!
//! The merge-rank lookup uses `Vocab::encoder` directly: the BPE
//! trainer assigns IDs in training order, so token ID order matches
//! merge-application order. Lower id ⇒ earlier-trained merge ⇒ applied
//! first. This avoids carrying a separate `(left, right) → rank` table
//! and avoids any per-call `Vec<u8>` allocation for hash keys.

use crate::vocab::Vocab;

/// Encode one piece (UTF-8 bytes of the byte-level-rendered chunk)
/// into a sequence of BPE token IDs, appending to `out`.
pub fn byte_pair_encode(piece: &[u8], vocab: &Vocab, out: &mut Vec<u32>) {
    let n = piece.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        // Single byte — must be in the initial alphabet.
        out.push(*vocab.encoder.get(piece).unwrap_or_else(|| {
            // Fallback only if the alphabet doesn't cover all 256 bytes,
            // which would be a vocab-load bug. Panic-with-context.
            panic!("single byte 0x{:02X} missing from vocab", piece[0]);
        }));
        return;
    }

    // `parts[i].0` is the byte-position where part i starts. The piece
    // is conceptually split as `piece[parts[0].0..parts[1].0]`,
    // `piece[parts[1].0..parts[2].0]`, … `piece[parts[n-1].0..n]`.
    // `parts[i].1` is the merge rank of the pair *starting* at part i
    // (i.e. piece[parts[i].0..parts[i+2].0]), or u32::MAX if no merge.
    let mut parts: Vec<(usize, u32)> = Vec::with_capacity(n + 1);
    for i in 0..n {
        parts.push((i, u32::MAX));
    }
    parts.push((n, u32::MAX)); // sentinel

    // Seed initial ranks for every adjacent pair.
    for i in 0..parts.len() - 2 {
        parts[i].1 = pair_rank(piece, &parts, i, vocab);
    }

    loop {
        // Find the part with the lowest rank.
        let mut min_rank = u32::MAX;
        let mut min_idx: Option<usize> = None;
        for (i, &(_, r)) in parts.iter().enumerate().take(parts.len() - 1) {
            if r < min_rank {
                min_rank = r;
                min_idx = Some(i);
            }
        }
        let i = match min_idx {
            Some(i) if min_rank != u32::MAX => i,
            _ => break,
        };

        // Merge: consume part i+1. The merged piece is now part i.
        parts.remove(i + 1);

        // Recompute the rank at i (new pair: merged ++ what was i+2 → now i+1).
        parts[i].1 = pair_rank(piece, &parts, i, vocab);
        // And the rank at i-1 (its right side just changed).
        if i > 0 {
            let im1 = i - 1;
            parts[im1].1 = pair_rank(piece, &parts, im1, vocab);
        }
    }

    // Emit one token per remaining slice.
    for w in parts.windows(2) {
        let bytes = &piece[w[0].0..w[1].0];
        match vocab.encoder.get(bytes) {
            Some(&id) => out.push(id),
            None => {
                // Should never happen: every BPE merge that was applied
                // exists in encoder by construction. Defensive fallback:
                // emit each byte individually so we don't return wrong IDs.
                for &b in bytes {
                    out.push(*vocab.encoder.get(&[b][..]).expect("byte in alphabet"));
                }
            }
        }
    }
}

/// Lookup the merge rank of the pair `parts[i] ++ parts[i+1]`. Returns
/// `u32::MAX` if the merged sequence isn't a vocab entry.
#[inline]
fn pair_rank(piece: &[u8], parts: &[(usize, u32)], i: usize, vocab: &Vocab) -> u32 {
    if i + 2 >= parts.len() {
        return u32::MAX;
    }
    let start = parts[i].0;
    let end = parts[i + 2].0;
    vocab
        .encoder
        .get(&piece[start..end])
        .copied()
        .unwrap_or(u32::MAX)
}

/// Encode an arbitrary byte slice. Applies the GPT-2 byte-level mapping
/// first (matches HF's ByteLevel pre-tokenizer), then BPE-encodes the
/// resulting byte-level UTF-8 byte sequence. The vocab is keyed by
/// byte-level UTF-8 bytes (see [`crate::vocab`]).
pub fn encode_raw_bytes(raw: &[u8], vocab: &Vocab, out: &mut Vec<u32>) {
    let mut bl_buf: Vec<u8> = Vec::with_capacity(raw.len() * 2);
    crate::bytelevel::encode_bytes_into(raw, &mut bl_buf);
    byte_pair_encode(&bl_buf, vocab, out);
}

/// Same as [`encode_raw_bytes`] but reuses a caller-owned `Vec<u8>`
/// scratch buffer for the byte-level intermediate to avoid per-call
/// allocation in tight loops.
pub fn encode_raw_bytes_with_buf(
    raw: &[u8],
    vocab: &Vocab,
    bl_buf: &mut Vec<u8>,
    out: &mut Vec<u32>,
) {
    crate::bytelevel::encode_bytes_into(raw, bl_buf);
    byte_pair_encode(bl_buf, vocab, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocab;

    fn vocab() -> Vocab {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../model/tokenizer.json")
            .canonicalize()
            .expect("model exists");
        Vocab::from_tokenizer_json(&p).expect("loads")
    }

    #[test]
    fn empty_input_emits_nothing() {
        let v = vocab();
        let mut out = Vec::new();
        byte_pair_encode(&[], &v, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn single_byte_emits_one_id() {
        let v = vocab();
        // "a" in byte-level form is just "a" (it's printable ASCII).
        let mut out = Vec::new();
        byte_pair_encode(b"a", &v, &mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn ascii_word_merges() {
        let v = vocab();
        // "def" → byte-level "def" → some BPE merge that's in the vocab.
        let mut out = Vec::new();
        encode_raw_bytes(b"def", &v, &mut out);
        assert!(!out.is_empty(), "must produce at least one token");
        // It should be quite short — "def" is a common BPE merge.
        assert!(out.len() <= 3, "got {} tokens", out.len());
    }

    #[test]
    fn parity_with_hf_on_known_fixtures() {
        // Expected outputs captured from HF's `Tokenizer.from_file(...)` on
        // model/tokenizer.json (V2). If the cute-bpe algorithm or the vocab
        // loader desynchronises, this test catches it immediately.
        let v = vocab();
        let fixtures: &[(&[u8], &[u32])] = &[
            (b"def", &[50348]),
            (b"hello", &[56183]),
            (b" return", &[50398]),
            (b"abc123", &[55018, 55296]),
            (b"self", &[50293]),
            (b"Hello", &[57992]),
        ];
        for (input, expected) in fixtures {
            let mut got = Vec::new();
            encode_raw_bytes(input, &v, &mut got);
            assert_eq!(
                &got[..],
                *expected,
                "encode({:?}): expected {:?}, got {:?}",
                std::str::from_utf8(input).unwrap_or("?"),
                expected,
                got
            );
        }
    }
}
