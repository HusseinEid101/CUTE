//! GPT-2 byte-level encoding.
//!
//! Maps the 256 byte values 0x00..=0xFF to a 256-codepoint Unicode subset
//! that avoids whitespace + control characters, so byte sequences can pass
//! through whitespace-aware BPE pre-tokenizers without losing information.
//! Used by GPT-2, cl100k, and our HF-trained `tokenizer.json` alike.
//!
//! Population is identical to `tokenizers/src/pre_tokenizers/byte_level.rs`
//! (HuggingFace) and OpenAI's `tiktoken` reference. We embed the table
//! here as a constant so we never depend on either crate at runtime.

use once_cell::sync::Lazy;

/// `BYTE_TO_CHAR[b]` is the Unicode codepoint used to represent byte `b`
/// in the byte-level vocabulary. Always a 1-codepoint mapping.
pub static BYTE_TO_CHAR: Lazy<[char; 256]> = Lazy::new(build_byte_to_char);

/// Inverse of [`BYTE_TO_CHAR`] — look up a codepoint, get the original byte.
pub static CHAR_TO_BYTE: Lazy<rustc_hash::FxHashMap<char, u8>> = Lazy::new(|| {
    BYTE_TO_CHAR
        .iter()
        .enumerate()
        .map(|(b, &c)| (c, b as u8))
        .collect()
});

fn build_byte_to_char() -> [char; 256] {
    // Bytes that print "nicely" stay as themselves.
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs: Vec<u32> = bs.clone();
    // Everything else maps to a codepoint past 0xFF, sequentially.
    let mut n: u32 = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        table[*b as usize] = char::from_u32(*c).expect("valid codepoint");
    }
    table
}

/// Encode raw bytes as a byte-level string.
#[inline]
pub fn encode_bytes(bytes: &[u8]) -> String {
    let table = &*BYTE_TO_CHAR;
    let mut s = String::with_capacity(bytes.len());
    for &b in bytes {
        s.push(table[b as usize]);
    }
    s
}

/// Encode raw bytes as the UTF-8 byte representation of the byte-level
/// codepoints. Skips the `String` intermediate and writes directly into
/// a `Vec<u8>` — saves one allocation in the encode hot path.
#[inline]
pub fn encode_bytes_into(bytes: &[u8], out: &mut Vec<u8>) {
    let table = &*BYTE_TO_CHAR;
    out.clear();
    out.reserve(bytes.len() * 2);
    let mut buf = [0u8; 4];
    for &b in bytes {
        let s = table[b as usize].encode_utf8(&mut buf);
        out.extend_from_slice(s.as_bytes());
    }
}

/// Inverse: byte-level string back to raw bytes.
#[inline]
pub fn decode_bytes(s: &str) -> Vec<u8> {
    let map = &*CHAR_TO_BYTE;
    let mut out = Vec::with_capacity(s.len());
    for c in s.chars() {
        if let Some(&b) = map.get(&c) {
            out.push(b);
        } else {
            // Non-byte-level char — surface it as UTF-8 bytes. This
            // happens for PUA chars and AddedTokens; they have their
            // own atomic IDs handled at the encode/decode layer above.
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_a_bijection() {
        let to = &*BYTE_TO_CHAR;
        let from = &*CHAR_TO_BYTE;
        for b in 0u8..=255 {
            let c = to[b as usize];
            assert_eq!(from[&c], b);
        }
    }

    #[test]
    fn space_maps_to_capital_g_dot() {
        // Ġ (U+0120) is the GPT-2 byte-level rendering of space.
        assert_eq!(BYTE_TO_CHAR[b' ' as usize], '\u{120}');
    }

    #[test]
    fn newline_maps_to_capital_c_dot() {
        // Ċ (U+010A) is the byte-level rendering of newline (0x0A).
        assert_eq!(BYTE_TO_CHAR[b'\n' as usize], '\u{10A}');
    }

    #[test]
    fn encode_decode_roundtrip() {
        let bytes = b"def hello():\n    return 42\n";
        let enc = encode_bytes(bytes);
        let dec = decode_bytes(&enc);
        assert_eq!(dec, bytes);
    }
}
