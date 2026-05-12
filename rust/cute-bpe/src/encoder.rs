//! tiktoken-style byte-pair encode.
//!
//! Given a piece of raw bytes (post byte-level encoding *into byte form*
//! — see note below) and a `Vocab`, emit the sequence of BPE token IDs
//! that match the canonical HF/tiktoken merge order.
//!
//! Two code paths:
//! - **Short pieces** (`n <= SMALL_PIECE_THRESHOLD = 64`): linear-scan
//!   for min rank in a stack-allocated `SmallVec<[(usize, u32); 80]>`.
//!   Cheap initial cost dominates for short pieces; the O(n²) merge
//!   loop is fine when n is tiny.
//! - **Long pieces** (`n > 64`): doubly-linked-list of parts plus
//!   `BinaryHeap<(Reverse<u32>, usize)>` for min-rank lookups. Stale
//!   heap entries are skipped lazily. Asymptotically O(n log n).
//!
//! Note on representation: in HF/GPT-2-style BPE the "bytes" we work
//! over are UTF-8 bytes of the **byte-level-rendered string** (so a
//! single space byte 0x20 becomes the UTF-8 of `Ġ` which is two bytes:
//! 0xC4 0xA0). The vocab and merges in `tokenizer.json` are likewise
//! keyed by the byte sequences of byte-level-rendered chunks.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use smallvec::SmallVec;

use crate::vocab::Vocab;

/// Threshold above which we switch to the heap-based BPE path. The
/// short path is `SmallVec` + linear-scan-for-min: zero allocation
/// for any `n <= SMALL_PARTS_INLINE - 1` and very low constant
/// factor up to ~64 bytes. Beyond that, the O(n²) merge loop starts
/// to dominate and the heap path's O(n log n) wins.
const SMALL_PIECE_THRESHOLD: usize = 64;
const SMALL_PARTS_INLINE: usize = 80;

thread_local! {
    /// Reusable scratch heap for the long-piece BPE path. Cleared
    /// at the start of each `byte_pair_encode_heap` call; kept alive
    /// across calls so we amortize allocation.
    ///
    /// Tuple layout: `(Reverse<rank>, Reverse<idx>)`. The first key
    /// gives us a min-by-rank ordering; the second resolves
    /// equal-rank pairs in favour of the **leftmost** (smallest) idx,
    /// matching tiktoken's left-to-right tie-break. (Without
    /// `Reverse(idx)` the natural tuple ordering would pop the
    /// largest idx first, producing a different — and incorrect —
    /// merge sequence relative to HF tokenizers.)
    static HEAP_SCRATCH: RefCell<BinaryHeap<(Reverse<u32>, Reverse<usize>)>> =
        RefCell::new(BinaryHeap::with_capacity(256));

    /// Reusable scratch for the long-piece linked-list parts.
    /// Layout: (byte_start, prev_idx, next_idx, rank). `next_idx == usize::MAX`
    /// marks the tail sentinel.
    static LINK_SCRATCH: RefCell<Vec<(usize, usize, usize, u32)>> =
        RefCell::new(Vec::with_capacity(512));
}

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
            panic!("single byte 0x{:02X} missing from vocab", piece[0]);
        }));
        return;
    }
    if n <= SMALL_PIECE_THRESHOLD {
        byte_pair_encode_small(piece, vocab, out);
    } else {
        byte_pair_encode_heap(piece, vocab, out);
    }
}

/// Linear-scan BPE for short pieces. Uses a `SmallVec` so the part
/// table lives on the stack (no allocation) for any `n <= 79`.
#[inline]
fn byte_pair_encode_small(piece: &[u8], vocab: &Vocab, out: &mut Vec<u32>) {
    let n = piece.len();
    // `parts[i].0` is the byte-position where part i starts.
    // `parts[i].1` is the merge rank of the pair *starting* at part i
    // (i.e. piece[parts[i].0..parts[i+2].0]), or u32::MAX if no merge.
    let mut parts: SmallVec<[(usize, u32); SMALL_PARTS_INLINE]> = SmallVec::with_capacity(n + 1);
    for i in 0..n {
        parts.push((i, u32::MAX));
    }
    parts.push((n, u32::MAX));

    // Seed initial ranks.
    for i in 0..parts.len().saturating_sub(2) {
        parts[i].1 = pair_rank_slice(piece, parts[i].0, parts[i + 2].0, vocab);
    }

    loop {
        // Find part with lowest rank.
        let mut min_rank = u32::MAX;
        let mut min_idx: usize = usize::MAX;
        let scan_len = parts.len().saturating_sub(1);
        for (i, &(_, r)) in parts.iter().take(scan_len).enumerate() {
            if r < min_rank {
                min_rank = r;
                min_idx = i;
            }
        }
        if min_idx == usize::MAX || min_rank == u32::MAX {
            break;
        }
        let i = min_idx;

        // Consume part i+1. Merged piece is now part i.
        parts.remove(i + 1);

        // Recompute rank at i (new right neighbor).
        if i + 2 < parts.len() {
            parts[i].1 = pair_rank_slice(piece, parts[i].0, parts[i + 2].0, vocab);
        } else {
            parts[i].1 = u32::MAX;
        }
        // And the rank at i-1 (its right side just changed).
        if i > 0 && i + 1 < parts.len() {
            let im1 = i - 1;
            parts[im1].1 = pair_rank_slice(piece, parts[im1].0, parts[i + 1].0, vocab);
        }
    }

    // Emit one token per remaining slice.
    for w in parts.windows(2) {
        emit_slice(piece, w[0].0, w[1].0, vocab, out);
    }
}

/// Heap + doubly-linked-list BPE for long pieces. Used when the
/// naive O(n²) of `byte_pair_encode_small` would start to dominate.
fn byte_pair_encode_heap(piece: &[u8], vocab: &Vocab, out: &mut Vec<u32>) {
    let n = piece.len();
    HEAP_SCRATCH.with(|heap_cell| {
        LINK_SCRATCH.with(|parts_cell| {
            let mut heap = heap_cell.borrow_mut();
            let mut parts = parts_cell.borrow_mut();
            heap.clear();
            parts.clear();
            parts.reserve(n + 1);

            // Build linked list: parts[i] = (byte_start, prev, next, rank).
            // Index n is the tail sentinel (no part, just an end marker).
            for i in 0..n {
                let prev = if i == 0 { usize::MAX } else { i - 1 };
                let next = i + 1;
                parts.push((i, prev, next, u32::MAX));
            }
            parts.push((n, n.saturating_sub(1), usize::MAX, u32::MAX)); // sentinel

            // Seed ranks + heap.
            for i in 0..n.saturating_sub(1) {
                let next = parts[i].2;
                let next_next = parts[next].2;
                if next_next == usize::MAX {
                    break;
                }
                let r = pair_rank_slice(piece, parts[i].0, parts[next_next].0, vocab);
                parts[i].3 = r;
                if r != u32::MAX {
                    heap.push((Reverse(r), Reverse(i)));
                }
            }
            // Last pre-sentinel part also needs a rank if it forms a pair.
            // (Handled by the loop above; nothing extra to do.)

            while let Some((Reverse(rank), Reverse(i))) = heap.pop() {
                // Skip stale entries.
                if i >= parts.len() {
                    continue;
                }
                if parts[i].3 != rank {
                    continue;
                }
                let next = parts[i].2;
                if next == usize::MAX || next == n {
                    continue;
                }
                let next_next = parts[next].2;
                if next_next == usize::MAX {
                    continue;
                }

                // Merge `next` into `i`: i now spans piece[parts[i].0 .. parts[next_next].0].
                parts[i].2 = next_next;
                if next_next != usize::MAX {
                    parts[next_next].1 = i;
                }
                // Mark `next` removed by leaving rank invalid (it's no longer
                // reachable from the linked list).
                parts[next].3 = u32::MAX;

                // Recompute rank at i.
                let new_next = parts[i].2;
                if new_next != usize::MAX && new_next != n {
                    let new_next_next = parts[new_next].2;
                    if new_next_next != usize::MAX {
                        let r = pair_rank_slice(piece, parts[i].0, parts[new_next_next].0, vocab);
                        parts[i].3 = r;
                        if r != u32::MAX {
                            heap.push((Reverse(r), Reverse(i)));
                        }
                    } else {
                        parts[i].3 = u32::MAX;
                    }
                } else {
                    parts[i].3 = u32::MAX;
                }

                // Recompute rank at i.prev (its right neighbor just grew).
                let prev = parts[i].1;
                if prev != usize::MAX {
                    let prev_next = parts[prev].2; // == i
                    let prev_next_next = parts[prev_next].2;
                    if prev_next_next != usize::MAX {
                        let r =
                            pair_rank_slice(piece, parts[prev].0, parts[prev_next_next].0, vocab);
                        parts[prev].3 = r;
                        if r != u32::MAX {
                            heap.push((Reverse(r), Reverse(prev)));
                        }
                    } else {
                        parts[prev].3 = u32::MAX;
                    }
                }
            }

            // Walk the linked list from the head, emitting one token per node.
            let mut i: usize = 0;
            while i < n {
                let end = parts[i].2;
                let end_byte = if end == usize::MAX || end == n {
                    n
                } else {
                    parts[end].0
                };
                emit_slice(piece, parts[i].0, end_byte, vocab, out);
                if end == usize::MAX || end == n {
                    break;
                }
                i = end;
            }
        });
    });
}

/// Look up the BPE id for `piece[start..end]` and push to `out`. Falls
/// back to byte-by-byte emission only if the slice somehow isn't in the
/// encoder (shouldn't happen for any merge actually applied).
#[inline]
fn emit_slice(piece: &[u8], start: usize, end: usize, vocab: &Vocab, out: &mut Vec<u32>) {
    let bytes = &piece[start..end];
    match vocab.encoder.get(bytes) {
        Some(&id) => out.push(id),
        None => {
            for &b in bytes {
                out.push(*vocab.encoder.get(&[b][..]).expect("byte in alphabet"));
            }
        }
    }
}

/// Look up the merge rank of `piece[start..end]`. Returns `u32::MAX` if
/// the merged sequence isn't a vocab entry.
#[inline]
fn pair_rank_slice(piece: &[u8], start: usize, end: usize, vocab: &Vocab) -> u32 {
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
        let mut out = Vec::new();
        byte_pair_encode(b"a", &v, &mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn ascii_word_merges() {
        let v = vocab();
        let mut out = Vec::new();
        encode_raw_bytes(b"def", &v, &mut out);
        assert!(!out.is_empty(), "must produce at least one token");
        assert!(out.len() <= 3, "got {} tokens", out.len());
    }

    #[test]
    fn parity_with_hf_on_known_fixtures() {
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

    #[test]
    fn long_piece_uses_heap_path() {
        // A 200-char ASCII run should trigger the heap path.
        let v = vocab();
        let raw: Vec<u8> = b"abcdefghij".iter().cycle().take(200).copied().collect();
        let mut small_out = Vec::new();
        encode_raw_bytes(&raw, &v, &mut small_out);
        assert!(!small_out.is_empty());
    }

    #[test]
    fn small_and_heap_paths_agree() {
        // Construct a piece just over the threshold and verify both paths
        // produce identical IDs (by temporarily forcing each).
        let v = vocab();
        let raw: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
        let mut bl = Vec::new();
        crate::bytelevel::encode_bytes_into(raw, &mut bl);
        let mut small_out = Vec::new();
        let mut heap_out = Vec::new();
        // Force-call each path directly.
        byte_pair_encode_small(&bl, &v, &mut small_out);
        byte_pair_encode_heap(&bl, &v, &mut heap_out);
        assert_eq!(small_out, heap_out, "small vs heap path disagree");
    }
}
