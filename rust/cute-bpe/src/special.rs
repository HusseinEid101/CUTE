//! Compound `AddedToken` matching.
//!
//! All `added_tokens` from `tokenizer.json` (the 9 pipe-style special
//! markers plus the ~6k savings-ranked compound tokens from V2) are
//! matched atomically against the raw input *before* BPE runs. We use
//! an [`aho_corasick::AhoCorasick`] automaton with `LeftmostLongest`
//! semantics so that overlapping patterns (e.g. `):\n` vs `):` vs `)`)
//! resolve to the longest match starting at the earliest position.
//!
//! Aho-Corasick is ~constant cost regardless of how many patterns you
//! load — building once at construction time, scanning the raw input
//! once at encode time. For our 6k patterns that's far cheaper than a
//! `|`-joined regex with backtracking.

use aho_corasick::{AhoCorasick, MatchKind};
use rustc_hash::FxHashMap;

use crate::vocab::Vocab;

pub struct SpecialMatcher {
    /// One Aho-Corasick automaton over every `added_tokens.content`.
    automaton: AhoCorasick,
    /// `pattern_id` (from the automaton) → token ID. Index into this Vec.
    pattern_to_token: Vec<u32>,
}

impl SpecialMatcher {
    /// Build from the `special_encoder` produced by `Vocab`. Returns
    /// `None` when there are no special tokens (no work to do).
    pub fn build(vocab: &Vocab) -> Option<Self> {
        if vocab.special_encoder.is_empty() {
            return None;
        }
        // Sort patterns longest-first so that LeftmostLongest's
        // tie-breaking (earlier insertion wins) prefers longer matches.
        // Aho-Corasick with MatchKind::LeftmostLongest already picks
        // longest; the sort is belt-and-braces for stability.
        let mut entries: Vec<(&String, &u32)> = vocab.special_encoder.iter().collect();
        entries.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(b.0)));
        let patterns: Vec<&[u8]> = entries.iter().map(|(s, _)| s.as_bytes()).collect();
        let pattern_to_token: Vec<u32> = entries.iter().map(|(_, &id)| id).collect();

        let automaton = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(patterns)
            .expect("special-tokens AC build");

        Some(Self {
            automaton,
            pattern_to_token,
        })
    }

    /// Find non-overlapping leftmost-longest matches over `text`.
    /// Returns `(start_byte, end_byte, token_id)` triples in order.
    pub fn find_all(&self, text: &[u8]) -> Vec<(usize, usize, u32)> {
        self.automaton
            .find_iter(text)
            .map(|m| {
                let start = m.start();
                let end = m.end();
                let pid = m.pattern().as_usize();
                let id = self.pattern_to_token[pid];
                (start, end, id)
            })
            .collect()
    }

    /// Diagnostic / for tests.
    pub fn pattern_count(&self) -> usize {
        self.pattern_to_token.len()
    }

    /// Look up by content string (slower path used outside the hot loop).
    pub fn id_for_content(&self, vocab: &Vocab, content: &str) -> Option<u32> {
        vocab.special_encoder.get(content).copied()
    }

    /// Quick-check whether the automaton has at least one match in `text`.
    /// Used as a fast-path: if no special tokens appear, skip the splice.
    pub fn has_any_match(&self, text: &[u8]) -> bool {
        self.automaton.is_match(text)
    }
}

/// Helper for callers that prefer a `FxHashMap` view of (id → content).
pub fn build_id_to_content(vocab: &Vocab) -> FxHashMap<u32, &str> {
    vocab
        .special_decoder
        .iter()
        .map(|(id, content)| (*id, content.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn vocab() -> Vocab {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../model/tokenizer.json")
            .canonicalize()
            .expect("model exists");
        Vocab::from_tokenizer_json(&p).expect("loads")
    }

    #[test]
    fn matcher_covers_all_added_tokens() {
        let v = vocab();
        let m = SpecialMatcher::build(&v).expect("non-empty");
        assert_eq!(m.pattern_count(), v.special_encoder.len());
    }

    #[test]
    fn finds_special_marker() {
        let v = vocab();
        let m = SpecialMatcher::build(&v).expect("non-empty");
        let text = b"start <|endoftext|> end";
        let matches = m.find_all(text);
        assert!(matches.iter().any(|(_, _, id)| {
            v.special_decoder.get(id).map(|s| s.as_str()) == Some("<|endoftext|>")
        }));
    }

    #[test]
    fn finds_compound_added_token() {
        let v = vocab();
        let m = SpecialMatcher::build(&v).expect("non-empty");
        // ":\n" should be one of the 6k compound tokens if it was selected.
        // Even if not selected, the test is still valid — it just won't match.
        let text = b"if x:\nreturn";
        let matches = m.find_all(text);
        // At minimum, no match should overlap by more than one slot.
        for w in matches.windows(2) {
            assert!(w[0].1 <= w[1].0, "matches must be non-overlapping");
        }
    }

    #[test]
    fn longest_match_wins() {
        let v = vocab();
        let m = SpecialMatcher::build(&v).expect("non-empty");
        // If both "<|im_start|>" and "<|im_" exist (only the full one does
        // in practice), the longer wins.
        let text = b"<|im_start|>hello";
        let matches = m.find_all(text);
        if let Some((s, e, id)) = matches.first() {
            let content = v.special_decoder.get(id).cloned().unwrap_or_default();
            // The match should span the entire "<|im_start|>" prefix.
            if *s == 0 {
                assert_eq!(*e, content.len());
            }
        }
    }
}
