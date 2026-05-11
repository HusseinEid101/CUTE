//! Frequency counting for BPE training corpus.
//!
//! Single-text [`count_in_text`] mirrors `frequency.py:24`.
//! Multi-shard parallel [`count_frequencies`] replaces the Python
//! `ProcessPoolExecutor` with Rayon work-stealing, eliminating IPC overhead.
//!
//! Output format matches the Python `Counter[str]` shape: a dict mapping
//! token → integer count, with the boost-weight ceiling applied so any
//! nonzero accumulation produces at least 1 count (mirrors fix #4 from
//! the build plan referenced in `frequency.py` docstring).

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use ahash::AHashMap;
use flate2::read::GzDecoder;
use rayon::prelude::*;

use crate::identifier;
use crate::tokens;

/// Per-text counter accumulator.
///
/// `counter` and `boost_acc` are mutated in place. The Python wrapper
/// converts them to a `Counter[str]` and `defaultdict[str, float]` at the
/// FFI boundary.
pub fn count_in_text(
    text: &str,
    counter: &mut AHashMap<String, u64>,
    boost_acc: &mut AHashMap<String, f64>,
    boost_weight: f64,
    max_token_len: usize,
) {
    let mut buf: Vec<String> = Vec::new();
    for tok in tokens::iter_token_strings(text) {
        // Codepoint-count parity with Python's `len(tok)`. Bail out early
        // when we exceed the limit instead of counting all characters.
        if codepoint_count_exceeds(tok, max_token_len) {
            continue;
        }
        *counter.entry(tok.to_string()).or_insert(0) += 1;
        if boost_weight > 0.0 && identifier::is_identifier(tok) {
            buf.clear();
            identifier::extend_split_identifier(tok, &mut buf);
            for part in buf.iter() {
                if part != tok {
                    *boost_acc.entry(part.clone()).or_insert(0.0) += boost_weight;
                }
            }
        }
    }
}

#[inline]
fn codepoint_count_exceeds(s: &str, limit: usize) -> bool {
    let mut count = 0usize;
    for _ in s.chars() {
        count += 1;
        if count > limit {
            return true;
        }
    }
    false
}

/// Parallel shard iteration. Reads gzipped JSONL records of shape
/// `{"path": ..., "text": ..., "sha256": ...}`, accumulates per-shard
/// counters, then merges deterministically.
///
/// Returns `(counter, raw_token_count_for_diagnostics)`.
pub fn count_frequencies(
    shards: &[PathBuf],
    boost_weight: f64,
    max_token_len: usize,
) -> std::io::Result<HashMap<String, u64>> {
    if shards.is_empty() {
        return Ok(HashMap::new());
    }

    let per_shard: Result<Vec<_>, std::io::Error> = shards
        .par_iter()
        .map(|shard| count_one_shard(shard, boost_weight, max_token_len))
        .collect();
    let per_shard = per_shard?;

    // Deterministic merge: per_shard preserves shard input order.
    let mut total: AHashMap<String, u64> = AHashMap::new();
    let mut boost_total: AHashMap<String, f64> = AHashMap::new();
    for (counter, boost) in per_shard {
        for (k, v) in counter {
            *total.entry(k).or_insert(0) += v;
        }
        for (k, v) in boost {
            *boost_total.entry(k).or_insert(0.0) += v;
        }
    }
    for (tok, frac) in boost_total {
        let bonus = frac.ceil() as u64;
        if bonus > 0 {
            *total.entry(tok).or_insert(0) += bonus;
        }
    }

    Ok(total.into_iter().collect())
}

fn count_one_shard(
    shard: &Path,
    boost_weight: f64,
    max_token_len: usize,
) -> std::io::Result<(AHashMap<String, u64>, AHashMap<String, f64>)> {
    let file = std::fs::File::open(shard)?;
    let reader = BufReader::new(GzDecoder::new(file));

    let mut counter: AHashMap<String, u64> = AHashMap::new();
    let mut boost: AHashMap<String, f64> = AHashMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let val: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("shard JSON parse error: {e}"),
            )
        })?;
        if let Some(text) = val.get("text").and_then(|v| v.as_str()) {
            count_in_text(text, &mut counter, &mut boost, boost_weight, max_token_len);
        }
    }
    Ok((counter, boost))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_simple_text() {
        let mut c = AHashMap::new();
        let mut b = AHashMap::new();
        count_in_text("def foo def bar", &mut c, &mut b, 0.0, 50);
        assert_eq!(c.get("def").copied(), Some(2));
        assert_eq!(c.get("foo").copied(), Some(1));
        assert_eq!(c.get("bar").copied(), Some(1));
    }

    #[test]
    fn skips_oversized_tokens() {
        let mut c = AHashMap::new();
        let mut b = AHashMap::new();
        let long = "x".repeat(60);
        count_in_text(&long, &mut c, &mut b, 0.0, 50);
        // long token (60 codepoints > limit 50) should be skipped entirely.
        assert!(c.get(long.as_str()).is_none());
    }

    #[test]
    fn boost_accumulates_for_identifier_subparts() {
        let mut c = AHashMap::new();
        let mut b = AHashMap::new();
        count_in_text("calculateTotal myVar", &mut c, &mut b, 0.5, 50);
        // Sub-parts (calculate, Total, my, Var) get fractional boost
        assert!(b.get("calculate").is_some());
        assert!(b.get("Total").is_some());
        assert!(b.get("my").is_some());
        assert!(b.get("Var").is_some());
    }
}
