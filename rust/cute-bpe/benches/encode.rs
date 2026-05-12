//! End-to-end encode benchmark on the bundled production model.
//!
//! Measures `CuteBpe::encode` on a small Python sample (~1.7 KB) and a
//! large Python sample (~31 KB) so we can track regressions in the
//! short-prompt p50 (most common workload) and the long-prompt p50
//! (where the BPE merge loop dominates).
//!
//! Run with:
//!   cargo bench -p cute-bpe --bench encode
//!
//! Outputs HTML reports under `target/criterion/`.

use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use cute_bpe::CuteBpe;

fn model_paths() -> (PathBuf, PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../model")
        .canonicalize()
        .expect("model dir resolves");
    (root.join("tokenizer.json"), root.join("cute_mapping.json"))
}

const SMALL_SAMPLE: &str = r#"
import json
from pathlib import Path

def load_config(path: Path) -> dict:
    """Load JSON config from disk."""
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)

class TokenizerConfig:
    def __init__(self, vocab_size: int = 200_000, seed: int = 42):
        self.vocab_size = vocab_size
        self.seed = seed
        self._cache: dict[str, int] = {}

    def get(self, key: str, default: int = 0) -> int:
        return self._cache.get(key, default)

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self._cache[k] = v

if __name__ == "__main__":
    cfg = TokenizerConfig(vocab_size=200_000)
    cfg.update(workers=8, seed=1234)
    print(cfg.get("workers"))
"#;

fn large_sample() -> String {
    // Concatenate the small sample ~18× to get ~31 KB of representative code.
    SMALL_SAMPLE.repeat(18)
}

fn bench_encode_short(c: &mut Criterion) {
    let (t, m) = model_paths();
    let bpe = CuteBpe::from_paths(&t, &m).expect("loads");
    let sample = SMALL_SAMPLE;
    let n_bytes = sample.len() as u64;

    let mut group = c.benchmark_group("encode_short");
    group.throughput(Throughput::Bytes(n_bytes));
    group.bench_function("cute-bpe/short", |b| {
        b.iter_batched(|| sample, |s| bpe.encode(s), BatchSize::SmallInput);
    });
    group.finish();
}

fn bench_encode_long(c: &mut Criterion) {
    let (t, m) = model_paths();
    let bpe = CuteBpe::from_paths(&t, &m).expect("loads");
    let sample = large_sample();
    let n_bytes = sample.len() as u64;

    let mut group = c.benchmark_group("encode_long");
    group.throughput(Throughput::Bytes(n_bytes));
    group.bench_function("cute-bpe/long", |b| {
        b.iter_batched(|| sample.clone(), |s| bpe.encode(&s), BatchSize::SmallInput);
    });
    group.finish();
}

fn bench_encode_ascii_only(c: &mut Criterion) {
    // Pure ASCII keywords-heavy snippet (no leading whitespace blocks),
    // representative of dense API surface.
    let (t, m) = model_paths();
    let bpe = CuteBpe::from_paths(&t, &m).expect("loads");
    let sample = "def fast_encode(self,text):return list(self._cute_bpe.encode(text))".repeat(8);
    let n_bytes = sample.len() as u64;

    let mut group = c.benchmark_group("encode_ascii_only");
    group.throughput(Throughput::Bytes(n_bytes));
    group.bench_function("cute-bpe/ascii_only", |b| {
        b.iter_batched(|| sample.clone(), |s| bpe.encode(&s), BatchSize::SmallInput);
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_encode_short,
    bench_encode_long,
    bench_encode_ascii_only
);
criterion_main!(benches);
