use ahash::AHashMap;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cute_core::pretok::{cute_split_text, pretokenize_to_string, WordToPua};

const SAMPLE: &str = include_str!("sample_code.txt");

fn build_mapping() -> WordToPua {
    let mut m: WordToPua = AHashMap::new();
    let pairs: &[(&str, char)] = &[
        ("def", '\u{F0001}'),
        ("return", '\u{F0002}'),
        ("class", '\u{F0003}'),
        ("self", '\u{F0004}'),
        ("import", '\u{F0005}'),
        ("from", '\u{F0006}'),
        ("if", '\u{F0007}'),
        ("else", '\u{F0008}'),
        ("for", '\u{F0009}'),
        ("in", '\u{F000A}'),
        ("calculate", '\u{F000B}'),
        ("total", '\u{F000C}'),
        ("user", '\u{F000D}'),
        ("id", '\u{F000E}'),
    ];
    for (w, c) in pairs {
        m.insert((*w).to_string(), *c);
    }
    m
}

fn bench_pretokenize_to_string(c: &mut Criterion) {
    let mapping = build_mapping();
    let mut group = c.benchmark_group("pretok");
    group.throughput(criterion::Throughput::Bytes(SAMPLE.len() as u64));
    group.bench_function("pretokenize_to_string/5KB", |b| {
        b.iter(|| pretokenize_to_string(black_box(SAMPLE), black_box(&mapping)));
    });
    group.bench_function("cute_split_text/5KB", |b| {
        b.iter(|| cute_split_text(black_box(SAMPLE), black_box(&mapping)));
    });
    group.finish();
}

criterion_group!(benches, bench_pretokenize_to_string);
criterion_main!(benches);
