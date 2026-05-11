use ahash::AHashMap;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cute_core::decode::reverse_pua_substitute;
use cute_core::pretok::{pretokenize_to_string, PuaToWord, WordToPua};

const SAMPLE: &str = include_str!("sample_code.txt");

fn build_mappings() -> (WordToPua, PuaToWord) {
    let pairs: &[(&str, char)] = &[
        ("def", '\u{F0001}'),
        ("return", '\u{F0002}'),
        ("class", '\u{F0003}'),
        ("self", '\u{F0004}'),
    ];
    let mut wtp: WordToPua = AHashMap::new();
    let mut ptw: PuaToWord = AHashMap::new();
    for (w, c) in pairs {
        wtp.insert((*w).to_string(), *c);
        ptw.insert(*c, (*w).to_string());
    }
    (wtp, ptw)
}

fn bench_reverse(c: &mut Criterion) {
    let (wtp, ptw) = build_mappings();
    let pre = pretokenize_to_string(SAMPLE, &wtp);
    let mut group = c.benchmark_group("decode");
    group.throughput(criterion::Throughput::Bytes(pre.len() as u64));
    group.bench_function("reverse_pua_substitute/5KB", |b| {
        b.iter(|| reverse_pua_substitute(black_box(&pre), black_box(&ptw)));
    });
    group.finish();
}

criterion_group!(benches, bench_reverse);
criterion_main!(benches);
