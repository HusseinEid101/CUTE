use ahash::AHashMap;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cute_core::frequency::count_in_text;

const SAMPLE: &str = include_str!("sample_code.txt");

fn bench_count_in_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("frequency");
    group.throughput(criterion::Throughput::Bytes(SAMPLE.len() as u64));
    group.bench_function("count_in_text/5KB/no_boost", |b| {
        b.iter(|| {
            let mut counter: AHashMap<String, u64> = AHashMap::new();
            let mut boost: AHashMap<String, f64> = AHashMap::new();
            count_in_text(black_box(SAMPLE), &mut counter, &mut boost, 0.0, 50);
            black_box((counter, boost))
        });
    });
    group.bench_function("count_in_text/5KB/boost_0.3", |b| {
        b.iter(|| {
            let mut counter: AHashMap<String, u64> = AHashMap::new();
            let mut boost: AHashMap<String, f64> = AHashMap::new();
            count_in_text(black_box(SAMPLE), &mut counter, &mut boost, 0.3, 50);
            black_box((counter, boost))
        });
    });
    group.finish();
}

criterion_group!(benches, bench_count_in_text);
criterion_main!(benches);
