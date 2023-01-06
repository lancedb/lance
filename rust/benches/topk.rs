use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};

use lance::index::ann::sort::find_topk;
use lance::tests::generate_random_array;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("TopK");
    let compare  = |l: &f32, r: &f32| l.total_cmp(r);
    group.significance_level(0.1).sample_size(10).measurement_time(Duration::new(300, 0));
    group.bench_function("Top10From1M", |b| {
        let topk = 10;
        let card = 1000000;
        let mut arr: Vec<f32> = generate_random_array(card).values().to_vec();
        let mut indices: Vec<u64> = (0..card as u64).collect();
        b.iter(|| find_topk(&mut arr, &mut indices, topk, &compare));
    });
    group.bench_function("Top100From1M", |b| {
        let topk = 100;
        let card = 1000000;
        let mut arr: Vec<f32> = generate_random_array(card).values().to_vec();
        let mut indices: Vec<u64> = (0..card as u64).collect();
        b.iter(|| find_topk(&mut arr, &mut indices, topk, &compare));
    });
    group.bench_function("Top1000From1M", |b| {
        let topk = 1000;
        let card = 1000000;
        let mut arr: Vec<f32> = generate_random_array(card).values().to_vec();
        let mut indices: Vec<u64> = (0..card as u64).collect();
        b.iter(|| find_topk(&mut arr, &mut indices, topk, &compare));
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
