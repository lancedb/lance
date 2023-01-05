use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};

use lance::index::ann::distance::euclidean_distance;
use lance::tests::{generate_random_array, generate_random_matrix};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("L2 distance");
    group.significance_level(0.1).sample_size(10).measurement_time(Duration::new(300, 0));
    group.bench_function("128x1M", |b| {
        let point = generate_random_array(128);
        let mat = generate_random_matrix(128, 1024 * 1024);
        b.iter(|| euclidean_distance(&point, &mat));
    });
    group.bench_function("1024x1M", |b| {
        let point = generate_random_array(1024);
        let mat = generate_random_matrix(1024, 1024 * 1024);
        b.iter(|| euclidean_distance(&point, &mat));
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
