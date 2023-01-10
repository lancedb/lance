use criterion::{criterion_group, criterion_main, Criterion};

use arrow_array::FixedSizeListArray;

use lance::index::ann::distance::euclidean_distance;
use lance::{arrow::FixedSizeListArrayExt, utils::generate_random_array};

fn bench_distance(c: &mut Criterion) {
    const DIMENSION: i32 = 1024;

    let key = generate_random_array(DIMENSION as usize);
    // 1M of 1024 D vectors. 4GB in memory.
    let values = generate_random_array(1024 * 1024 * DIMENSION as usize);
    let target = FixedSizeListArray::try_new(values, DIMENSION).unwrap();

    c.bench_function("L2 distance", |b| {
        b.iter(|| {
            euclidean_distance(&key, &target).unwrap();
        })
    });
}

criterion_group!(benches, bench_distance);
criterion_main!(benches);
