use std::env::current_dir;

use criterion::{criterion_group, criterion_main, Criterion};
use lance::dataset::Dataset;
use lance::index::ann::{FlatIndex, Query};
use lance::utils::generate_random_array;

fn bench_search(c: &mut Criterion) {
    const NUM_THREADS: usize = 8;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(NUM_THREADS)
        .build()
        .unwrap();

    c.bench_function("vec-flat-index(1024 / 1M)", move |b| {
        let dataset_uri = current_dir().unwrap().join("vec_data");
        let dataset = runtime.block_on(async {
            Dataset::open(dataset_uri.as_path().to_str().unwrap())
                .await
                .unwrap()
        });

        let index = FlatIndex::new(&dataset, "vec".to_string());
        let params = Query {
            key: generate_random_array(256),
            k: 10,
            nprob: 0,
        };

        b.to_async(&runtime).iter(|| async {
            index.search(&params).await.unwrap();
        })
    });
}

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_search);
criterion_main!(benches);
