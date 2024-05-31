// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! cargo bench --profile-time=5 --bench open_fragment

use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use std::sync::Arc;

use lance::dataset::Dataset;

fn bench_open(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dataset = rt.block_on(async {
        let metadata = [
            ("arrow:extension:name", "arrow.fixed_shape_tensor"),
            ("arrow:extension:metadata", "{ \"shape\": [100, 200, 500] }"),
        ];
        let field = Field::new("i", DataType::Int32, true).with_metadata(
            metadata
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        );
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![field.clone(), field.clone(), field.clone()])),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap();

        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema().clone()),
            "memory://test",
            None,
        )
        .await
        .unwrap()
    });

    let fragment = dataset.get_fragments().pop().unwrap();

    // Warm up cache
    rt.block_on(async { fragment.open(dataset.schema(), true).await.unwrap() });

    c.bench_function("Scan full dataset", |b| {
        b.to_async(&rt).iter(|| async {
            let _reader = fragment.open(dataset.schema(), true).await.unwrap();
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_open);
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_open);
criterion_main!(benches);
