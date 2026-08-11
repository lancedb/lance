// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compares fixed-size fragment aggregation with materializing per-fragment JNI payload data.
//!
//! The flattened baseline mirrors the native work behind Java
//! `Dataset.getFragmentStatistics()`: it allocates three `i64` values per fragment before JNI
//! copies and Java-side array splitting. `Dataset.fragment_summary()` returns five scalars
//! regardless of fragment count.
//!
//! At 100,000 fragments, the flattened path allocates a 2.4 MB native vector, followed by a
//! 2.4 MB Java `long[]` JNI copy and 1.6 MB across the three final Java primitive arrays. The
//! aggregate path creates one fixed-size Java object.
//!
//! ```text
//! cargo bench -p lance --bench fragment_summary
//! ```

use std::hint::black_box;

use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lance::Dataset;
use lance::dataset::transaction::Operation;
use lance_core::utils::tempfile::TempStrDir;
use lance_table::format::Fragment;

const NUM_FRAGMENTS: usize = 100_000;

struct Fixture {
    _data_dir: TempStrDir,
    dataset: Dataset,
}

impl Fixture {
    async fn open() -> Self {
        let data_dir = TempStrDir::default();
        let schema =
            lance_core::datatypes::Schema::try_from(&ArrowSchema::new(vec![ArrowField::new(
                "value",
                DataType::Int64,
                false,
            )]))
            .unwrap();
        let fragments = (0..NUM_FRAGMENTS)
            .map(|id| {
                let mut fragment = Fragment::new(id as u64);
                fragment.physical_rows = Some(1_000 + id % 100);
                fragment
            })
            .collect();
        let operation = Operation::Overwrite {
            fragments,
            schema,
            config_upsert_values: None,
            initial_bases: None,
        };
        let dataset = Dataset::commit(
            data_dir.as_str(),
            operation,
            None,
            None,
            None,
            Default::default(),
            false,
        )
        .await
        .unwrap();

        Self {
            _data_dir: data_dir,
            dataset,
        }
    }
}

fn flattened_fragment_statistics(dataset: &Dataset) -> Vec<i64> {
    let fragments = dataset.fragments();
    let mut statistics = Vec::with_capacity(fragments.len() * 3);
    for fragment in fragments.iter() {
        let physical_rows = fragment.physical_rows.unwrap_or(0) as i64;
        let deleted_rows = fragment
            .deletion_file
            .as_ref()
            .and_then(|deletion_file| deletion_file.num_deleted_rows)
            .unwrap_or(0) as i64;
        statistics.push(fragment.id as i64);
        statistics.push(physical_rows - deleted_rows);
        statistics.push(fragment.files.len() as i64);
    }
    statistics
}

fn bench_fragment_statistics(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let fixture = runtime.block_on(Fixture::open());

    let summary = fixture.dataset.fragment_summary().unwrap();
    assert_eq!(summary.fragment_count, NUM_FRAGMENTS as u64);
    assert_eq!(
        flattened_fragment_statistics(&fixture.dataset).len(),
        NUM_FRAGMENTS * 3
    );

    let mut group = c.benchmark_group("fragment_statistics/100k_fragments");
    group.throughput(Throughput::Elements(NUM_FRAGMENTS as u64));
    group.bench_function("aggregate_summary", |b| {
        b.iter(|| black_box(fixture.dataset.fragment_summary().unwrap()))
    });
    group.bench_function("materialize_flattened_statistics", |b| {
        b.iter(|| black_box(flattened_fragment_statistics(&fixture.dataset)))
    });
    group.finish();
}

criterion_group!(benches, bench_fragment_statistics);
criterion_main!(benches);
