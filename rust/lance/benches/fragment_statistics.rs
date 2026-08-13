// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compares the old and new per-fragment JNI payload preparation paths.
//!
//! The flattened baseline mirrors the old Java `Dataset.getFragmentStatistics()` path: it creates
//! `FileFragment` wrappers and allocates three `i64` values per fragment before JNI copies and
//! Java-side array splitting. The typed-chunk path mirrors the current implementation's native
//! preparation before it copies directly into the three final Java arrays.
//!
//! At 100,000 fragments, the old flattened path allocates a 2.4 MB native vector, followed by a
//! 2.4 MB Java `long[]` JNI copy and 1.6 MB across the three final Java primitive arrays. The
//! typed-chunk path bounds native staging memory to 64 KiB and creates only the 1.6 MB final Java
//! arrays.
//!
//! ```text
//! cargo bench -p lance --bench fragment_statistics
//! ```

use std::hint::black_box;

use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lance::Dataset;
use lance::dataset::transaction::Operation;
use lance_core::utils::tempfile::TempStrDir;
use lance_table::format::Fragment;

const NUM_FRAGMENTS: usize = 100_000;
const STATISTICS_CHUNK_SIZE: usize = 4096;

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

fn legacy_flattened_fragment_statistics(dataset: &Dataset) -> Vec<i64> {
    let fragments = dataset.get_fragments();
    let mut statistics = Vec::with_capacity(fragments.len() * 3);
    for fragment in fragments.iter() {
        let metadata = fragment.metadata();
        let physical_rows = metadata.physical_rows.unwrap_or(0) as i64;
        let deleted_rows = metadata
            .deletion_file
            .as_ref()
            .and_then(|deletion_file| deletion_file.num_deleted_rows)
            .unwrap_or(0) as i64;
        statistics.push(metadata.id as i64);
        statistics.push(physical_rows - deleted_rows);
        statistics.push(metadata.files.len() as i64);
    }
    statistics
}

fn prepare_typed_fragment_statistics_chunks(dataset: &Dataset) -> usize {
    let fragments = dataset.fragments();
    let chunk_capacity = fragments.len().min(STATISTICS_CHUNK_SIZE);
    let mut ids = Vec::with_capacity(chunk_capacity);
    let mut row_counts = Vec::with_capacity(chunk_capacity);
    let mut data_file_nums = Vec::with_capacity(chunk_capacity);
    let mut value_count = 0;

    for fragments in fragments.chunks(STATISTICS_CHUNK_SIZE) {
        ids.clear();
        row_counts.clear();
        data_file_nums.clear();
        for fragment in fragments {
            let physical_rows = fragment.physical_rows.unwrap_or(0) as i64;
            let deleted_rows = fragment
                .deletion_file
                .as_ref()
                .and_then(|deletion_file| deletion_file.num_deleted_rows)
                .unwrap_or(0) as i64;
            ids.push(fragment.id as i32);
            row_counts.push(physical_rows - deleted_rows);
            data_file_nums.push(fragment.files.len() as i32);
        }
        value_count += ids.len() + row_counts.len() + data_file_nums.len();
        black_box((&ids, &row_counts, &data_file_nums));
    }

    value_count
}

fn bench_fragment_statistics(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let fixture = runtime.block_on(Fixture::open());

    assert_eq!(
        legacy_flattened_fragment_statistics(&fixture.dataset).len(),
        NUM_FRAGMENTS * 3
    );
    assert_eq!(
        prepare_typed_fragment_statistics_chunks(&fixture.dataset),
        NUM_FRAGMENTS * 3
    );

    let mut group = c.benchmark_group("fragment_statistics/100k_fragments");
    group.throughput(Throughput::Elements(NUM_FRAGMENTS as u64));
    group.bench_function("legacy_materialize_flattened_statistics", |b| {
        b.iter(|| black_box(legacy_flattened_fragment_statistics(&fixture.dataset)))
    });
    group.bench_function("prepare_typed_statistics_chunks", |b| {
        b.iter(|| black_box(prepare_typed_fragment_statistics_chunks(&fixture.dataset)))
    });
    group.finish();
}

criterion_group!(benches, bench_fragment_statistics);
criterion_main!(benches);
