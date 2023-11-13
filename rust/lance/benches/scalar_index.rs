use std::sync::Arc;

use arrow_array::{
    types::{UInt32Type, UInt64Type},
    RecordBatch, RecordBatchReader,
};
use async_trait::async_trait;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::scalar::ScalarValue;
use futures::{
    stream::{self, BoxStream},
    StreamExt, TryStreamExt,
};
use lance::{
    io::{object_store::ObjectStoreParams, ObjectStore},
    Dataset,
};
use lance_core::{Error, Result};
use lance_datagen::{array, gen, BatchCount, RowCount};
use lance_index::scalar::{
    btree::{train_btree_index, BTreeIndex, BtreeTrainingSource},
    flat::FlatIndexTrainer,
    lance_format::LanceIndexStore,
    IndexStore, ScalarIndex, ScalarQuery,
};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use tempfile::TempDir;

struct BenchmarkFixture {
    _datadir: TempDir,
    index_store: Arc<dyn IndexStore>,
    baseline_dataset: Arc<Dataset>,
}

struct BenchmarkDataSource {}

impl BenchmarkDataSource {
    fn test_data() -> impl RecordBatchReader {
        gen()
            .col(Some("values".to_string()), array::step::<UInt32Type>())
            .col(Some("row_ids".to_string()), array::step::<UInt64Type>())
            .into_reader_rows(RowCount::from(1024), BatchCount::from(100 * 1024))
    }
}

#[async_trait]
impl BtreeTrainingSource for BenchmarkDataSource {
    async fn scan_ordered_chunks(
        self: Box<Self>,
        _chunk_size: u32,
    ) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        Ok(stream::iter(Self::test_data().map(|batch| batch.map_err(Error::from))).boxed())
    }
}

impl BenchmarkFixture {
    fn test_store(tempdir: &TempDir) -> Arc<dyn IndexStore> {
        let test_path = tempdir.path();
        let (object_store, test_path) = ObjectStore::from_path(
            test_path.as_os_str().to_str().unwrap(),
            &ObjectStoreParams::default(),
        )
        .unwrap();
        Arc::new(LanceIndexStore::new(object_store, test_path))
    }

    async fn write_baseline_data(tempdir: &TempDir) -> Arc<Dataset> {
        let test_path = tempdir.path().as_os_str().to_str().unwrap();
        Arc::new(
            Dataset::write(BenchmarkDataSource::test_data(), test_path, None)
                .await
                .unwrap(),
        )
    }

    async fn train_scalar_index(index_store: &Arc<dyn IndexStore>) {
        let sub_index_trainer = FlatIndexTrainer::new(arrow_schema::DataType::UInt32);

        train_btree_index(
            Box::new(BenchmarkDataSource {}),
            &sub_index_trainer,
            index_store.as_ref(),
        )
        .await
        .unwrap();
    }

    async fn open() -> Self {
        let tempdir = tempfile::tempdir().unwrap();
        let index_store = Self::test_store(&tempdir);
        let baseline_dataset = Self::write_baseline_data(&tempdir).await;
        Self::train_scalar_index(&index_store).await;

        Self {
            _datadir: tempdir,
            index_store,
            baseline_dataset,
        }
    }
}

async fn baseline_equality_search(fixture: &BenchmarkFixture) {
    let mut stream = fixture
        .baseline_dataset
        .scan()
        .filter("values == 10000")
        .unwrap()
        .with_row_id()
        .try_into_stream()
        .await
        .unwrap();
    let mut num_rows = 0;
    while let Some(batch) = stream.try_next().await.unwrap() {
        num_rows += batch.num_rows();
    }
    assert_eq!(num_rows, 1);
}

async fn warm_indexed_equality_search(index: &BTreeIndex) {
    let row_ids = index
        .search(&ScalarQuery::Equals(ScalarValue::UInt32(Some(10000))))
        .await
        .unwrap();
    assert_eq!(row_ids.len(), 1);
}

async fn baseline_inequality_search(fixture: &BenchmarkFixture) {
    let mut stream = fixture
        .baseline_dataset
        .scan()
        .filter("values >= 50000000")
        .unwrap()
        .with_row_id()
        .try_into_stream()
        .await
        .unwrap();
    let mut num_rows = 0;
    while let Some(batch) = stream.try_next().await.unwrap() {
        num_rows += batch.num_rows();
    }
    // 100Mi - 50M = 54,857,600
    assert_eq!(num_rows, 54857600);
}

async fn warm_indexed_inequality_search(index: &BTreeIndex) {
    let row_ids = index
        .search(&ScalarQuery::Range(
            std::ops::Bound::Included(ScalarValue::UInt32(Some(50_000_000))),
            std::ops::Bound::Unbounded,
        ))
        .await
        .unwrap();
    // 100Mi - 50M = 54,857,600
    assert_eq!(row_ids.len(), 54857600);
}

async fn warm_indexed_isin_search(index: &BTreeIndex) {
    let row_ids = index
        .search(&ScalarQuery::IsIn(vec![
            ScalarValue::UInt32(Some(10000)),
            ScalarValue::UInt32(Some(50000000)),
            ScalarValue::UInt32(Some(150000000)), // Not found
            ScalarValue::UInt32(Some(287123)),
        ]))
        .await
        .unwrap();
    // Only 3 because 150M is not in dataset
    assert_eq!(row_ids.len(), 3);
}

fn bench_baseline(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let fixture = rt.block_on(BenchmarkFixture::open());

    c.bench_function("baseline_equality", |b| {
        b.iter(|| rt.block_on(baseline_equality_search(&fixture)));
    });

    c.bench_function("baseline_inequality", |b| {
        b.iter(|| rt.block_on(baseline_inequality_search(&fixture)));
    });
}

fn bench_warm_indexed(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(BenchmarkFixture::open());

    let index = rt
        .block_on(BTreeIndex::load(fixture.index_store.clone()))
        .unwrap();

    c.bench_function("windexed_equality", |b| {
        b.iter(|| rt.block_on(warm_indexed_equality_search(index.as_ref())))
    });

    c.bench_function("windexed_inequality", |b| {
        b.iter(|| rt.block_on(warm_indexed_inequality_search(index.as_ref())))
    });

    c.bench_function("windexed_is_in", |b| {
        b.iter(|| rt.block_on(warm_indexed_isin_search(index.as_ref())))
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_baseline, bench_warm_indexed);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_baseline, bench_warm_indexed);

criterion_main!(benches);
