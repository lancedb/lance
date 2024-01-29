// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use arrow_array::types::{UInt32Type, UInt64Type};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use datafusion_sql::sqlparser::dialect::PostgreSqlDialect;
use datafusion_sql::sqlparser::parser::Parser;
use lance_core::Result;
use lance_datafusion::exec::reader_to_stream;
use lance_datafusion::testing::TestingSqlParser;
use lance_datagen::{array, BatchCount, RowCount, DEFAULT_SEED};
use lance_index::scalar::btree::{train_btree_index, BTreeIndex, BtreeTrainingSource};
use lance_index::scalar::expression::{
    apply_scalar_indices, IndexInformationProvider, ScalarIndexExpr, ScalarIndexLoader,
};
use lance_index::scalar::flat::{FlatIndex, FlatIndexMetadata};
use lance_index::scalar::{IndexReader, IndexStore, IndexWriter, ScalarIndex, ScalarQuery};
use std::any::Any;
use std::collections::HashMap;
use std::ops::Bound;
use std::sync::{Arc, Mutex};

use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

fn bench_flat_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let data = lance_datagen::BatchGeneratorBuilder::new_with_seed(DEFAULT_SEED)
        .col(Some("values".to_string()), array::step::<UInt32Type>())
        .col(Some("row_ids".to_string()), array::step::<UInt64Type>())
        .into_batch_rows(RowCount::from(4096))
        .unwrap();

    let flat_idx = FlatIndex::from_existing_data(Arc::new(data));
    c.bench_function(&format!("flat_equality"), |b| {
        let query = ScalarQuery::Equals(ScalarValue::UInt32(Some(500)));

        b.iter(|| {
            let query_result = rt.block_on(flat_idx.search(&query)).unwrap();
            assert_eq!(query_result.len(), 1);
        })
    });
    for num_vals in [1_u32, 3, 50] {
        c.bench_function(&format!("flat_is_in_{num_vals}"), |b| {
            let values = (0..num_vals)
                .map(|idx| ScalarValue::UInt32(Some(10 * idx)))
                .collect::<Vec<_>>();
            let query = ScalarQuery::IsIn(values);

            b.iter(|| {
                let query_result = rt.block_on(flat_idx.search(&query)).unwrap();
                assert_eq!(query_result.len(), num_vals as usize);
            })
        });
        c.bench_function(&format!("flat_one_sided_range_{num_vals}"), |b| {
            let query = ScalarQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::UInt32(Some(num_vals))),
            );
            b.iter(|| {
                let query_result = rt.block_on(flat_idx.search(&query)).unwrap();
                assert_eq!(query_result.len(), num_vals as usize);
            })
        });
        c.bench_function(&format!("flat_two_sided_range_{num_vals}"), |b| {
            let query = ScalarQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(20))),
                Bound::Excluded(ScalarValue::UInt32(Some(20 + num_vals))),
            );
            b.iter(|| {
                let query_result = rt.block_on(flat_idx.search(&query)).unwrap();
                assert_eq!(query_result.len(), num_vals as usize);
            })
        });
    }
}

#[derive(Debug)]
pub struct MockIndexStore {
    serialized_indices: Arc<Mutex<HashMap<String, Vec<RecordBatch>>>>,
}

impl MockIndexStore {
    fn new() -> Self {
        Self {
            serialized_indices: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[derive(Debug)]
pub struct MockIndexStoreRef {
    serialized_indices: Arc<Mutex<HashMap<String, Vec<RecordBatch>>>>,
    idx_name: String,
}

#[async_trait]
impl IndexWriter for MockIndexStoreRef {
    async fn write_record_batch(&mut self, batch: RecordBatch) -> Result<u64> {
        let mut indices = self.serialized_indices.lock().unwrap();
        let indices = indices.get_mut(&self.idx_name).unwrap();
        let res_idx = indices.len() as u64;
        indices.push(batch);
        Ok(res_idx)
    }

    async fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl IndexReader for MockIndexStoreRef {
    async fn read_record_batch(&self, n: u32) -> Result<RecordBatch> {
        let indices = self.serialized_indices.lock().unwrap();
        let indices = indices.get(&self.idx_name).unwrap();
        Ok(indices[n as usize].clone())
    }

    async fn num_batches(&self) -> u32 {
        let indices = self.serialized_indices.lock().unwrap();
        let indices = indices.get(&self.idx_name).unwrap();
        indices.len() as u32
    }
}

#[async_trait]
impl IndexStore for MockIndexStore {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn new_index_file(
        &self,
        name: &str,
        _schema: Arc<Schema>,
    ) -> Result<Box<dyn IndexWriter>> {
        let mut indices = self.serialized_indices.lock().unwrap();
        let name = name.to_string();
        indices.insert(name.clone(), Vec::new());
        Ok(Box::new(MockIndexStoreRef {
            serialized_indices: self.serialized_indices.clone(),
            idx_name: name,
        }))
    }

    async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>> {
        let name = name.to_string();
        Ok(Arc::new(MockIndexStoreRef {
            serialized_indices: self.serialized_indices.clone(),
            idx_name: name,
        }))
    }

    /// Copy a range of batches from an index file from this store to another
    ///
    /// This is often useful when remapping or updating
    async fn copy_index_file(&self, _name: &str, _dest_store: &dyn IndexStore) -> Result<()> {
        todo!()
    }
}

pub struct MockDataSource {
    total_num_rows: u64,
}

impl MockDataSource {
    fn new(total_num_rows: u64) -> Self {
        Self { total_num_rows }
    }
}

#[async_trait]
impl BtreeTrainingSource for MockDataSource {
    async fn scan_ordered_chunks(
        self: Box<Self>,
        chunk_size: u32,
    ) -> Result<SendableRecordBatchStream> {
        let num_batches =
            ((self.total_num_rows + chunk_size as u64 - 1) / chunk_size as u64) as u32;
        let data = lance_datagen::BatchGeneratorBuilder::new_with_seed(DEFAULT_SEED)
            .col(Some("values".to_string()), array::step::<UInt32Type>())
            .col(Some("row_ids".to_string()), array::step::<UInt64Type>())
            .into_reader_rows(
                RowCount::from(chunk_size as u64),
                BatchCount::from(num_batches),
            );
        Ok(reader_to_stream(Box::new(data))?.0)
    }
}

fn bench_btree_search(c: &mut Criterion) {
    const NUM_ROWS: u64 = 200 * 1024 * 1024;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let index_store = Arc::new(MockIndexStore::new());
    // 200Mi rows
    let data_source = Box::new(MockDataSource::new(NUM_ROWS));

    let flat_trainer = FlatIndexMetadata::new(DataType::UInt32);
    rt.block_on(train_btree_index(
        data_source,
        &flat_trainer,
        index_store.as_ref(),
    ))
    .unwrap();

    let btree_index = rt.block_on(BTreeIndex::load(index_store.clone())).unwrap();
    // It's cheating to search the beginning of the index so we search the middle
    let offset = NUM_ROWS / 2;

    c.bench_function(&format!("btree_equality"), |b| {
        let query = ScalarQuery::Equals(ScalarValue::UInt32(Some(offset as u32)));

        b.iter(|| {
            let query_result = rt.block_on(btree_index.search(&query)).unwrap();
            assert_eq!(query_result.len(), 1);
        })
    });

    for num_vals in [1_u32, 100, 1000] {
        c.bench_function(&format!("btree_is_in_{num_vals}"), |b| {
            let values = (0..num_vals)
                .map(|idx| ScalarValue::UInt32(Some(5000 * idx)))
                .collect::<Vec<_>>();
            let query = ScalarQuery::IsIn(values);

            b.iter(|| {
                let query_result = rt.block_on(btree_index.search(&query)).unwrap();
                assert_eq!(query_result.len(), num_vals as usize);
            })
        });
        c.bench_function(&format!("btree_one_sided_range_{num_vals}"), |b| {
            let query = ScalarQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::UInt32(Some(num_vals))),
            );
            b.iter(|| {
                let query_result = rt.block_on(btree_index.search(&query)).unwrap();
                assert_eq!(query_result.len(), num_vals as usize);
            })
        });
        c.bench_function(&format!("btree_two_sided_range_{num_vals}"), |b| {
            let query = ScalarQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(offset as u32))),
                Bound::Excluded(ScalarValue::UInt32(Some(offset as u32 + num_vals))),
            );
            b.iter(|| {
                let query_result = rt.block_on(btree_index.search(&query)).unwrap();
                assert_eq!(query_result.len(), num_vals as usize);
            })
        });
    }
}

struct MockIndexInfoProvider {}

impl IndexInformationProvider for MockIndexInfoProvider {
    fn get_index(&self, col: &str) -> Option<&DataType> {
        Some(&DataType::UInt32)
    }
}

struct MockIndexLoader {
    index: Arc<dyn ScalarIndex>,
}

impl MockIndexLoader {
    fn new(index: Arc<dyn ScalarIndex>) -> Self {
        Self { index }
    }
}

#[async_trait]
impl ScalarIndexLoader for MockIndexLoader {
    async fn load_index(&self, _: &str) -> Result<Arc<dyn ScalarIndex>> {
        Ok(self.index.clone())
    }
}

fn bench_btree_expressions(c: &mut Criterion) {
    const NUM_ROWS: u64 = 200 * 1024 * 1024;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let index_store = Arc::new(MockIndexStore::new());
    // 200Mi rows
    let data_source = Box::new(MockDataSource::new(NUM_ROWS));

    let flat_trainer = FlatIndexMetadata::new(DataType::UInt32);
    rt.block_on(train_btree_index(
        data_source,
        &flat_trainer,
        index_store.as_ref(),
    ))
    .unwrap();

    let btree_index = rt.block_on(BTreeIndex::load(index_store.clone())).unwrap();
    let index_loader = MockIndexLoader::new(btree_index);

    let index_info_provider = MockIndexInfoProvider {};
    let schema = Schema::new(vec![Field::new("values", DataType::UInt32, true)]);

    let mut sql_parser = TestingSqlParser::try_new(schema).unwrap();

    let mut bench_expr_with_name = |expr: &str, name: &str| {
        c.bench_function(name, |b| {
            let query = sql_parser.parse_expr(expr).unwrap();
            let indexed = apply_scalar_indices(query, &index_info_provider);
            let query = indexed.scalar_query.unwrap();

            b.iter(|| {
                rt.block_on(query.evaluate(&index_loader)).unwrap();
            })
        });
    };
    let mut bench_expr = |expr: &str| bench_expr_with_name(expr, &format!("expr {}", expr));
    bench_expr("values = 500");
    bench_expr("values != 500");
    bench_expr("values IN (500)");
    bench_expr("values NOT IN (500)");
    bench_expr("values != 500 AND values != 5000 AND values != 10000 AND values != 15000");
    bench_expr("values NOT IN (500, 5000, 10000, 15000)");

    let vals = (500..800)
        .map(|val| val.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    bench_expr_with_name(&format!("values IN ({vals})"), "values IN (many)");
    bench_expr_with_name(&format!("values NOT IN ({vals})"), "values NOT IN (many)");
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets =bench_flat_search, bench_btree_search, bench_btree_expressions);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_flat_search, bench_btree_search, bench_btree_expressions);

criterion_main!(benches);
