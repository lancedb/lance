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

//! Utilities for serializing and deserializing scalar indices in the lance format

use std::{any::Any, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use async_trait::async_trait;
use snafu::{location, Location};

use lance_core::{
    io::{
        object_store::ObjectStore, writer::FileWriterOptions, FileReader, FileWriter,
        ReadBatchParams,
    },
    Error, Result,
};
use object_store::path::Path;

use super::{IndexReader, IndexStore, IndexWriter};

/// An index store that serializes scalar indices using the lance format
///
/// Scalar indices are made up of named collections of record batches.  This
/// struct relies on there being a dedicated directory for the index and stores
/// each collection in a file in the lance format.
#[derive(Debug)]
pub struct LanceIndexStore {
    object_store: ObjectStore,
    index_dir: Path,
}

impl LanceIndexStore {
    /// Create a new index store at the given directory
    pub fn new(object_store: ObjectStore, index_dir: Path) -> Self {
        Self {
            object_store,
            index_dir,
        }
    }
}

#[async_trait]
impl IndexWriter for FileWriter {
    async fn write_record_batch(&mut self, batch: RecordBatch) -> Result<u64> {
        let offset = self.tell().await?;
        self.write(&[batch]).await?;
        Ok(offset as u64)
    }

    async fn finish(&mut self) -> Result<()> {
        Self::finish(self).await.map(|_| ())
    }
}

#[async_trait]
impl IndexReader for FileReader {
    async fn read_record_batch(&self, offset: u32) -> Result<RecordBatch> {
        self.read_batch(offset as i32, ReadBatchParams::RangeFull, self.schema())
            .await
    }

    async fn num_batches(&self) -> u32 {
        self.num_batches() as u32
    }
}

#[async_trait]
impl IndexStore for LanceIndexStore {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn new_index_file(
        &self,
        name: &str,
        schema: Arc<Schema>,
    ) -> Result<Box<dyn IndexWriter>> {
        let path = self.index_dir.child(name);
        let schema = schema.as_ref().try_into()?;
        let writer = FileWriter::try_new(
            &self.object_store,
            &path,
            schema,
            &FileWriterOptions::default(),
        )
        .await?;
        Ok(Box::new(writer))
    }

    async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>> {
        let path = self.index_dir.child(name);
        let file_reader = FileReader::try_new(&self.object_store, &path).await?;
        Ok(Arc::new(file_reader))
    }

    async fn copy_index_file(&self, name: &str, dest_store: &dyn IndexStore) -> Result<()> {
        let path = self.index_dir.child(name);

        let other_store = dest_store.as_any().downcast_ref::<Self>();
        if let Some(dest_lance_store) = other_store {
            // If both this store and the destination are lance stores we can use object_store's copy
            // This does blindly assume that both stores are using the same underlying object_store
            // but there is no easy way to verify this and it happens to always be true at the moment
            let dest_path = dest_lance_store.index_dir.child(name);
            self.object_store.copy(&path, &dest_path).await
        } else {
            let reader = self.open_index_file(name).await?;
            let num_batches = reader.num_batches().await;
            if num_batches == 0 {
                return Err(Error::Internal {
                    message:
                        "Cannot copy an empty index file because the schema cannot be determined"
                            .into(),
                    location: location!(),
                });
            }
            let first_batch = reader.read_record_batch(0).await?;
            let schema = first_batch.schema();
            let mut writer = dest_store.new_index_file(name, schema).await?;
            writer.write_record_batch(first_batch).await?;
            for batch_index in 1..num_batches {
                writer
                    .write_record_batch(reader.read_record_batch(batch_index).await?)
                    .await?;
            }
            writer.finish().await?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {

    use std::{ops::Bound, path::Path};

    use crate::scalar::{
        btree::{train_btree_index, BTreeIndex, BtreeTrainingSource},
        flat::FlatIndexMetadata,
        ScalarIndex, ScalarQuery,
    };

    use super::*;
    use arrow_array::{
        cast::AsArray,
        types::{Float32Type, Int32Type, UInt64Type},
        RecordBatchIterator, RecordBatchReader, UInt64Array,
    };
    use arrow_schema::{DataType, Field};
    use arrow_select::take::TakeOptions;
    use datafusion::physical_plan::SendableRecordBatchStream;
    use datafusion_common::ScalarValue;
    use lance_datafusion::exec::reader_to_stream;
    use lance_datagen::{array, gen, BatchCount, RowCount};
    use tempfile::{tempdir, TempDir};

    fn test_store(tempdir: &TempDir) -> Arc<dyn IndexStore> {
        let test_path: &Path = tempdir.path();
        let (object_store, test_path) =
            ObjectStore::from_path(test_path.as_os_str().to_str().unwrap()).unwrap();
        Arc::new(LanceIndexStore::new(object_store, test_path.to_owned()))
    }

    struct MockTrainingSource {
        data: SendableRecordBatchStream,
    }

    impl MockTrainingSource {
        fn new(data: impl RecordBatchReader + Send + 'static) -> Self {
            Self {
                data: reader_to_stream(Box::new(data)).unwrap().0,
            }
        }
    }

    #[async_trait]
    impl BtreeTrainingSource for MockTrainingSource {
        async fn scan_ordered_chunks(
            self: Box<Self>,
            _chunk_size: u32,
        ) -> Result<SendableRecordBatchStream> {
            Ok(self.data)
        }
    }

    async fn train_index(
        index_store: &Arc<dyn IndexStore>,
        data: impl RecordBatchReader + Send + Sync + 'static,
        value_type: DataType,
    ) {
        let sub_index_trainer = FlatIndexMetadata::new(value_type);

        let data = Box::new(MockTrainingSource::new(data));
        train_btree_index(data, &sub_index_trainer, index_store.as_ref())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_basic_btree() {
        let tempdir = tempdir().unwrap();
        let index_store = test_store(&tempdir);
        let data = gen()
            .col(Some("values".to_string()), array::step::<Int32Type>())
            .col(Some("row_ids".to_string()), array::step::<UInt64Type>())
            .into_reader_rows(RowCount::from(4096), BatchCount::from(100));
        train_index(&index_store, data, DataType::Int32).await;
        let index = BTreeIndex::load(index_store).await.unwrap();

        let row_ids = index
            .search(&ScalarQuery::Equals(ScalarValue::Int32(Some(10000))))
            .await
            .unwrap();

        assert_eq!(1, row_ids.len());
        assert_eq!(Some(10000), row_ids.values().into_iter().copied().next());

        let row_ids = index
            .search(&ScalarQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::Int32(Some(-100))),
            ))
            .await
            .unwrap();

        assert_eq!(0, row_ids.len());

        let row_ids = index
            .search(&ScalarQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::Int32(Some(100))),
            ))
            .await
            .unwrap();

        assert_eq!(100, row_ids.len());
    }

    #[tokio::test]
    async fn test_btree_update() {
        let index_dir = tempdir().unwrap();
        let index_store = test_store(&index_dir);
        let data = gen()
            .col(Some("values".to_string()), array::step::<Int32Type>())
            .col(Some("row_ids".to_string()), array::step::<UInt64Type>())
            .into_reader_rows(RowCount::from(4096), BatchCount::from(100));
        train_index(&index_store, data, DataType::Int32).await;
        let index = BTreeIndex::load(index_store).await.unwrap();

        let data = gen()
            .col(Some("values".to_string()), array::step::<Int32Type>())
            .col(Some("row_ids".to_string()), array::step::<UInt64Type>())
            .into_reader_rows(RowCount::from(4096), BatchCount::from(100));

        let updated_index_dir = tempdir().unwrap();
        let updated_index_store = test_store(&updated_index_dir);
        index
            .update(
                reader_to_stream(Box::new(data)).unwrap().0,
                updated_index_store.as_ref(),
            )
            .await
            .unwrap();
        let updated_index = BTreeIndex::load(updated_index_store).await.unwrap();

        let row_ids = updated_index
            .search(&ScalarQuery::Equals(ScalarValue::Int32(Some(10000))))
            .await
            .unwrap();

        assert_eq!(2, row_ids.len());
        assert_eq!(
            vec![10000, 10000],
            row_ids.values().into_iter().copied().collect::<Vec<_>>()
        );
    }

    async fn check(index: &BTreeIndex, query: ScalarQuery, expected: &[u64]) {
        let results = index.search(&query).await.unwrap();
        let expected_arr = UInt64Array::from_iter_values(expected.iter().copied());
        assert_eq!(results, expected_arr);
    }

    #[tokio::test]
    async fn test_btree_with_gaps() {
        let tempdir = tempdir().unwrap();
        let index_store = test_store(&tempdir);
        let batch_one = gen()
            .col(
                Some("values".to_string()),
                array::cycle::<Int32Type>(vec![0, 1, 4, 5]),
            )
            .col(
                Some("row_ids".to_string()),
                array::cycle::<UInt64Type>(vec![0, 1, 2, 3]),
            )
            .into_batch_rows(RowCount::from(4));
        let batch_two = gen()
            .col(
                Some("values".to_string()),
                array::cycle::<Int32Type>(vec![10, 11, 11, 15]),
            )
            .col(
                Some("row_ids".to_string()),
                array::cycle::<UInt64Type>(vec![40, 50, 60, 70]),
            )
            .into_batch_rows(RowCount::from(4));
        let batch_three = gen()
            .col(
                Some("values".to_string()),
                array::cycle::<Int32Type>(vec![15, 15, 15, 15]),
            )
            .col(
                Some("row_ids".to_string()),
                array::cycle::<UInt64Type>(vec![400, 500, 600, 700]),
            )
            .into_batch_rows(RowCount::from(4));
        let batch_four = gen()
            .col(
                Some("values".to_string()),
                array::cycle::<Int32Type>(vec![15, 16, 20, 20]),
            )
            .col(
                Some("row_ids".to_string()),
                array::cycle::<UInt64Type>(vec![4000, 5000, 6000, 7000]),
            )
            .into_batch_rows(RowCount::from(4));
        let batches = vec![batch_one, batch_two, batch_three, batch_four];
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::Int32, false),
            Field::new("row_ids", DataType::UInt64, false),
        ]));
        let data = RecordBatchIterator::new(batches, schema);
        train_index(&index_store, data, DataType::Int32).await;
        let index = BTreeIndex::load(index_store).await.unwrap();

        // The above should create four pages
        //
        // 0 - 5
        // 10 - 15
        // 15 - 15
        // 15 - 20
        //
        // This will help us test various indexing corner cases

        // No results (off the left side)
        check(
            &index,
            ScalarQuery::Equals(ScalarValue::Int32(Some(-3))),
            &[],
        )
        .await;

        check(
            &index,
            ScalarQuery::Range(
                Bound::Unbounded,
                Bound::Included(ScalarValue::Int32(Some(-3))),
            ),
            &[],
        )
        .await;

        check(
            &index,
            ScalarQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(-10))),
                Bound::Included(ScalarValue::Int32(Some(-3))),
            ),
            &[],
        )
        .await;

        // Hitting the middle of a bucket
        check(
            &index,
            ScalarQuery::Equals(ScalarValue::Int32(Some(4))),
            &[2],
        )
        .await;

        // Hitting a gap between two buckets
        check(
            &index,
            ScalarQuery::Equals(ScalarValue::Int32(Some(7))),
            &[],
        )
        .await;

        // Hitting the lowest of the overlapping buckets
        check(
            &index,
            ScalarQuery::Equals(ScalarValue::Int32(Some(11))),
            &[50, 60],
        )
        .await;

        // Hitting the 15 shared on all three buckets
        check(
            &index,
            ScalarQuery::Equals(ScalarValue::Int32(Some(15))),
            &[70, 400, 500, 600, 700, 4000],
        )
        .await;

        // Hitting the upper part of the three overlapping buckets
        check(
            &index,
            ScalarQuery::Equals(ScalarValue::Int32(Some(20))),
            &[6000, 7000],
        )
        .await;

        // Ranges that capture multiple buckets
        check(
            &index,
            ScalarQuery::Range(
                Bound::Unbounded,
                Bound::Included(ScalarValue::Int32(Some(11))),
            ),
            &[0, 1, 2, 3, 40, 50, 60],
        )
        .await;

        check(
            &index,
            ScalarQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::Int32(Some(11))),
            ),
            &[0, 1, 2, 3, 40],
        )
        .await;

        check(
            &index,
            ScalarQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(4))),
                Bound::Unbounded,
            ),
            &[
                2, 3, 40, 50, 60, 70, 400, 500, 600, 700, 4000, 5000, 6000, 7000,
            ],
        )
        .await;

        check(
            &index,
            ScalarQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(4))),
                Bound::Included(ScalarValue::Int32(Some(11))),
            ),
            &[2, 3, 40, 50, 60],
        )
        .await;

        check(
            &index,
            ScalarQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(4))),
                Bound::Excluded(ScalarValue::Int32(Some(11))),
            ),
            &[2, 3, 40],
        )
        .await;

        check(
            &index,
            ScalarQuery::Range(
                Bound::Excluded(ScalarValue::Int32(Some(4))),
                Bound::Unbounded,
            ),
            &[
                3, 40, 50, 60, 70, 400, 500, 600, 700, 4000, 5000, 6000, 7000,
            ],
        )
        .await;

        check(
            &index,
            ScalarQuery::Range(
                Bound::Excluded(ScalarValue::Int32(Some(4))),
                Bound::Included(ScalarValue::Int32(Some(11))),
            ),
            &[3, 40, 50, 60],
        )
        .await;

        check(
            &index,
            ScalarQuery::Range(
                Bound::Excluded(ScalarValue::Int32(Some(4))),
                Bound::Excluded(ScalarValue::Int32(Some(11))),
            ),
            &[3, 40],
        )
        .await;

        check(
            &index,
            ScalarQuery::Range(
                Bound::Excluded(ScalarValue::Int32(Some(-50))),
                Bound::Excluded(ScalarValue::Int32(Some(1000))),
            ),
            &[
                0, 1, 2, 3, 40, 50, 60, 70, 400, 500, 600, 700, 4000, 5000, 6000, 7000,
            ],
        )
        .await;
    }

    #[tokio::test]
    async fn test_btree_types() {
        for data_type in &[
            DataType::Boolean,
            DataType::Int32,
            DataType::Utf8,
            DataType::Float32,
            DataType::Date32,
        ] {
            let tempdir = tempdir().unwrap();
            let index_store = test_store(&tempdir);
            let data: RecordBatch = gen()
                .col(Some("values".to_string()), array::rand_type(data_type))
                .col(Some("row_ids".to_string()), array::step::<UInt64Type>())
                .into_batch_rows(RowCount::from(4096 * 3))
                .unwrap();

            let sample_value = ScalarValue::try_from_array(data.column(0), 0).unwrap();
            let sample_row_id = data.column(1).as_primitive::<UInt64Type>().value(0);

            let sort_indices = arrow::compute::sort_to_indices(data.column(0), None, None).unwrap();
            let sorted_values = arrow_select::take::take(
                data.column(0),
                &sort_indices,
                Some(TakeOptions {
                    check_bounds: false,
                }),
            )
            .unwrap();
            let sorted_row_ids = arrow_select::take::take(
                data.column(1),
                &sort_indices,
                Some(TakeOptions {
                    check_bounds: false,
                }),
            )
            .unwrap();
            let sorted_batch =
                RecordBatch::try_new(data.schema().clone(), vec![sorted_values, sorted_row_ids])
                    .unwrap();

            let batch_one = sorted_batch.slice(0, 4096);
            let batch_two = sorted_batch.slice(4096, 4096);
            let batch_three = sorted_batch.slice(8192, 4096);
            let training_data = RecordBatchIterator::new(
                vec![batch_one, batch_two, batch_three].into_iter().map(Ok),
                data.schema().clone(),
            );

            train_index(&index_store, training_data, data_type.clone()).await;
            let index = BTreeIndex::load(index_store).await.unwrap();

            let row_ids = index
                .search(&ScalarQuery::Equals(sample_value))
                .await
                .unwrap();

            // The random data may have had duplicates so there might be more than 1 result
            // but even for boolean we shouldn't match the entire thing
            assert!(!row_ids.is_empty());
            assert!(row_ids.len() < data.num_rows());
            assert!(row_ids.values().iter().any(|val| *val == sample_row_id));
        }
    }

    #[tokio::test]
    async fn btree_reject_nan() {
        let tempdir = tempdir().unwrap();
        let index_store = test_store(&tempdir);
        let batch = gen()
            .col(
                Some("values".to_string()),
                array::cycle::<Float32Type>(vec![0.0, f32::NAN]),
            )
            .col(
                Some("row_ids".to_string()),
                array::cycle::<UInt64Type>(vec![0, 1]),
            )
            .into_batch_rows(RowCount::from(2));
        let batches = vec![batch];
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::Float32, false),
            Field::new("row_ids", DataType::UInt64, false),
        ]));
        let data = RecordBatchIterator::new(batches, schema);
        let sub_index_trainer = FlatIndexMetadata::new(DataType::Float32);

        let data = Box::new(MockTrainingSource::new(data));
        // Until DF handles NaN reliably we need to make sure we reject input
        // containing NaN
        assert!(
            train_btree_index(data, &sub_index_trainer, index_store.as_ref())
                .await
                .is_err()
        );
    }
}
