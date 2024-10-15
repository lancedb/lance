// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for serializing and deserializing scalar indices in the lance format

use std::cmp::min;
use std::collections::HashMap;
use std::{any::Any, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use futures::TryStreamExt;
use lance_core::{cache::FileMetadataCache, Error, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::v2;
use lance_file::v2::reader::FileReaderOptions;
use lance_file::writer::FileWriterOptions;
use lance_file::{
    reader::FileReader,
    writer::{FileWriter, ManifestProvider},
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::{object_store::ObjectStore, ReadBatchParams};
use lance_table::format::SelfDescribingFileReader;
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;

use super::{IndexReader, IndexStore, IndexWriter};

/// An index store that serializes scalar indices using the lance format
///
/// Scalar indices are made up of named collections of record batches.  This
/// struct relies on there being a dedicated directory for the index and stores
/// each collection in a file in the lance format.
#[derive(Debug)]
pub struct LanceIndexStore {
    object_store: Arc<ObjectStore>,
    index_dir: Path,
    metadata_cache: FileMetadataCache,
    scheduler: Arc<ScanScheduler>,
    use_legacy_format: bool,
}

impl DeepSizeOf for LanceIndexStore {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.object_store.deep_size_of_children(context)
            + self.index_dir.as_ref().deep_size_of_children(context)
            + self.metadata_cache.deep_size_of_children(context)
    }
}

impl LanceIndexStore {
    /// Create a new index store at the given directory
    pub fn new(
        object_store: ObjectStore,
        index_dir: Path,
        metadata_cache: FileMetadataCache,
    ) -> Self {
        let object_store = Arc::new(object_store);
        let scheduler = ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        Self {
            object_store,
            index_dir,
            metadata_cache,
            scheduler,
            use_legacy_format: false,
        }
    }

    pub fn with_legacy_format(mut self, use_legacy_format: bool) -> Self {
        self.use_legacy_format = use_legacy_format;
        self
    }
}

#[async_trait]
impl<M: ManifestProvider + Send + Sync> IndexWriter for FileWriter<M> {
    async fn write_record_batch(&mut self, batch: RecordBatch) -> Result<u64> {
        let offset = self.tell().await?;
        self.write(&[batch]).await?;
        Ok(offset as u64)
    }

    async fn finish(&mut self) -> Result<()> {
        Self::finish(self).await.map(|_| ())
    }

    async fn finish_with_metadata(&mut self, metadata: HashMap<String, String>) -> Result<()> {
        Self::finish_with_metadata(self, &metadata)
            .await
            .map(|_| ())
    }
}

#[async_trait]
impl IndexWriter for v2::writer::FileWriter {
    async fn write_record_batch(&mut self, batch: RecordBatch) -> Result<u64> {
        let offset = self.tell().await?;
        self.write_batch(&batch).await?;
        Ok(offset)
    }

    async fn finish(&mut self) -> Result<()> {
        Self::finish(self).await.map(|_| ())
    }

    async fn finish_with_metadata(&mut self, metadata: HashMap<String, String>) -> Result<()> {
        metadata.into_iter().for_each(|(k, v)| {
            self.add_schema_metadata(k, v);
        });
        Self::finish(self).await.map(|_| ())
    }
}

#[async_trait]
impl IndexReader for FileReader {
    async fn read_record_batch(&self, offset: u32) -> Result<RecordBatch> {
        self.read_batch(offset as i32, ReadBatchParams::RangeFull, self.schema())
            .await
    }

    async fn read_range(
        &self,
        range: std::ops::Range<usize>,
        projection: Option<&[&str]>,
    ) -> Result<RecordBatch> {
        let projection = match projection {
            Some(projection) => self.schema().project(projection)?,
            None => self.schema().clone(),
        };
        self.read_range(range, &projection).await
    }

    async fn num_batches(&self) -> u32 {
        self.num_batches() as u32
    }

    fn num_rows(&self) -> usize {
        self.len()
    }

    fn schema(&self) -> &lance_core::datatypes::Schema {
        Self::schema(self)
    }
}

#[async_trait]
impl IndexReader for v2::reader::FileReader {
    async fn read_record_batch(&self, _offset: u32) -> Result<RecordBatch> {
        unimplemented!("v2 format has no concept of row groups")
    }

    async fn read_range(
        &self,
        range: std::ops::Range<usize>,
        projection: Option<&[&str]>,
    ) -> Result<RecordBatch> {
        let projection = if let Some(projection) = projection {
            v2::reader::ReaderProjection::from_column_names(self.schema(), projection)?
        } else {
            v2::reader::ReaderProjection::from_whole_schema(
                self.schema(),
                self.metadata().version(),
            )
        };
        let batches = self
            .read_stream_projected(
                ReadBatchParams::Range(range),
                u32::MAX,
                u32::MAX,
                projection,
                FilterExpression::no_filter(),
            )?
            .try_collect::<Vec<_>>()
            .await?;
        assert_eq!(batches.len(), 1);
        Ok(batches[0].clone())
    }

    // V2 format has removed the row group concept,
    // so here we assume each batch is with 4096 rows.
    async fn num_batches(&self) -> u32 {
        unimplemented!("v2 format has no concept of row groups")
    }

    fn num_rows(&self) -> usize {
        Self::num_rows(self) as usize
    }

    fn schema(&self) -> &lance_core::datatypes::Schema {
        Self::schema(self)
    }
}

#[async_trait]
impl IndexStore for LanceIndexStore {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn io_parallelism(&self) -> usize {
        self.object_store.io_parallelism()
    }

    async fn new_index_file(
        &self,
        name: &str,
        schema: Arc<Schema>,
    ) -> Result<Box<dyn IndexWriter>> {
        let path = self.index_dir.child(name);
        let schema = schema.as_ref().try_into()?;
        if self.use_legacy_format {
            let writer = FileWriter::<ManifestDescribing>::try_new(
                &self.object_store,
                &path,
                schema,
                &FileWriterOptions::default(),
            )
            .await?;
            Ok(Box::new(writer))
        } else {
            let writer = self.object_store.create(&path).await?;
            let writer = v2::writer::FileWriter::try_new(
                writer,
                schema,
                v2::writer::FileWriterOptions::default(),
            )?;
            Ok(Box::new(writer))
        }
    }

    async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>> {
        let path = self.index_dir.child(name);
        let file_scheduler = self.scheduler.open_file(&path).await?;
        match v2::reader::FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &self.metadata_cache,
            FileReaderOptions::default(),
        )
        .await
        {
            Ok(reader) => Ok(Arc::new(reader)),
            Err(e) => {
                // If the error is a version conflict we can try to read the file with v1 reader
                if let Error::VersionConflict { .. } = e {
                    let path = self.index_dir.child(name);
                    let file_reader = FileReader::try_new_self_described(
                        &self.object_store,
                        &path,
                        Some(&self.metadata_cache),
                    )
                    .await?;
                    Ok(Arc::new(file_reader))
                } else {
                    Err(e)
                }
            }
        }
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
            let mut writer = dest_store
                .new_index_file(name, Arc::new(reader.schema().into()))
                .await?;

            for offset in (0..reader.num_rows()).step_by(4096) {
                let next_offset = min(offset + 4096, reader.num_rows());
                let batch = reader.read_range(offset..next_offset, None).await?;
                writer.write_record_batch(batch).await?;
            }
            writer.finish().await?;

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {

    use std::{collections::HashMap, ops::Bound, path::Path};

    use crate::scalar::{
        bitmap::{train_bitmap_index, BitmapIndex},
        btree::{train_btree_index, BTreeIndex, TrainingSource},
        flat::FlatIndexMetadata,
        label_list::{train_label_list_index, LabelListIndex},
        LabelListQuery, SargableQuery, ScalarIndex,
    };

    use super::*;
    use arrow::{buffer::ScalarBuffer, datatypes::UInt8Type};
    use arrow_array::{
        cast::AsArray,
        types::{Float32Type, Int32Type, UInt64Type},
        RecordBatchIterator, RecordBatchReader, StringArray, UInt64Array,
    };
    use arrow_schema::Schema as ArrowSchema;
    use arrow_schema::{DataType, Field, TimeUnit};
    use arrow_select::take::TakeOptions;
    use datafusion::physical_plan::SendableRecordBatchStream;
    use datafusion_common::ScalarValue;
    use lance_core::{cache::CapacityMode, utils::mask::RowIdTreeMap};
    use lance_datagen::{array, gen, ArrayGeneratorExt, BatchCount, ByteCount, RowCount};
    use tempfile::{tempdir, TempDir};

    fn test_store(tempdir: &TempDir) -> Arc<dyn IndexStore> {
        let test_path: &Path = tempdir.path();
        let (object_store, test_path) =
            ObjectStore::from_path(test_path.as_os_str().to_str().unwrap()).unwrap();
        let cache = FileMetadataCache::with_capacity(128 * 1024 * 1024, CapacityMode::Bytes);
        Arc::new(LanceIndexStore::new(
            object_store,
            test_path.to_owned(),
            cache,
        ))
    }

    fn legacy_test_store(tempdir: &TempDir) -> Arc<dyn IndexStore> {
        let test_path: &Path = tempdir.path();
        let cache = FileMetadataCache::with_capacity(128 * 1024 * 1024, CapacityMode::Bytes);
        let (object_store, test_path) =
            ObjectStore::from_path(test_path.as_os_str().to_str().unwrap()).unwrap();
        Arc::new(
            LanceIndexStore::new(object_store, test_path.to_owned(), cache)
                .with_legacy_format(true),
        )
    }

    struct MockTrainingSource {
        data: SendableRecordBatchStream,
    }

    impl MockTrainingSource {
        async fn new(data: impl RecordBatchReader + Send + 'static) -> Self {
            Self {
                data: lance_datafusion::utils::reader_to_stream(Box::new(data)),
            }
        }
    }

    #[async_trait]
    impl TrainingSource for MockTrainingSource {
        async fn scan_ordered_chunks(
            self: Box<Self>,
            _chunk_size: u32,
        ) -> Result<SendableRecordBatchStream> {
            Ok(self.data)
        }

        async fn scan_unordered_chunks(
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

        let data = Box::new(MockTrainingSource::new(data).await);
        train_btree_index(data, &sub_index_trainer, index_store.as_ref())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_basic_btree() {
        let tempdir = tempdir().unwrap();
        let index_store = legacy_test_store(&tempdir);
        let data = gen()
            .col("values", array::step::<Int32Type>())
            .col("row_ids", array::step::<UInt64Type>())
            .into_reader_rows(RowCount::from(4096), BatchCount::from(100));
        train_index(&index_store, data, DataType::Int32).await;
        let index = BTreeIndex::load(index_store).await.unwrap();

        let row_ids = index
            .search(&SargableQuery::Equals(ScalarValue::Int32(Some(10000))))
            .await
            .unwrap();

        assert_eq!(Some(1), row_ids.len());
        assert!(row_ids.contains(10000));

        let row_ids = index
            .search(&SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::Int32(Some(-100))),
            ))
            .await
            .unwrap();

        assert_eq!(Some(0), row_ids.len());

        let row_ids = index
            .search(&SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::Int32(Some(100))),
            ))
            .await
            .unwrap();

        assert_eq!(Some(100), row_ids.len());
    }

    #[tokio::test]
    async fn test_btree_update() {
        let index_dir = tempdir().unwrap();
        let index_store = legacy_test_store(&index_dir);
        let data = gen()
            .col("values", array::step::<Int32Type>())
            .col("row_ids", array::step::<UInt64Type>())
            .into_reader_rows(RowCount::from(4096), BatchCount::from(100));
        train_index(&index_store, data, DataType::Int32).await;
        let index = BTreeIndex::load(index_store).await.unwrap();

        let data = gen()
            .col("values", array::step_custom::<Int32Type>(4096 * 100, 1))
            .col("row_ids", array::step_custom::<UInt64Type>(4096 * 100, 1))
            .into_reader_rows(RowCount::from(4096), BatchCount::from(100));

        let updated_index_dir = tempdir().unwrap();
        let updated_index_store = legacy_test_store(&updated_index_dir);
        index
            .update(
                lance_datafusion::utils::reader_to_stream(Box::new(data)),
                updated_index_store.as_ref(),
            )
            .await
            .unwrap();
        let updated_index = BTreeIndex::load(updated_index_store).await.unwrap();

        let row_ids = updated_index
            .search(&SargableQuery::Equals(ScalarValue::Int32(Some(10000))))
            .await
            .unwrap();

        assert_eq!(Some(1), row_ids.len());
        assert!(row_ids.contains(10000));

        let row_ids = updated_index
            .search(&SargableQuery::Equals(ScalarValue::Int32(Some(500_000))))
            .await
            .unwrap();

        assert_eq!(Some(1), row_ids.len());
        assert!(row_ids.contains(500_000));
    }

    async fn check(index: &BTreeIndex, query: SargableQuery, expected: &[u64]) {
        let results = index.search(&query).await.unwrap();
        let expected_arr = RowIdTreeMap::from_iter(expected);
        assert_eq!(results, expected_arr);
    }

    #[tokio::test]
    async fn test_btree_with_gaps() {
        let tempdir = tempdir().unwrap();
        let index_store = legacy_test_store(&tempdir);
        let batch_one = gen()
            .col("values", array::cycle::<Int32Type>(vec![0, 1, 4, 5]))
            .col("row_ids", array::cycle::<UInt64Type>(vec![0, 1, 2, 3]))
            .into_batch_rows(RowCount::from(4));
        let batch_two = gen()
            .col("values", array::cycle::<Int32Type>(vec![10, 11, 11, 15]))
            .col("row_ids", array::cycle::<UInt64Type>(vec![40, 50, 60, 70]))
            .into_batch_rows(RowCount::from(4));
        let batch_three = gen()
            .col("values", array::cycle::<Int32Type>(vec![15, 15, 15, 15]))
            .col(
                "row_ids",
                array::cycle::<UInt64Type>(vec![400, 500, 600, 700]),
            )
            .into_batch_rows(RowCount::from(4));
        let batch_four = gen()
            .col("values", array::cycle::<Int32Type>(vec![15, 16, 20, 20]))
            .col(
                "row_ids",
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
            SargableQuery::Equals(ScalarValue::Int32(Some(-3))),
            &[],
        )
        .await;

        check(
            &index,
            SargableQuery::Range(
                Bound::Unbounded,
                Bound::Included(ScalarValue::Int32(Some(-3))),
            ),
            &[],
        )
        .await;

        check(
            &index,
            SargableQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(-10))),
                Bound::Included(ScalarValue::Int32(Some(-3))),
            ),
            &[],
        )
        .await;

        // Hitting the middle of a bucket
        check(
            &index,
            SargableQuery::Equals(ScalarValue::Int32(Some(4))),
            &[2],
        )
        .await;

        // Hitting a gap between two buckets
        check(
            &index,
            SargableQuery::Equals(ScalarValue::Int32(Some(7))),
            &[],
        )
        .await;

        // Hitting the lowest of the overlapping buckets
        check(
            &index,
            SargableQuery::Equals(ScalarValue::Int32(Some(11))),
            &[50, 60],
        )
        .await;

        // Hitting the 15 shared on all three buckets
        check(
            &index,
            SargableQuery::Equals(ScalarValue::Int32(Some(15))),
            &[70, 400, 500, 600, 700, 4000],
        )
        .await;

        // Hitting the upper part of the three overlapping buckets
        check(
            &index,
            SargableQuery::Equals(ScalarValue::Int32(Some(20))),
            &[6000, 7000],
        )
        .await;

        // Ranges that capture multiple buckets
        check(
            &index,
            SargableQuery::Range(
                Bound::Unbounded,
                Bound::Included(ScalarValue::Int32(Some(11))),
            ),
            &[0, 1, 2, 3, 40, 50, 60],
        )
        .await;

        check(
            &index,
            SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::Int32(Some(11))),
            ),
            &[0, 1, 2, 3, 40],
        )
        .await;

        check(
            &index,
            SargableQuery::Range(
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
            SargableQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(4))),
                Bound::Included(ScalarValue::Int32(Some(11))),
            ),
            &[2, 3, 40, 50, 60],
        )
        .await;

        check(
            &index,
            SargableQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(4))),
                Bound::Excluded(ScalarValue::Int32(Some(11))),
            ),
            &[2, 3, 40],
        )
        .await;

        check(
            &index,
            SargableQuery::Range(
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
            SargableQuery::Range(
                Bound::Excluded(ScalarValue::Int32(Some(4))),
                Bound::Included(ScalarValue::Int32(Some(11))),
            ),
            &[3, 40, 50, 60],
        )
        .await;

        check(
            &index,
            SargableQuery::Range(
                Bound::Excluded(ScalarValue::Int32(Some(4))),
                Bound::Excluded(ScalarValue::Int32(Some(11))),
            ),
            &[3, 40],
        )
        .await;

        check(
            &index,
            SargableQuery::Range(
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
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            DataType::Date64,
            DataType::Date32,
            DataType::Time64(TimeUnit::Nanosecond),
            DataType::Time32(TimeUnit::Second),
            // Not supported today, error from datafusion:
            // Min/max accumulator not implemented for Duration(Nanosecond)
            // DataType::Duration(TimeUnit::Nanosecond),
        ] {
            let tempdir = tempdir().unwrap();
            let index_store = legacy_test_store(&tempdir);
            let data: RecordBatch = gen()
                .col("values", array::rand_type(data_type))
                .col("row_ids", array::step::<UInt64Type>())
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
                .search(&SargableQuery::Equals(sample_value))
                .await
                .unwrap();

            // The random data may have had duplicates so there might be more than 1 result
            // but even for boolean we shouldn't match the entire thing
            assert!(!row_ids.is_empty());
            assert!(row_ids.len().unwrap() < data.num_rows() as u64);
            assert!(row_ids.contains(sample_row_id));
        }
    }

    #[tokio::test]
    async fn btree_reject_nan() {
        let tempdir = tempdir().unwrap();
        let index_store = legacy_test_store(&tempdir);
        let batch = gen()
            .col("values", array::cycle::<Float32Type>(vec![0.0, f32::NAN]))
            .col("row_ids", array::cycle::<UInt64Type>(vec![0, 1]))
            .into_batch_rows(RowCount::from(2));
        let batches = vec![batch];
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::Float32, false),
            Field::new("row_ids", DataType::UInt64, false),
        ]));
        let data = RecordBatchIterator::new(batches, schema);
        let sub_index_trainer = FlatIndexMetadata::new(DataType::Float32);

        let data = Box::new(MockTrainingSource::new(data).await);
        // Until DF handles NaN reliably we need to make sure we reject input
        // containing NaN
        assert!(
            train_btree_index(data, &sub_index_trainer, index_store.as_ref())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn btree_entire_null_page() {
        let tempdir = tempdir().unwrap();
        let index_store = legacy_test_store(&tempdir);
        let batch = gen()
            .col(
                "values",
                array::rand_utf8(ByteCount::from(0), false).with_nulls(&[true]),
            )
            .col("row_ids", array::step::<UInt64Type>())
            .into_batch_rows(RowCount::from(4096));
        assert_eq!(batch.as_ref().unwrap()["values"].null_count(), 4096);
        let batches = vec![batch];
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::Utf8, true),
            Field::new("row_ids", DataType::UInt64, false),
        ]));
        let data = RecordBatchIterator::new(batches, schema);
        let sub_index_trainer = FlatIndexMetadata::new(DataType::Utf8);

        let data = Box::new(MockTrainingSource::new(data).await);
        train_btree_index(data, &sub_index_trainer, index_store.as_ref())
            .await
            .unwrap();

        let index = BTreeIndex::load(index_store).await.unwrap();

        let row_ids = index
            .search(&SargableQuery::Equals(ScalarValue::Utf8(Some(
                "foo".to_string(),
            ))))
            .await
            .unwrap();
        assert!(row_ids.is_empty());

        let row_ids = index.search(&SargableQuery::IsNull()).await.unwrap();
        assert_eq!(row_ids.len(), Some(4096));
    }

    async fn train_bitmap(
        index_store: &Arc<dyn IndexStore>,
        data: impl RecordBatchReader + Send + Sync + 'static,
    ) {
        let data = Box::new(MockTrainingSource::new(data).await);
        train_bitmap_index(data, index_store.as_ref())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_bitmap_working() {
        let tempdir = tempdir().unwrap();
        let index_store = test_store(&tempdir);

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("values", DataType::Utf8, true),
            Field::new("row_ids", DataType::UInt64, false),
        ]));

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![Some("abcd"), None, Some("abcd")])),
                Arc::new(UInt64Array::from(vec![1, 2, 3])),
            ],
        )
        .unwrap();

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![
                    Some("apple"),
                    Some("hello"),
                    Some("abcd"),
                ])),
                Arc::new(UInt64Array::from(vec![4, 5, 6])),
            ],
        )
        .unwrap();

        let batches = vec![batch1, batch2];
        let data = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        train_bitmap(&index_store, data).await;

        let index = BitmapIndex::load(index_store).await.unwrap();

        let row_ids = index
            .search(&SargableQuery::Equals(ScalarValue::Utf8(None)))
            .await
            .unwrap();

        assert_eq!(Some(1), row_ids.len());
        assert!(row_ids.contains(2));

        let row_ids = index
            .search(&SargableQuery::Equals(ScalarValue::Utf8(Some(
                "abcd".to_string(),
            ))))
            .await
            .unwrap();

        assert_eq!(Some(3), row_ids.len());
        assert!(row_ids.contains(1));
        assert!(row_ids.contains(3));
        assert!(row_ids.contains(6));
    }

    #[tokio::test]
    async fn test_basic_bitmap() {
        let tempdir = tempdir().unwrap();
        let index_store = test_store(&tempdir);
        let data = gen()
            .col("values", array::step::<Int32Type>())
            .col("row_ids", array::step::<UInt64Type>())
            .into_reader_rows(RowCount::from(4096), BatchCount::from(100));
        train_bitmap(&index_store, data).await;
        let index = BitmapIndex::load(index_store).await.unwrap();

        let row_ids = index
            .search(&SargableQuery::Equals(ScalarValue::Int32(Some(10000))))
            .await
            .unwrap();

        assert_eq!(Some(1), row_ids.len());
        assert!(row_ids.contains(10000));

        let row_ids = index
            .search(&SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::Int32(Some(-100))),
            ))
            .await
            .unwrap();

        assert!(row_ids.is_empty());

        let row_ids = index
            .search(&SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::Int32(Some(100))),
            ))
            .await
            .unwrap();

        assert_eq!(Some(100), row_ids.len());
    }

    async fn check_bitmap(index: &BitmapIndex, query: SargableQuery, expected: &[u64]) {
        let results = index.search(&query).await.unwrap();
        let expected_arr = RowIdTreeMap::from_iter(expected);
        assert_eq!(results, expected_arr);
    }

    #[tokio::test]
    async fn test_bitmap_with_gaps() {
        let tempdir = tempdir().unwrap();
        let index_store = test_store(&tempdir);
        let batch_one = gen()
            .col("values", array::cycle::<Int32Type>(vec![0, 1, 4, 5]))
            .col("row_ids", array::cycle::<UInt64Type>(vec![0, 1, 2, 3]))
            .into_batch_rows(RowCount::from(4));
        let batch_two = gen()
            .col("values", array::cycle::<Int32Type>(vec![10, 11, 11, 15]))
            .col("row_ids", array::cycle::<UInt64Type>(vec![40, 50, 60, 70]))
            .into_batch_rows(RowCount::from(4));
        let batch_three = gen()
            .col("values", array::cycle::<Int32Type>(vec![15, 15, 15, 15]))
            .col(
                "row_ids",
                array::cycle::<UInt64Type>(vec![400, 500, 600, 700]),
            )
            .into_batch_rows(RowCount::from(4));
        let batch_four = gen()
            .col("values", array::cycle::<Int32Type>(vec![15, 16, 20, 20]))
            .col(
                "row_ids",
                array::cycle::<UInt64Type>(vec![4000, 5000, 6000, 7000]),
            )
            .into_batch_rows(RowCount::from(4));
        let batches = vec![batch_one, batch_two, batch_three, batch_four];
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::Int32, false),
            Field::new("row_ids", DataType::UInt64, false),
        ]));
        let data = RecordBatchIterator::new(batches, schema);
        train_bitmap(&index_store, data).await;
        let index = BitmapIndex::load(index_store).await.unwrap();

        // The above should create four pages
        //
        // 0 - 5
        // 10 - 15
        // 15 - 15
        // 15 - 20
        //
        // This will help us test various indexing corner cases

        // No results (off the left side)
        check_bitmap(
            &index,
            SargableQuery::Equals(ScalarValue::Int32(Some(-3))),
            &[],
        )
        .await;

        check_bitmap(
            &index,
            SargableQuery::Range(
                Bound::Unbounded,
                Bound::Included(ScalarValue::Int32(Some(-3))),
            ),
            &[],
        )
        .await;

        check_bitmap(
            &index,
            SargableQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(-10))),
                Bound::Included(ScalarValue::Int32(Some(-3))),
            ),
            &[],
        )
        .await;

        // Hitting the middle of a bucket
        check_bitmap(
            &index,
            SargableQuery::Equals(ScalarValue::Int32(Some(4))),
            &[2],
        )
        .await;

        // Hitting a gap between two buckets
        check_bitmap(
            &index,
            SargableQuery::Equals(ScalarValue::Int32(Some(7))),
            &[],
        )
        .await;

        // Hitting the lowest of the overlapping buckets
        check_bitmap(
            &index,
            SargableQuery::Equals(ScalarValue::Int32(Some(11))),
            &[50, 60],
        )
        .await;

        // Hitting the 15 shared on all three buckets
        check_bitmap(
            &index,
            SargableQuery::Equals(ScalarValue::Int32(Some(15))),
            &[70, 400, 500, 600, 700, 4000],
        )
        .await;

        // Hitting the upper part of the three overlapping buckets
        check_bitmap(
            &index,
            SargableQuery::Equals(ScalarValue::Int32(Some(20))),
            &[6000, 7000],
        )
        .await;

        // Ranges that capture multiple buckets
        check_bitmap(
            &index,
            SargableQuery::Range(
                Bound::Unbounded,
                Bound::Included(ScalarValue::Int32(Some(11))),
            ),
            &[0, 1, 2, 3, 40, 50, 60],
        )
        .await;

        check_bitmap(
            &index,
            SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::Int32(Some(11))),
            ),
            &[0, 1, 2, 3, 40],
        )
        .await;

        check_bitmap(
            &index,
            SargableQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(4))),
                Bound::Unbounded,
            ),
            &[
                2, 3, 40, 50, 60, 70, 400, 500, 600, 700, 4000, 5000, 6000, 7000,
            ],
        )
        .await;

        check_bitmap(
            &index,
            SargableQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(4))),
                Bound::Included(ScalarValue::Int32(Some(11))),
            ),
            &[2, 3, 40, 50, 60],
        )
        .await;

        check_bitmap(
            &index,
            SargableQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(4))),
                Bound::Excluded(ScalarValue::Int32(Some(11))),
            ),
            &[2, 3, 40],
        )
        .await;

        check_bitmap(
            &index,
            SargableQuery::Range(
                Bound::Excluded(ScalarValue::Int32(Some(4))),
                Bound::Unbounded,
            ),
            &[
                3, 40, 50, 60, 70, 400, 500, 600, 700, 4000, 5000, 6000, 7000,
            ],
        )
        .await;

        check_bitmap(
            &index,
            SargableQuery::Range(
                Bound::Excluded(ScalarValue::Int32(Some(4))),
                Bound::Included(ScalarValue::Int32(Some(11))),
            ),
            &[3, 40, 50, 60],
        )
        .await;

        check_bitmap(
            &index,
            SargableQuery::Range(
                Bound::Excluded(ScalarValue::Int32(Some(4))),
                Bound::Excluded(ScalarValue::Int32(Some(11))),
            ),
            &[3, 40],
        )
        .await;

        check_bitmap(
            &index,
            SargableQuery::Range(
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
    async fn test_bitmap_update() {
        let index_dir = tempdir().unwrap();
        let index_store = test_store(&index_dir);
        let data = gen()
            .col("values", array::step::<Int32Type>())
            .col("row_ids", array::step::<UInt64Type>())
            .into_reader_rows(RowCount::from(4096), BatchCount::from(1));
        train_bitmap(&index_store, data).await;
        let index = BitmapIndex::load(index_store).await.unwrap();

        let data = gen()
            .col("values", array::step_custom::<Int32Type>(4096, 1))
            .col("row_ids", array::step_custom::<UInt64Type>(4096, 1))
            .into_reader_rows(RowCount::from(4096), BatchCount::from(1));

        let updated_index_dir = tempdir().unwrap();
        let updated_index_store = test_store(&updated_index_dir);
        index
            .update(
                lance_datafusion::utils::reader_to_stream(Box::new(data)),
                updated_index_store.as_ref(),
            )
            .await
            .unwrap();
        let updated_index = BitmapIndex::load(updated_index_store).await.unwrap();

        let row_ids = updated_index
            .search(&SargableQuery::Equals(ScalarValue::Int32(Some(5000))))
            .await
            .unwrap();

        assert_eq!(Some(1), row_ids.len());
        assert!(row_ids.contains(5000));
    }

    #[tokio::test]
    async fn test_bitmap_remap() {
        let index_dir = tempdir().unwrap();
        let index_store = test_store(&index_dir);
        let data = gen()
            .col("values", array::step::<Int32Type>())
            .col("row_ids", array::step::<UInt64Type>())
            .into_reader_rows(RowCount::from(50), BatchCount::from(1));
        train_bitmap(&index_store, data).await;
        let index = BitmapIndex::load(index_store).await.unwrap();

        let mapping = (0..50)
            .map(|i| {
                let map_result = if i == 5 {
                    Some(65)
                } else if i == 7 {
                    None
                } else {
                    Some(i)
                };
                (i, map_result)
            })
            .collect::<HashMap<_, _>>();

        let remapped_dir = tempdir().unwrap();
        let remapped_store = test_store(&remapped_dir);
        index
            .remap(&mapping, remapped_store.as_ref())
            .await
            .unwrap();
        let remapped_index = BitmapIndex::load(remapped_store).await.unwrap();

        // Remapped to new value
        assert!(remapped_index
            .search(&SargableQuery::Equals(ScalarValue::Int32(Some(5))))
            .await
            .unwrap()
            .contains(65));
        // Deleted
        assert!(remapped_index
            .search(&SargableQuery::Equals(ScalarValue::Int32(Some(7))))
            .await
            .unwrap()
            .is_empty());
        // Not remapped
        assert!(remapped_index
            .search(&SargableQuery::Equals(ScalarValue::Int32(Some(3))))
            .await
            .unwrap()
            .contains(3));
    }

    async fn train_tag(
        index_store: &Arc<dyn IndexStore>,
        data: impl RecordBatchReader + Send + Sync + 'static,
    ) {
        let data = Box::new(MockTrainingSource::new(data).await);
        train_label_list_index(data, index_store.as_ref())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_label_list_index() {
        let tempdir = tempdir().unwrap();
        let index_store = test_store(&tempdir);
        let data = gen()
            .col(
                "values",
                array::rand_type(&DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::UInt8,
                    false,
                )))),
            )
            .col("row_ids", array::step::<UInt64Type>())
            .into_batch_rows(RowCount::from(40960))
            .unwrap();

        let batch_reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema().clone());

        // This is probably enough data that we can be assured each tag is used at least once
        train_tag(&index_store, batch_reader).await;

        // We scan through each list, if it was a match we run match_fn to check
        // if the match was correct if it was not a match we run no_match_fn to check
        // if the no-match was correct
        type MatchFn = Box<dyn Fn(&ScalarBuffer<u8>) -> bool>;
        let check = |query: LabelListQuery, match_fn: MatchFn, no_match_fn: MatchFn| {
            let index_store = index_store.clone();
            let data = data.clone();
            async move {
                let index = LabelListIndex::load(index_store).await.unwrap();
                let row_ids = index.search(&query).await.unwrap();

                let row_ids_set = row_ids
                    .row_ids()
                    .unwrap()
                    .map(u64::from)
                    .collect::<std::collections::HashSet<_>>();

                for (list, row_id) in data
                    .column(0)
                    .as_list::<i32>()
                    .iter()
                    .zip(data.column(1).as_primitive::<UInt64Type>())
                {
                    let list = list.unwrap();
                    let row_id = row_id.unwrap();
                    let vals = list.as_primitive::<UInt8Type>().values();
                    if row_ids_set.contains(&row_id) {
                        assert!(match_fn(vals));
                    } else {
                        assert!(no_match_fn(vals));
                    }
                }
            }
        };

        // Simple check for 1 value (doesn't matter intersection vs union)
        check(
            LabelListQuery::HasAnyLabel(vec![ScalarValue::UInt8(Some(1))]),
            Box::new(|vals| vals.iter().any(|val| *val == 1)),
            Box::new(|vals| vals.iter().all(|val| *val != 1)),
        )
        .await;
        check(
            LabelListQuery::HasAllLabels(vec![ScalarValue::UInt8(Some(1))]),
            Box::new(|vals| vals.iter().any(|val| *val == 1)),
            Box::new(|vals| vals.iter().all(|val| *val != 1)),
        )
        .await;
        // Set intersection
        check(
            LabelListQuery::HasAllLabels(vec![
                ScalarValue::UInt8(Some(1)),
                ScalarValue::UInt8(Some(2)),
            ]),
            // Match must have 1 and 2
            Box::new(|vals| vals.iter().any(|val| *val == 1) && vals.iter().any(|val| *val == 2)),
            // No-match must either not have 1 or not have 2
            Box::new(|vals| vals.iter().all(|val| *val != 1) || vals.iter().all(|val| *val != 2)),
        )
        .await;
        // Set union
        check(
            LabelListQuery::HasAnyLabel(vec![
                ScalarValue::UInt8(Some(1)),
                ScalarValue::UInt8(Some(2)),
            ]),
            // Match either have 1 or have 2
            Box::new(|vals| vals.iter().any(|val| *val == 1) || vals.iter().any(|val| *val == 2)),
            // No-match must not have 1 and not have 2
            Box::new(|vals| vals.iter().all(|val| *val != 1) && vals.iter().all(|val| *val != 2)),
        )
        .await;
    }
}
