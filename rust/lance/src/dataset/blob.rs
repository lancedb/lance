// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, future::Future, ops::DerefMut, sync::Arc};

use arrow::array::AsArray;
use arrow::datatypes::{UInt32Type, UInt64Type, UInt8Type};
use object_store::path::Path;
use snafu::location;
use tokio::sync::Mutex;

use super::take::TakeBuilder;
use super::{Dataset, ProjectionRequest};
use arrow_array::{Array, StructArray};
use lance_core::datatypes::BlobVersion;
use lance_core::{utils::address::RowAddress, Error, Result};
use lance_io::traits::Reader;

pub const BLOB_VERSION_CONFIG_KEY: &str = "lance.blob.version";

pub fn blob_version_from_config(config: &HashMap<String, String>) -> BlobVersion {
    config
        .get(BLOB_VERSION_CONFIG_KEY)
        .and_then(|value| BlobVersion::from_config_value(value))
        .unwrap_or(BlobVersion::V1)
}

/// Current state of the reader.  Held in a mutex for easy sharing
///
/// The u64 is the cursor in the file that the reader is currently at
/// (note that seeks are allowed before the file is opened)
#[derive(Debug)]
enum ReaderState {
    Uninitialized(u64),
    Open((u64, Arc<dyn Reader>)),
    Closed,
}

/// A file-like object that represents a blob in a dataset
#[derive(Debug)]
pub struct BlobFile {
    dataset: Arc<Dataset>,
    reader: Arc<Mutex<ReaderState>>,
    data_file: Path,
    position: u64,
    size: u64,
}

impl BlobFile {
    /// Create a new BlobFile
    ///
    /// See [`crate::dataset::Dataset::take_blobs`]
    pub fn new(
        dataset: Arc<Dataset>,
        field_id: u32,
        row_addr: u64,
        position: u64,
        size: u64,
    ) -> Self {
        let frag_id = RowAddress::from(row_addr).fragment_id();
        let frag = dataset.get_fragment(frag_id as usize).unwrap();
        let data_file = frag.data_file_for_field(field_id).unwrap();
        let data_file = dataset.data_dir().child(data_file.path.as_str());
        Self {
            dataset,
            data_file,
            position,
            size,
            reader: Arc::new(Mutex::new(ReaderState::Uninitialized(0))),
        }
    }

    /// Close the blob file, releasing any associated resources
    pub async fn close(&self) -> Result<()> {
        let mut reader = self.reader.lock().await;
        *reader = ReaderState::Closed;
        Ok(())
    }

    /// Returns true if the blob file is closed
    pub async fn is_closed(&self) -> bool {
        matches!(*self.reader.lock().await, ReaderState::Closed)
    }

    async fn do_with_reader<
        T,
        Fut: Future<Output = Result<(u64, T)>>,
        Func: FnOnce(u64, Arc<dyn Reader>) -> Fut,
    >(
        &self,
        func: Func,
    ) -> Result<T> {
        let mut reader = self.reader.lock().await;
        if let ReaderState::Uninitialized(cursor) = *reader {
            let opened = self.dataset.object_store.open(&self.data_file).await?;
            let opened = Arc::<dyn Reader>::from(opened);
            *reader = ReaderState::Open((cursor, opened.clone()));
        }
        match reader.deref_mut() {
            ReaderState::Open((cursor, reader)) => {
                let (new_cursor, data) = func(*cursor, reader.clone()).await?;
                *cursor = new_cursor;
                Ok(data)
            }
            ReaderState::Closed => Err(Error::IO {
                location: location!(),
                source: "Blob file is already closed".into(),
            }),
            _ => unreachable!(),
        }
    }

    /// Read the entire blob file from the current cursor position
    /// to the end of the file
    ///
    /// After this call the cursor will be pointing to the end of
    /// the file.
    pub async fn read(&self) -> Result<bytes::Bytes> {
        let position = self.position;
        let size = self.size;
        self.do_with_reader(|cursor, reader| async move {
            let start = position as usize + cursor as usize;
            let end = (position + size) as usize;
            Ok((end as u64, reader.get_range(start..end).await?))
        })
        .await
    }

    /// Read up to `len` bytes from the current cursor position
    ///
    /// After this call the cursor will be pointing to the end of
    /// the read data.
    pub async fn read_up_to(&self, len: usize) -> Result<bytes::Bytes> {
        let position = self.position;
        let size = self.size;
        self.do_with_reader(|cursor, reader| async move {
            let start = position as usize + cursor as usize;
            let read_size = len.min((size - cursor) as usize);
            let end = start + read_size;
            let data = reader.get_range(start..end).await?;
            Ok((end as u64 - position, data))
        })
        .await
    }

    /// Seek to a new cursor position in the file
    pub async fn seek(&self, new_cursor: u64) -> Result<()> {
        let mut reader = self.reader.lock().await;
        match reader.deref_mut() {
            ReaderState::Open((cursor, _)) => {
                *cursor = new_cursor;
                Ok(())
            }
            ReaderState::Closed => Err(Error::IO {
                location: location!(),
                source: "Blob file is already closed".into(),
            }),
            ReaderState::Uninitialized(cursor) => {
                *cursor = new_cursor;
                Ok(())
            }
        }
    }

    /// Return the current cursor position in the file
    pub async fn tell(&self) -> Result<u64> {
        let reader = self.reader.lock().await;
        match *reader {
            ReaderState::Open((cursor, _)) => Ok(cursor),
            ReaderState::Closed => Err(Error::IO {
                location: location!(),
                source: "Blob file is already closed".into(),
            }),
            ReaderState::Uninitialized(cursor) => Ok(cursor),
        }
    }

    /// Return the size of the blob file in bytes
    pub fn size(&self) -> u64 {
        self.size
    }
}

pub(super) async fn take_blobs(
    dataset: &Arc<Dataset>,
    row_ids: &[u64],
    column: &str,
) -> Result<Vec<BlobFile>> {
    let projection = dataset.schema().project(&[column])?;
    let blob_field = &projection.fields[0];
    let blob_field_id = blob_field.id;
    if !projection.fields[0].is_blob() {
        return Err(Error::InvalidInput {
            location: location!(),
            source: format!("the column '{}' is not a blob column", column).into(),
        });
    }
    let description_and_addr = dataset
        .take_builder(row_ids, projection)?
        .with_row_address(true)
        .execute()
        .await?;
    let descriptions = description_and_addr.column(0).as_struct();
    let row_addrs = description_and_addr.column(1).as_primitive::<UInt64Type>();
    let blob_field_id = blob_field_id as u32;

    match dataset.blob_version() {
        BlobVersion::V1 => collect_blob_files_v1(dataset, blob_field_id, descriptions, row_addrs),
        BlobVersion::V2 => collect_blob_files_v2(dataset, blob_field_id, descriptions, row_addrs),
    }
}

/// Take [BlobFile] by row addresses.
///
/// Row addresses are `u64` values encoding `(fragment_id << 32) | row_offset`.
/// Use this method when you already have row addresses, for example from
/// a scan with `with_row_address()`. For row IDs (stable identifiers), use
/// [`Dataset::take_blobs`]. For row indices (offsets), use
/// [`Dataset::take_blobs_by_indices`].
pub async fn take_blobs_by_addresses(
    dataset: &Arc<Dataset>,
    row_addrs: &[u64],
    column: &str,
) -> Result<Vec<BlobFile>> {
    let projection = dataset.schema().project(&[column])?;
    let blob_field = &projection.fields[0];
    let blob_field_id = blob_field.id;
    if !projection.fields[0].is_blob() {
        return Err(Error::InvalidInput {
            location: location!(),
            source: format!("the column '{}' is not a blob column", column).into(),
        });
    }

    // Convert Schema to ProjectionPlan
    let projection_request = ProjectionRequest::from(projection);
    let projection_plan = Arc::new(projection_request.into_projection_plan(dataset.clone())?);

    // Use try_new_from_addresses to bypass row ID index lookup.
    // This is critical when enable_stable_row_ids=true because row addresses
    // (fragment_id << 32 | row_offset) are different from row IDs (sequential integers).
    let description_and_addr =
        TakeBuilder::try_new_from_addresses(dataset.clone(), row_addrs.to_vec(), projection_plan)?
            .with_row_address(true)
            .execute()
            .await?;

    let descriptions = description_and_addr.column(0).as_struct();
    let row_addrs_result = description_and_addr.column(1).as_primitive::<UInt64Type>();
    let blob_field_id = blob_field_id as u32;

    match dataset.blob_version() {
        BlobVersion::V1 => {
            collect_blob_files_v1(dataset, blob_field_id, descriptions, row_addrs_result)
        }
        BlobVersion::V2 => {
            collect_blob_files_v2(dataset, blob_field_id, descriptions, row_addrs_result)
        }
    }
}

const INLINE_BLOB_KIND: u8 = 0;

fn collect_blob_files_v1(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    descriptions: &StructArray,
    row_addrs: &arrow::array::PrimitiveArray<UInt64Type>,
) -> Result<Vec<BlobFile>> {
    let positions = descriptions.column(0).as_primitive::<UInt64Type>();
    let sizes = descriptions.column(1).as_primitive::<UInt64Type>();

    Ok(row_addrs
        .values()
        .iter()
        .zip(positions.iter())
        .zip(sizes.iter())
        .filter_map(|((row_addr, position), size)| {
            let position = position?;
            let size = size?;
            Some((*row_addr, position, size))
        })
        .map(|(row_addr, position, size)| {
            BlobFile::new(dataset.clone(), blob_field_id, row_addr, position, size)
        })
        .collect())
}

fn collect_blob_files_v2(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    descriptions: &StructArray,
    row_addrs: &arrow::array::PrimitiveArray<UInt64Type>,
) -> Result<Vec<BlobFile>> {
    let kinds = descriptions.column(0).as_primitive::<UInt8Type>();
    let positions = descriptions.column(1).as_primitive::<UInt64Type>();
    let sizes = descriptions.column(2).as_primitive::<UInt64Type>();
    let _blob_ids = descriptions.column(3).as_primitive::<UInt32Type>();
    let _uris = descriptions.column(4).as_string::<i32>();

    let mut files = Vec::with_capacity(row_addrs.len());
    for (idx, row_addr) in row_addrs.values().iter().enumerate() {
        if kinds.is_null(idx) {
            // Null row
            continue;
        }
        let kind = kinds.value(idx);
        match kind {
            INLINE_BLOB_KIND => {
                if positions.is_null(idx) || sizes.is_null(idx) {
                    continue;
                }
                let position = positions.value(idx);
                let size = sizes.value(idx);
                files.push(BlobFile::new(
                    dataset.clone(),
                    blob_field_id,
                    *row_addr,
                    position,
                    size,
                ));
            }
            other => {
                return Err(Error::NotSupported {
                    source: format!("Blob kind {} is not supported", other).into(),
                    location: location!(),
                });
            }
        }
    }

    Ok(files)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{array::AsArray, datatypes::UInt64Type};
    use arrow_array::RecordBatch;
    use futures::TryStreamExt;
    use lance_arrow::DataTypeExt;
    use lance_io::stream::RecordBatchStream;

    use lance_core::{utils::tempfile::TempStrDir, Error, Result};
    use lance_datagen::{array, BatchCount, RowCount};
    use lance_file::version::LanceFileVersion;

    use crate::{utils::test::TestDatasetGenerator, Dataset};

    struct BlobTestFixture {
        _test_dir: TempStrDir,
        dataset: Arc<Dataset>,
        data: Vec<RecordBatch>,
    }

    impl BlobTestFixture {
        async fn new() -> Self {
            let test_dir = TempStrDir::default();

            let data = lance_datagen::gen_batch()
                .col("filterme", array::step::<UInt64Type>())
                .col("blobs", array::blob())
                .into_reader_rows(RowCount::from(10), BatchCount::from(10))
                .map(|batch| Ok(batch?))
                .collect::<Result<Vec<_>>>()
                .unwrap();

            let dataset = Arc::new(
                TestDatasetGenerator::new(data.clone(), LanceFileVersion::default())
                    .make_hostile(&test_dir)
                    .await,
            );

            Self {
                _test_dir: test_dir,
                dataset,
                data,
            }
        }
    }

    #[tokio::test]
    pub async fn test_take_blobs() {
        let fixture = BlobTestFixture::new().await;

        let row_ids = fixture
            .dataset
            .scan()
            .project::<String>(&[])
            .unwrap()
            .filter("filterme >= 50")
            .unwrap()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let row_ids = row_ids.column(0).as_primitive::<UInt64Type>().values();
        let row_ids = vec![row_ids[5], row_ids[17], row_ids[33]];

        let blobs = fixture.dataset.take_blobs(&row_ids, "blobs").await.unwrap();

        for (actual_idx, (expected_batch_idx, expected_row_idx)) in
            [(5, 5), (6, 7), (8, 3)].iter().enumerate()
        {
            let val = blobs[actual_idx].read().await.unwrap();
            let expected = fixture.data[*expected_batch_idx]
                .column(1)
                .as_binary::<i64>()
                .value(*expected_row_idx);

            assert_eq!(&val, expected);
        }
    }

    #[tokio::test]
    pub async fn test_take_blobs_by_indices() {
        let fixture = BlobTestFixture::new().await;

        let fragments = fixture.dataset.fragments();
        assert!(fragments.len() >= 2);
        let mut indices = Vec::with_capacity(fragments.len());
        let mut last = 2;

        for frag in fragments.iter() {
            indices.push(last as u64);
            last += frag.num_rows().unwrap_or(0);
        }
        indices.pop();

        // Row indices
        assert_eq!(indices, [2, 12, 22, 32, 42, 52, 62, 72, 82]);
        let blobs = fixture
            .dataset
            .take_blobs_by_indices(&indices, "blobs")
            .await
            .unwrap();

        // Row IDs
        let row_ids = fragments
            .iter()
            .map(|frag| (frag.id << 32) + 2)
            .collect::<Vec<_>>();
        let blobs2 = fixture.dataset.take_blobs(&row_ids, "blobs").await.unwrap();

        for (blob1, blob2) in blobs.iter().zip(blobs2.iter()) {
            assert_eq!(blob1.position, blob2.position);
            assert_eq!(blob1.size, blob2.size);
            assert_eq!(blob1.data_file, blob2.data_file);
        }
    }

    #[tokio::test]
    pub async fn test_take_blob_id_not_exist() {
        let fixture = BlobTestFixture::new().await;

        let err = fixture.dataset.take_blobs(&[1000], "blobs").await;

        assert!(matches!(err, Err(Error::InvalidInput { .. })));
    }

    #[tokio::test]
    pub async fn test_take_blob_not_blob_col() {
        let fixture = BlobTestFixture::new().await;

        let err = fixture.dataset.take_blobs(&[0], "filterme").await;

        assert!(matches!(err, Err(Error::InvalidInput { .. })));
        assert!(err.unwrap_err().to_string().contains("not a blob column"));
    }

    #[tokio::test]
    pub async fn test_scan_blobs() {
        let fixture = BlobTestFixture::new().await;

        // By default, scanning a blob column will load descriptions
        let batches = fixture
            .dataset
            .scan()
            .project(&["blobs"])
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();

        let schema = batches.schema();

        assert!(schema.fields[0].data_type().is_struct());

        let batches = batches.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(batches.len(), 10);
        for batch in batches.iter() {
            assert_eq!(batch.num_columns(), 1);
            assert!(batch.column(0).data_type().is_struct());
        }

        // Should also be able to scan with filter
        let batches = fixture
            .dataset
            .scan()
            .project(&["blobs"])
            .unwrap()
            .filter("filterme = 50")
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();

        let schema = batches.schema();

        assert!(schema.fields[0].data_type().is_struct());

        let batches = batches.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(batches.len(), 1);
        for batch in batches.iter() {
            assert_eq!(batch.num_columns(), 1);
            assert!(batch.column(0).data_type().is_struct());
        }
    }

    /// Test that take_blobs_by_indices works correctly with enable_stable_row_ids=true.
    ///
    /// This is a regression test for a bug where take_blobs_by_indices would fail
    /// with "index out of bounds" for fragment 1+ when stable row IDs are enabled.
    /// The bug was caused by passing row addresses (from row_offsets_to_row_addresses)
    /// to blob::take_blobs which expected row IDs. When stable row IDs are enabled,
    /// row addresses (fragment_id << 32 | offset) are different from row IDs
    /// (sequential integers), causing the row ID index lookup to fail for fragment 1+.
    #[tokio::test]
    pub async fn test_take_blobs_by_indices_with_stable_row_ids() {
        use crate::dataset::WriteParams;
        use arrow_array::RecordBatchIterator;

        let test_dir = TempStrDir::default();

        // Create test data with blob column
        let data = lance_datagen::gen_batch()
            .col("filterme", array::step::<UInt64Type>())
            .col("blobs", array::blob())
            .into_reader_rows(RowCount::from(6), BatchCount::from(1))
            .map(|batch| Ok(batch.unwrap()))
            .collect::<Result<Vec<_>>>()
            .unwrap();

        // Write with enable_stable_row_ids=true and force multiple fragments
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 3, // Force 2 fragments with 3 rows each
            ..Default::default()
        };

        let reader = RecordBatchIterator::new(data.clone().into_iter().map(Ok), data[0].schema());
        let dataset = Arc::new(
            Dataset::write(reader, &test_dir, Some(write_params))
                .await
                .unwrap(),
        );

        // Verify we have multiple fragments
        let fragments = dataset.fragments();
        assert!(
            fragments.len() >= 2,
            "Expected at least 2 fragments, got {}",
            fragments.len()
        );

        // Test first fragment (indices 0, 1, 2) - this always worked
        let blobs = dataset
            .take_blobs_by_indices(&[0, 1, 2], "blobs")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 3, "First fragment blobs should have 3 items");

        // Verify we can read the blob content
        for blob in &blobs {
            let content = blob.read().await.unwrap();
            assert!(!content.is_empty(), "Blob content should not be empty");
        }

        // Test second fragment (indices 3, 4, 5) - this was failing before the fix
        let blobs = dataset
            .take_blobs_by_indices(&[3, 4, 5], "blobs")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 3, "Second fragment blobs should have 3 items");

        // Verify we can read the blob content from second fragment
        for blob in &blobs {
            let content = blob.read().await.unwrap();
            assert!(!content.is_empty(), "Blob content should not be empty");
        }

        // Test mixed indices from both fragments
        let blobs = dataset
            .take_blobs_by_indices(&[1, 4], "blobs")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 2, "Mixed fragment blobs should have 2 items");
    }
}
