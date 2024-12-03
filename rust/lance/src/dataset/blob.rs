// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{future::Future, ops::DerefMut, sync::Arc};

use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use arrow_schema::DataType;
use datafusion::execution::SendableRecordBatchStream;
use futures::StreamExt;
use lance_core::{
    datatypes::{Schema, StorageClass, BLOB_META_KEY},
    error::CloneableResult,
    utils::{
        address::RowAddress,
        futures::{Capacity, SharedStreamExt},
    },
    Error, Result,
};
use lance_io::traits::Reader;
use object_store::path::Path;
use snafu::{location, Location};
use tokio::sync::Mutex;

use crate::io::exec::{ShareableRecordBatchStream, ShareableRecordBatchStreamAdapter};

use super::Dataset;

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
        let data_file = frag.data_file_for_field(field_id).unwrap().path.clone();
        let data_file = dataset.data_dir().child(data_file);
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
    if blob_field.data_type() != DataType::LargeBinary
        || !projection.fields[0].metadata.contains_key(BLOB_META_KEY)
    {
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
    let positions = descriptions.column(0).as_primitive::<UInt64Type>();
    let sizes = descriptions.column(1).as_primitive::<UInt64Type>();
    let row_addrs = description_and_addr.column(1).as_primitive::<UInt64Type>();

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
            BlobFile::new(
                dataset.clone(),
                blob_field_id as u32,
                row_addr,
                position,
                size,
            )
        })
        .collect())
}

pub trait BlobStreamExt: Sized {
    /// Splits a stream into a regular portion (the first stream)
    /// and a blob portion (the second stream)
    ///
    /// The first stream contains all fields with the default storage class and
    /// may be identical to self.
    ///
    /// The second stream may be None (if there are no fields with the blob storage class)
    /// or it contains all fields with the blob storage class.
    fn extract_blob_stream(self, schema: &Schema) -> (Self, Option<Self>);
}

impl BlobStreamExt for SendableRecordBatchStream {
    fn extract_blob_stream(self, schema: &Schema) -> (Self, Option<Self>) {
        let mut indices_with_blob = Vec::with_capacity(schema.fields.len());
        let mut indices_without_blob = Vec::with_capacity(schema.fields.len());
        for (idx, field) in schema.fields.iter().enumerate() {
            if field.storage_class() == StorageClass::Blob {
                indices_with_blob.push(idx);
            } else {
                indices_without_blob.push(idx);
            }
        }
        if indices_with_blob.is_empty() {
            (self, None)
        } else {
            let left_schema = Arc::new(self.schema().project(&indices_without_blob).unwrap());
            let right_schema = Arc::new(self.schema().project(&indices_with_blob).unwrap());

            let (left, right) = ShareableRecordBatchStream(self)
                .boxed()
                // If we are working with blobs then we are probably working with rather large batches
                // We don't want to read too far ahead.
                .share(Capacity::Bounded(1));

            let left = left.map(move |batch| match batch {
                CloneableResult(Ok(batch)) => {
                    CloneableResult(Ok(batch.project(&indices_without_blob).unwrap()))
                }
                CloneableResult(Err(err)) => CloneableResult(Err(err)),
            });

            let right = right.map(move |batch| match batch {
                CloneableResult(Ok(batch)) => {
                    CloneableResult(Ok(batch.project(&indices_with_blob).unwrap()))
                }
                CloneableResult(Err(err)) => CloneableResult(Err(err)),
            });

            let left = ShareableRecordBatchStreamAdapter::new(left_schema, left);
            let right = ShareableRecordBatchStreamAdapter::new(right_schema, right);
            (Box::pin(left), Some(Box::pin(right)))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{array::AsArray, datatypes::UInt64Type};
    use arrow_array::RecordBatch;
    use lance_core::{Error, Result};
    use lance_datagen::{array, BatchCount, RowCount};
    use lance_file::version::LanceFileVersion;
    use tempfile::{tempdir, TempDir};

    use crate::{utils::test::TestDatasetGenerator, Dataset};

    struct BlobTestFixture {
        _test_dir: TempDir,
        dataset: Arc<Dataset>,
        data: Vec<RecordBatch>,
    }

    impl BlobTestFixture {
        async fn new() -> Self {
            let test_dir = tempdir().unwrap();
            let test_uri = test_dir.path().to_str().unwrap();

            let data = lance_datagen::gen()
                .col("filterme", array::step::<UInt64Type>())
                .col("blobs", array::blob())
                .into_reader_rows(RowCount::from(10), BatchCount::from(10))
                .map(|batch| Ok(batch?))
                .collect::<Result<Vec<_>>>()
                .unwrap();

            let dataset = Arc::new(
                TestDatasetGenerator::new(data.clone(), LanceFileVersion::default())
                    .make_hostile(test_uri)
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
}
