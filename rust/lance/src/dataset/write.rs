// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use arrow_array::RecordBatchReader;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::StreamExt;
use lance_core::{datatypes::Schema, Error, Result};
use lance_datafusion::chunker::chunk_stream;
use lance_datafusion::utils::reader_to_stream;
use lance_file::writer::FileWriter;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_table::format::Fragment;
use lance_table::io::commit::CommitHandler;
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use snafu::{location, Location};
use tracing::instrument;
use uuid::Uuid;

use super::progress::{NoopFragmentWriteProgress, WriteFragmentProgress};
use super::DATA_DIR;

pub mod merge_insert;
pub mod update;

/// The mode to write dataset.
#[derive(Debug, Clone, Copy)]
pub enum WriteMode {
    /// Create a new dataset. Expect the dataset does not exist.
    Create,
    /// Append to an existing dataset.
    Append,
    /// Overwrite a dataset as a new version, or create new dataset if not exist.
    Overwrite,
}

impl TryFrom<&str> for WriteMode {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value.to_lowercase().as_str() {
            "create" => Ok(WriteMode::Create),
            "append" => Ok(WriteMode::Append),
            "overwrite" => Ok(WriteMode::Overwrite),
            _ => Err(Error::IO {
                message: format!("Invalid write mode: {}", value),
                location: location!(),
            }),
        }
    }
}

/// Dataset Write Parameters
#[derive(Debug, Clone)]
pub struct WriteParams {
    /// Max number of records per file.
    pub max_rows_per_file: usize,

    /// Max number of rows per row group.
    pub max_rows_per_group: usize,

    /// Max file size in bytes.
    ///
    /// This is a soft limit. The actual file size may be larger than this value
    /// by a few megabytes, since once we detect we hit this limit, we still
    /// need to flush the footer.
    ///
    /// This limit is checked after writing each group, so if max_rows_per_group
    /// is set to a large value, this limit may be exceeded by a large amount.
    ///
    /// The default is 90 GB. If you are using an object store such as S3, we
    /// currently have a hard 100 GB limit.
    pub max_bytes_per_file: usize,

    /// Write mode
    pub mode: WriteMode,

    pub store_params: Option<ObjectStoreParams>,

    pub progress: Arc<dyn WriteFragmentProgress>,

    /// If present, dataset will use this to update the latest version
    ///
    /// If not set, the default will be based on the object store.  Generally this will
    /// be RenameCommitHandler unless the object store does not handle atomic renames (e.g. S3)
    ///
    /// If a custom object store is provided (via store_params.object_store) then this
    /// must also be provided.
    pub commit_handler: Option<Arc<dyn CommitHandler>>,
}

impl Default for WriteParams {
    fn default() -> Self {
        Self {
            max_rows_per_file: 1024 * 1024, // 1 million
            max_rows_per_group: 1024,
            // object-store has a 100GB limit, so we should at least make sure
            // we are under that.
            max_bytes_per_file: 90 * 1024 * 1024 * 1024, // 90 GB
            mode: WriteMode::Create,
            store_params: None,
            progress: Arc::new(NoopFragmentWriteProgress::new()),
            commit_handler: None,
        }
    }
}

/// Writes the given data to the dataset and returns fragments.
///
/// NOTE: the fragments have not yet been assigned an ID. That must be done
/// by the caller. This is so this function can be called in parallel, and the
/// IDs can be assigned after writing is complete.
pub async fn write_fragments(
    dataset_uri: &str,
    data: impl RecordBatchReader + Send + 'static,
    params: WriteParams,
) -> Result<Vec<Fragment>> {
    let (object_store, base) = ObjectStore::from_uri_and_params(
        dataset_uri,
        &params.store_params.clone().unwrap_or_default(),
    )
    .await?;
    let (stream, schema) = reader_to_stream(Box::new(data)).await?;
    write_fragments_internal(Arc::new(object_store), &base, &schema, stream, params).await
}

/// Writes the given data to the dataset and returns fragments.
///
/// NOTE: the fragments have not yet been assigned an ID. That must be done
/// by the caller. This is so this function can be called in parallel, and the
/// IDs can be assigned after writing is complete.
///
/// This is a private variant that takes a `SendableRecordBatchStream` instead
/// of a reader. We don't expose the stream at our interface because it is a
/// DataFusion type.
#[instrument(level = "debug", skip_all)]
pub async fn write_fragments_internal(
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    schema: &Schema,
    data: SendableRecordBatchStream,
    mut params: WriteParams,
) -> Result<Vec<Fragment>> {
    // Make sure the max rows per group is not larger than the max rows per file
    params.max_rows_per_group = std::cmp::min(params.max_rows_per_group, params.max_rows_per_file);
    let mut buffered_reader = chunk_stream(data, params.max_rows_per_group);

    let writer_generator = WriterGenerator::new(object_store, base_dir, schema);
    let mut writer: Option<FileWriter<ManifestDescribing>> = None;
    let mut num_rows_in_current_file = 0;
    let mut fragments = Vec::new();
    while let Some(batch_chunk) = buffered_reader.next().await {
        let batch_chunk = batch_chunk?;

        if writer.is_none() {
            let (new_writer, new_fragment) = writer_generator.new_writer().await?;
            // rustc has a hard time analyzing the lifetime of the &str returned
            // by multipart_id(), so we convert it to an owned value here.
            let multipart_id = new_writer.multipart_id().to_string();
            params.progress.begin(&new_fragment, &multipart_id).await?;
            writer = Some(new_writer);
            fragments.push(new_fragment);
        }

        writer.as_mut().unwrap().write(&batch_chunk).await?;
        for batch in batch_chunk {
            num_rows_in_current_file += batch.num_rows();
        }

        if num_rows_in_current_file >= params.max_rows_per_file
            || writer.as_mut().unwrap().tell().await? >= params.max_bytes_per_file
        {
            let num_rows = writer.take().unwrap().finish().await?;
            debug_assert_eq!(num_rows, num_rows_in_current_file);
            params.progress.complete(fragments.last().unwrap()).await?;
            fragments.last_mut().unwrap().physical_rows = Some(num_rows);
            num_rows_in_current_file = 0;
        }
    }

    // Complete the final writer
    if let Some(mut writer) = writer.take() {
        let num_rows = writer.finish().await?;
        fragments.last_mut().unwrap().physical_rows = Some(num_rows);
    }

    Ok(fragments)
}

/// Creates new file writers for a given dataset.
struct WriterGenerator {
    object_store: Arc<ObjectStore>,
    base_dir: Path,
    schema: Schema,
}

impl WriterGenerator {
    pub fn new(object_store: Arc<ObjectStore>, base_dir: &Path, schema: &Schema) -> Self {
        Self {
            object_store,
            base_dir: base_dir.clone(),
            schema: schema.clone(),
        }
    }

    pub async fn new_writer(&self) -> Result<(FileWriter<ManifestDescribing>, Fragment)> {
        let data_file_path = format!("{}.lance", Uuid::new_v4());

        // Use temporary ID 0; will assign ID later.
        let mut fragment = Fragment::new(0);
        fragment.add_file(&data_file_path, &self.schema);

        let full_path = self.base_dir.child(DATA_DIR).child(data_file_path);
        let writer = FileWriter::try_new(
            self.object_store.as_ref(),
            &full_path,
            self.schema.clone(),
            &Default::default(),
        )
        .await?;

        Ok((writer, fragment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Schema as ArrowSchema};
    use datafusion::{error::DataFusionError, physical_plan::stream::RecordBatchStreamAdapter};
    use futures::TryStreamExt;

    #[tokio::test]
    async fn test_chunking_large_batches() {
        // Create a stream of 3 batches of 10 rows
        let schema = Arc::new(ArrowSchema::new(vec![arrow::datatypes::Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from_iter(0..28))])
                .unwrap();
        let batches: Vec<RecordBatch> =
            vec![batch.slice(0, 10), batch.slice(10, 10), batch.slice(20, 8)];
        let stream = RecordBatchStreamAdapter::new(
            schema.clone(),
            futures::stream::iter(batches.into_iter().map(Ok::<_, DataFusionError>)),
        );

        // Chunk into a stream of 3 row batches
        let chunks: Vec<Vec<RecordBatch>> = chunk_stream(Box::pin(stream), 3)
            .try_collect()
            .await
            .unwrap();

        assert_eq!(chunks.len(), 10);
        assert_eq!(chunks[0].len(), 1);

        for (i, chunk) in chunks.iter().enumerate() {
            let num_rows = chunk.iter().map(|batch| batch.num_rows()).sum::<usize>();
            if i < chunks.len() - 1 {
                assert_eq!(num_rows, 3);
            } else {
                // Last chunk is shorter
                assert_eq!(num_rows, 1);
            }
        }

        // The fourth chunk is split along the boundary between the original first
        // two batches.
        assert_eq!(chunks[3].len(), 2);
        assert_eq!(chunks[3][0].num_rows(), 1);
        assert_eq!(chunks[3][1].num_rows(), 2);
    }

    #[tokio::test]
    async fn test_chunking_small_batches() {
        // Create a stream of 10 batches of 3 rows
        let schema = Arc::new(ArrowSchema::new(vec![arrow::datatypes::Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from_iter(0..30))])
                .unwrap();

        let batches: Vec<RecordBatch> = (0..10).map(|i| batch.slice(i * 3, 3)).collect();
        let stream = RecordBatchStreamAdapter::new(
            schema.clone(),
            futures::stream::iter(batches.into_iter().map(Ok::<_, DataFusionError>)),
        );

        // Chunk into a stream of 10 row batches
        let chunks: Vec<Vec<RecordBatch>> = chunk_stream(Box::pin(stream), 10)
            .try_collect()
            .await
            .unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 4);
        assert_eq!(chunks[0][0], batch.slice(0, 3));
        assert_eq!(chunks[0][1], batch.slice(3, 3));
        assert_eq!(chunks[0][2], batch.slice(6, 3));
        assert_eq!(chunks[0][3], batch.slice(9, 1));

        for chunk in &chunks {
            let num_rows = chunk.iter().map(|batch| batch.num_rows()).sum::<usize>();
            assert_eq!(num_rows, 10);
        }
    }

    #[tokio::test]
    async fn test_file_size() {
        let schema = Arc::new(ArrowSchema::new(vec![arrow::datatypes::Field::new(
            "a",
            DataType::Int32,
            false,
        )]));

        // Write 1024 rows and show they are split into two files
        // 512 * 4 bytes = 2KB
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter(0..1024))],
        )
        .unwrap();

        let write_params = WriteParams {
            max_rows_per_file: 1024 * 10, // Won't be limited by this
            max_rows_per_group: 512,
            max_bytes_per_file: 2 * 1024,
            mode: WriteMode::Create,
            ..Default::default()
        };

        let data_stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            futures::stream::iter(std::iter::once(Ok(batch))),
        ));

        let schema = Schema::try_from(schema.as_ref()).unwrap();

        let object_store = Arc::new(ObjectStore::memory());
        let fragments = write_fragments_internal(
            object_store,
            &Path::from("test"),
            &schema,
            data_stream,
            write_params,
        )
        .await
        .unwrap();
        assert_eq!(fragments.len(), 2);
    }
}
