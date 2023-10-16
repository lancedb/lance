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
use datafusion::error::DataFusionError;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{StreamExt, TryStreamExt};
use object_store::path::Path;
use tracing::instrument;
use uuid::Uuid;

use super::progress::WriteFragmentProgress;
use super::{chunker::chunk_stream, DATA_DIR};
use crate::error::Result;
use crate::Error;
use crate::{
    datatypes::Schema,
    format::Fragment,
    io::{object_store::ObjectStoreParams, FileWriter, ObjectStore},
};

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

/// Dataset Write Parameters
#[derive(Debug, Clone)]
pub struct WriteParams {
    /// Max number of records per file.
    pub max_rows_per_file: usize,

    /// Max number of rows per row group.
    pub max_rows_per_group: usize,

    /// Write mode
    pub mode: WriteMode,

    pub store_params: Option<ObjectStoreParams>,

    pub progress: Option<Arc<dyn WriteFragmentProgress>>,

    /// If the last fragment is smaller than this size
    /// Any new write will be compacted into the last fragment
    /// Has no effect if the last fragment is covered by an index
    pub tail_compaction_size: Option<usize>,
}

impl Default for WriteParams {
    fn default() -> Self {
        Self {
            max_rows_per_file: 1024 * 1024, // 1 million
            max_rows_per_group: 1024,
            mode: WriteMode::Create,
            store_params: None,
            progress: None,
            tail_compaction_size: None,
        }
    }
}

/// Convert reader to a stream and a schema.
///
/// Will peek the first batch to get the dictionaries for dictionary columns.
///
/// NOTE: this does not validate the schema. For example, for appends the schema
/// should be checked to make sure it matches the existing dataset schema before
/// writing.
pub fn reader_to_stream(
    batches: Box<dyn RecordBatchReader + Send>,
) -> Result<(SendableRecordBatchStream, Schema)> {
    let arrow_schema = batches.schema();
    let mut schema: Schema = Schema::try_from(batches.schema().as_ref())?;
    let mut peekable = batches.peekable();
    if let Some(batch) = peekable.peek() {
        if let Ok(b) = batch {
            schema.set_dictionary(b)?;
        } else {
            return Err(Error::from(batch.as_ref().unwrap_err()));
        }
    }
    schema.validate()?;

    let stream = RecordBatchStreamAdapter::new(
        arrow_schema,
        futures::stream::iter(peekable).map_err(DataFusionError::from),
    );
    let stream = Box::pin(stream) as SendableRecordBatchStream;

    Ok((stream, schema))
}

/// Writes the given data to the dataset and returns fragments.
///
/// NOTE: the fragments have not yet been assigned an ID. That must be done
/// by the caller. This is so this function can be called in parallel, and the
/// IDs can be assigned after writing is complete.
#[instrument(skip_all)]
pub async fn write_fragments(
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
    let mut writer: Option<FileWriter> = None;
    let mut num_rows_in_current_file = 0;
    let mut fragments = Vec::new();
    while let Some(batch_chunk) = buffered_reader.next().await {
        let batch_chunk = batch_chunk?;

        if writer.is_none() {
            let (new_writer, new_fragment) = writer_generator.new_writer().await?;
            writer = Some(new_writer);
            fragments.push(new_fragment);
        }

        writer.as_mut().unwrap().write(&batch_chunk).await?;
        for batch in batch_chunk {
            num_rows_in_current_file += batch.num_rows();
        }

        if num_rows_in_current_file >= params.max_rows_per_file {
            let num_rows = writer.take().unwrap().finish().await?;
            debug_assert_eq!(num_rows, num_rows_in_current_file);
            fragments.last_mut().unwrap().physical_rows = num_rows;
            num_rows_in_current_file = 0;
        }
    }

    // Complete the final writer
    if let Some(mut writer) = writer.take() {
        let num_rows = writer.finish().await?;
        fragments.last_mut().unwrap().physical_rows = num_rows;
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

    pub async fn new_writer(&self) -> Result<(FileWriter, Fragment)> {
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
}
