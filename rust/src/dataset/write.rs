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

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use arrow_select::concat::concat_batches;
use datafusion::datasource::file_format::FileWriterMode;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{Stream, StreamExt};
use object_store::path::Path;
use uuid::Uuid;

use crate::error::Result;
use crate::{
    datatypes::Schema,
    format::Fragment,
    io::{object_store::ObjectStoreParams, FileWriter, ObjectStore},
    Dataset,
};

use super::DATA_DIR;

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
}

impl Default for WriteParams {
    fn default() -> Self {
        Self {
            max_rows_per_file: 1024 * 1024, // 1 million
            max_rows_per_group: 1024,
            mode: WriteMode::Create,
            store_params: None,
        }
    }
}

/// Writes the given data to the dataset and returns fragments.
///
/// NOTE: the fragments have not yet been assigned an ID. That must be done
/// by the caller. This is so this function can be called in parallel, and the
/// IDs can be assigned after writing is complete.
pub(crate) async fn write_fragments(
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    schema: &Schema,
    data: SendableRecordBatchStream,
    params: WriteParams,
) -> Result<Vec<Fragment>> {
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
            writer.take().unwrap().finish().await?;
            num_rows_in_current_file = 0;
        }
    }

    // Complete the final writer
    if let Some(mut writer) = writer.take() {
        writer.finish().await?;
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
        let writer =
            FileWriter::try_new(self.object_store.as_ref(), &full_path, self.schema.clone())
                .await?;

        Ok((writer, fragment))
    }
}

/// Create a new [FileWriter] with the related `data_file_path` under `<DATA_DIR>`.
async fn new_file_writer(
    object_store: &ObjectStore,
    base_dir: &Path,
    data_file_path: &str,
    schema: &Schema,
) -> Result<FileWriter> {
    let full_path = base_dir.child(DATA_DIR).child(data_file_path);
    FileWriter::try_new(object_store, &full_path, schema.clone()).await
}

fn chunk_stream(
    stream: SendableRecordBatchStream,
    chunk_size: usize,
) -> Pin<Box<dyn Stream<Item = Result<Vec<RecordBatch>>> + Send>> {
    let chunker = BatchReaderChunker::new(stream, chunk_size);
    futures::stream::unfold(chunker, |mut chunker| async move {
        match chunker.next().await {
            Some(Ok(batches)) => Some((Ok(batches), chunker)),
            Some(Err(e)) => Some((Err(e), chunker)),
            None => None,
        }
    })
    .boxed()
}

/// Wraps a RecordBatchReader into an iterator of RecordBatch chunks of a given size.
/// This slices but does not copy any buffers.
struct BatchReaderChunker {
    inner: SendableRecordBatchStream,
    buffered: VecDeque<RecordBatch>,
    output_size: usize,
    i: usize,
}

impl BatchReaderChunker {
    pub fn new(inner: SendableRecordBatchStream, output_size: usize) -> Self {
        Self {
            inner,
            buffered: VecDeque::new(),
            output_size,
            i: 0,
        }
    }

    fn buffered_len(&self) -> usize {
        let buffer_total: usize = self.buffered.iter().map(|batch| batch.num_rows()).sum();
        buffer_total - self.i
    }

    async fn fill_buffer(&mut self) -> Result<()> {
        while self.buffered_len() < self.output_size {
            match self.inner.next().await {
                Some(Ok(batch)) => self.buffered.push_back(batch),
                Some(Err(e)) => return Err(e.into()),
                None => break,
            }
        }
        Ok(())
    }

    fn clean_buffer(&mut self) -> Result<()> {
        while self
            .buffered
            .get(0)
            .map_or(false, |batch| self.i >= batch.num_rows())
        {
            self.i -= self.buffered.pop_front().unwrap().num_rows();
        }
        Ok(())
    }

    async fn next(&mut self) -> Option<Result<Vec<RecordBatch>>> {
        match self.fill_buffer().await {
            Ok(_) => {}
            Err(e) => return Some(Err(e)),
        };

        // Always starting within the first batch, since otherwise we would have
        // dropped it.
        let mut ending_batch = 0;
        let mut ending_i = self.i + self.output_size;
        for batch in self.buffered.iter() {
            if ending_i <= batch.num_rows() {
                break;
            }
            ending_i -= batch.num_rows();
            ending_batch += 1;
        }

        let mut batches = Vec::new();
        if ending_batch == 0 {
            // It's all in the first batch, so we can just slice it zero-copy
            let batch = self.buffered[0].slice(self.i, ending_i);
            batches.push(batch);
        } else {
            let mut start = self.i;
            for batch in self.buffered.iter().take(ending_batch) {
                batches.push(batch.slice(start, batch.num_rows()));
                start = 0;
            }
            batches.push(self.buffered[ending_batch].slice(start, ending_i));
        };

        self.i = ending_i;
        match self.clean_buffer() {
            Ok(_) => {}
            Err(e) => return Some(Err(e.into())),
        }

        Some(Ok(batches))
    }
}
