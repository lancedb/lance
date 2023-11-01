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

use std::collections::VecDeque;
use std::pin::Pin;

use arrow_array::RecordBatch;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{Stream, StreamExt};

use crate::Result;

/// Wraps a [`SendableRecordBatchStream`] into a stream of RecordBatch chunks of
/// a given size.  This slices but does not copy any buffers.
struct BatchReaderChunker {
    /// The inner stream
    inner: SendableRecordBatchStream,
    /// The batches that have been read from the inner stream but not yet fully yielded
    buffered: VecDeque<RecordBatch>,
    /// The number of rows to yield in each chunk
    output_size: usize,
    /// The position within the first batch in the buffer to start yielding from
    i: usize,
}

impl BatchReaderChunker {
    fn new(inner: SendableRecordBatchStream, output_size: usize) -> Self {
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

    async fn next(&mut self) -> Option<Result<Vec<RecordBatch>>> {
        match self.fill_buffer().await {
            Ok(_) => {}
            Err(e) => return Some(Err(e)),
        };

        let mut batches = Vec::new();

        let mut rows_collected = 0;

        while rows_collected < self.output_size {
            if let Some(batch) = self.buffered.pop_front() {
                let rows_remaining_in_batch = batch.num_rows() - self.i;
                let rows_to_take =
                    std::cmp::min(rows_remaining_in_batch, self.output_size - rows_collected);

                if rows_to_take == rows_remaining_in_batch {
                    // We're taking the whole batch, so we can just move it
                    let batch = if self.i == 0 {
                        batch
                    } else {
                        // We are taking the remainder of the batch, so we need to slice it
                        batch.slice(self.i, rows_to_take)
                    };
                    batches.push(batch);
                    self.i = 0;
                } else {
                    // We're taking a slice of the batch, so we need to copy it
                    batches.push(batch.slice(self.i, rows_to_take));
                    // And then we need to push the remainder back onto the front of the queue
                    self.i += rows_to_take;
                    self.buffered.push_front(batch);
                }

                rows_collected += rows_to_take;
            } else {
                break;
            }
        }

        if batches.is_empty() {
            None
        } else {
            Some(Ok(batches))
        }
    }
}

pub fn chunk_stream(
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
