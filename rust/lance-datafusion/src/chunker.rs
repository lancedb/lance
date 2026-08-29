// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;
use std::{collections::VecDeque, task::Context};

use arrow::compute::kernels;
use arrow_array::RecordBatch;
use datafusion::physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use datafusion_common::DataFusionError;
use futures::{Stream, StreamExt, TryStreamExt, ready};

use lance_core::Result;
use lance_core::error::DataFusionResult;

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
                // Skip empty batch
                if batch.num_rows() == 0 {
                    continue;
                }

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

struct BreakStreamState {
    max_rows: usize,
    rows_seen: usize,
    rows_remaining: usize,
    batch: Option<RecordBatch>,
}

impl BreakStreamState {
    fn next(mut self) -> Option<(Result<RecordBatch>, Self)> {
        if self.rows_remaining == 0 {
            return None;
        }
        if self.rows_remaining + self.rows_seen <= self.max_rows {
            self.rows_seen = (self.rows_seen + self.rows_remaining) % self.max_rows;
            self.rows_remaining = 0;
            let next = self.batch.take().unwrap();
            Some((Ok(next), self))
        } else {
            let rows_to_emit = self.max_rows - self.rows_seen;
            self.rows_seen = 0;
            self.rows_remaining -= rows_to_emit;
            let batch = self.batch.as_mut().unwrap();
            let next = batch.slice(0, rows_to_emit);
            *batch = batch.slice(rows_to_emit, batch.num_rows() - rows_to_emit);
            Some((Ok(next), self))
        }
    }
}

// Given a stream of record batches, and a desired break point, this will
// make sure that a new record batch is emitted every time `break_point` rows
// have passed.
//
// This method will not combine record batches in any way.  For example, if
// the input lengths are [3, 5, 8, 3, 5], and the break point is 10 then the
// output batches will be [3, 5, 2 (break inserted) 6, 3, 1 (break inserted) 4]
pub fn break_stream(
    stream: SendableRecordBatchStream,
    max_chunk_size: usize,
) -> Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>> {
    let mut rows_already_seen = 0;
    stream
        .map_ok(move |batch| {
            let state = BreakStreamState {
                rows_remaining: batch.num_rows(),
                max_rows: max_chunk_size,
                rows_seen: rows_already_seen,
                batch: Some(batch),
            };
            rows_already_seen = (state.rows_seen + state.rows_remaining) % state.max_rows;

            futures::stream::unfold(state, move |state| std::future::ready(state.next()))
                .fuse()
                .boxed()
        })
        .try_flatten()
        .boxed()
}

/// Given a stream of record batches, this will yield batches of a fixed size.
///
/// In order to avoid copying data the batches will be converted into a stream of
/// `Vec<RecordBatch>` where each item is a `Vec` of batches whose total size is
/// `chunk_size`.
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
    .fuse()
    .boxed()
}

/// Given a stream of record batches, this will yield batches of a fixed size.
///
/// This stream _will_ combine record batches and so it can be fairly expensive as it will
/// likely force a copy of incoming data.  However, it can be useful when users require
/// precise batch sizing.
pub fn chunk_concat_stream(
    stream: SendableRecordBatchStream,
    chunk_size: usize,
) -> SendableRecordBatchStream {
    let schema = stream.schema();
    let schema_copy = schema.clone();
    let chunked = chunk_stream(stream, chunk_size);
    let chunk_concat = chunked
        .and_then(move |batches| {
            std::future::ready(
                // chunk_stream is zero-copy and so it gives us pieces of batches.  However, the btree
                // index needs 1 batch-per-page and so we concatenate here.
                kernels::concat::concat_batches(&schema, batches.iter()).map_err(|e| e.into()),
            )
        })
        .map_err(DataFusionError::from)
        .boxed();
    Box::pin(RecordBatchStreamAdapter::new(schema_copy, chunk_concat))
}

/// Given a stream of record batches, this will yield batches of a fixed size.
///
/// This stream _will_ combine record batches and so it can be fairly expensive as it will
/// likely force a copy of all incoming data.  However, it can be useful when users require
/// precise batch sizing.
pub struct StrictBatchSizeStream<S> {
    inner: S,
    batch_size: usize,
    residual: Option<RecordBatch>,
}

impl<S: Stream<Item = DataFusionResult<RecordBatch>> + Unpin> StrictBatchSizeStream<S> {
    pub fn new(inner: S, batch_size: usize) -> Self {
        Self {
            inner,
            batch_size,
            residual: None,
        }
    }
}

/// Internal polling method for strict batch size enforcement.
///
/// # Use Case
/// When precise batch sizing is required (e.g., ML batch processing), this method guarantees
/// output batches exactly match batch_size until final partial batch. Maintains data integrity
/// across splits using row-aware splitting.
///
/// # Example
/// With batch_size=5 and input sequence:
/// - Fragment 1: 7 rows → splits into `[5,2]`
///   (queues 5, carries 2)
/// - Fragment 2: 4 rows → combines carried 2 + 4 = 6
///   splits into `[5,1]`
///
/// - Output batches: `[5]`, `[5]`, `[1]`
impl<S> Stream for StrictBatchSizeStream<S>
where
    S: Stream<Item = DataFusionResult<RecordBatch>> + Unpin,
{
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // Process residual first if present
            if let Some(residual) = self.residual.take() {
                if residual.num_rows() >= self.batch_size {
                    let split_at = self.batch_size;
                    let chunk = residual.slice(0, split_at);
                    let new_residual = residual.slice(split_at, residual.num_rows() - split_at);
                    self.residual = Some(new_residual);
                    return Poll::Ready(Some(Ok(chunk)));
                } else {
                    // Keep residual and proceed to get more data
                    self.residual = Some(residual);
                }
            }

            // Poll the inner stream for next batch
            match ready!(Pin::new(&mut self.inner).poll_next(cx)) {
                Some(Ok(batch)) => {
                    // Combine with residual if any
                    let current_batch = if let Some(residual) = self.residual.take() {
                        arrow::compute::concat_batches(&residual.schema(), &[residual, batch])
                            .map_err(|e| DataFusionError::External(Box::new(e)))?
                    } else {
                        batch
                    };

                    if current_batch.num_rows() >= self.batch_size {
                        let split_at = self.batch_size;
                        let chunk = current_batch.slice(0, split_at);
                        let new_residual =
                            current_batch.slice(split_at, current_batch.num_rows() - split_at);
                        if new_residual.num_rows() > 0 {
                            self.residual = Some(new_residual);
                        }
                        return Poll::Ready(Some(Ok(chunk)));
                    } else {
                        // Not enough rows, store as residual
                        self.residual = Some(current_batch);
                        continue;
                    }
                }
                Some(Err(e)) => return Poll::Ready(Some(Err(e))),
                None => {
                    return Poll::Ready(
                        self.residual
                            .take()
                            .filter(|r| r.num_rows() > 0)
                            .map(Ok::<_, DataFusionError>),
                    );
                }
            }
        }
    }
}

/// Concatenates incoming batches into chunks bounded by both row count and
/// byte size.
///
/// Batches are accumulated as zero-copy slices rather than concatenated on
/// every insert, so accumulating N micro-batches is O(N) and the copies happen
/// only when a chunk is emitted.  This keeps the memory lesson from #2438
/// (don't copy every incoming batch) while still coalescing small compaction
/// inputs.
///
/// A single input batch that exceeds the row budget is split into row-bounded
/// slices so that `max_rows_per_file` boundaries are honored.  The byte budget
/// is used only to decide when to stop accumulating and emit the buffered data;
/// an input batch that is individually larger than the byte budget is emitted
/// as-is rather than sliced, so caller-supplied batch boundaries are preserved.
pub struct ByteBoundedConcatChunker {
    inner: SendableRecordBatchStream,
    /// Schema to use for emitted batches.  This is the schema of the input
    /// stream, which may contain field-level metadata (e.g. blob thresholds)
    /// that `concat_batches` would otherwise strip from a freshly concatenated
    /// batch.
    schema: Arc<arrow_schema::Schema>,
    chunk_size: usize,
    max_bytes: usize,
    /// Buffered data that has not yet been emitted, kept as a list of
    /// zero-copy slices.  `buffered_rows`/`buffered_bytes` are the running
    /// totals so we don't rescan the list on every poll.
    buffered: Vec<RecordBatch>,
    buffered_rows: usize,
    buffered_bytes: usize,
    /// Slices of an input batch that exceeds `chunk_size` rows and still
    /// need to be emitted.  Only the row budget causes a single batch to be
    /// split; the byte budget is used only to trigger emission of buffered
    /// data.
    pending: VecDeque<RecordBatch>,
}

impl ByteBoundedConcatChunker {
    pub fn new(inner: SendableRecordBatchStream, chunk_size: usize, max_bytes: usize) -> Self {
        // Callers reject zero limits at the write boundary before constructing
        // this stream, so these are internal invariants rather than user-input
        // validation.
        debug_assert!(chunk_size > 0, "chunk_size must be greater than zero");
        debug_assert!(max_bytes > 0, "max_bytes must be greater than zero");
        let schema = inner.schema();
        Self {
            inner,
            schema,
            chunk_size,
            max_bytes,
            buffered: Vec::new(),
            buffered_rows: 0,
            buffered_bytes: 0,
            pending: VecDeque::new(),
        }
    }
}

impl Stream for ByteBoundedConcatChunker {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // Emit any pending slices of a previously split oversized batch
            // before pulling more data.
            if let Some(pending) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(pending)));
            }

            // If the buffered data already exceeds either budget, concatenate
            // and emit it.  Concatenation copies, so we only do it once per
            // emitted chunk rather than on every insert.
            if self.buffered_rows >= self.chunk_size || self.buffered_bytes >= self.max_bytes {
                if self.buffered.is_empty() {
                    return Poll::Ready(None);
                }
                let combined = self.concat_and_clear()?;
                return Poll::Ready(Some(Ok(combined)));
            }

            // Try to pull the next batch from the inner stream.
            let next = Pin::new(&mut self.inner).poll_next(cx);
            let Some(batch) = ready!(next) else {
                // Inner stream exhausted: emit whatever is buffered.
                if self.buffered.is_empty() {
                    return Poll::Ready(None);
                }
                let combined = self.concat_and_clear()?;
                return Poll::Ready(Some(Ok(combined)));
            };

            let batch = match batch {
                Ok(batch) => batch,
                Err(e) => {
                    // Preserve buffered data across an error so a later poll
                    // can still flush it if the caller retries.
                    return Poll::Ready(Some(Err(e)));
                }
            };

            if batch.num_rows() == 0 {
                continue;
            }
            let batch_rows = batch.num_rows();
            let batch_bytes = batch.get_array_memory_size();

            // A batch that alone exceeds the row budget is split into
            // row-bounded slices so that `max_rows_per_file` boundaries
            // are honored.  We do not split by bytes here: an
            // individually oversized batch is emitted as-is so that
            // caller-supplied batch boundaries are preserved.
            //
            // If we already have buffered rows, use the first part of
            // this batch to fill the remaining row capacity, emit that
            // full chunk, then queue the rest as slices.  This avoids
            // wasting the file-row budget and preserves every row.
            if batch_rows > self.chunk_size {
                if self.buffered_rows >= self.chunk_size {
                    // Buffered data already filled a chunk; emit it and
                    // slice the new batch on the next poll.
                    let combined = self.concat_and_clear()?;
                    self.push_slices(&batch, 0, batch_rows);
                    return Poll::Ready(Some(Ok(combined)));
                }

                if self.buffered.is_empty() {
                    self.push_slices(&batch, 0, batch_rows);
                    continue;
                }

                let needed = self.chunk_size - self.buffered_rows;
                let head = batch.slice(0, needed);
                self.buffered_bytes += head.get_array_memory_size();
                self.buffered.push(head);
                self.buffered_rows = self.chunk_size;
                let combined = self.concat_and_clear()?;
                self.push_slices(&batch, needed, batch_rows);
                return Poll::Ready(Some(Ok(combined)));
            }

            // A batch that would push us over either budget is emitted
            // with whatever we had, and this one is deferred to the next
            // poll.  When the row budget is the limiting factor, use the
            // leading rows of the new batch to fill the chunk exactly, emit
            // it, and carry over the remainder; this avoids wasting the
            // file-row budget.
            if !self.buffered.is_empty()
                && (self.buffered_rows + batch_rows > self.chunk_size
                    || self.buffered_bytes + batch_bytes > self.max_bytes)
            {
                if self.buffered_rows < self.chunk_size
                    && self.buffered_rows + batch_rows > self.chunk_size
                {
                    let needed = self.chunk_size - self.buffered_rows;
                    let head = batch.slice(0, needed);
                    self.buffered_bytes += head.get_array_memory_size();
                    self.buffered.push(head);
                    self.buffered_rows = self.chunk_size;
                    let combined = self.concat_and_clear()?;
                    let tail = batch.slice(needed, batch_rows - needed);
                    self.buffered_bytes = tail.get_array_memory_size();
                    self.buffered.push(tail);
                    self.buffered_rows = batch_rows - needed;
                    return Poll::Ready(Some(Ok(combined)));
                }

                let combined = self.concat_and_clear()?;
                self.buffered.push(batch);
                self.buffered_rows = batch_rows;
                self.buffered_bytes = batch_bytes;
                return Poll::Ready(Some(Ok(combined)));
            }
            self.buffered.push(batch);
            self.buffered_rows += batch_rows;
            self.buffered_bytes += batch_bytes;
        }
    }
}

impl ByteBoundedConcatChunker {
    /// Concatenate all buffered slices into one batch and reset the buffer.
    fn concat_and_clear(&mut self) -> DataFusionResult<RecordBatch> {
        let combined = arrow::compute::concat_batches(&self.schema, &self.buffered)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        self.buffered.clear();
        self.buffered_rows = 0;
        self.buffered_bytes = 0;
        Ok(combined)
    }

    /// Push zero-copy slices of `batch[start_rows..total_rows)` into `pending`.
    ///
    /// Full `chunk_size` slices go into `pending` (emitted on the next poll).
    /// A trailing slice shorter than `chunk_size` is *not* emitted as a
    /// complete chunk: it is kept in `buffered` so later input can fill the
    /// remaining `chunk_size` rows, otherwise the `max_rows_per_file` boundary
    /// is exceeded (the writer only rotates after writing a whole emitted
    /// chunk).
    fn push_slices(&mut self, batch: &RecordBatch, start_rows: usize, total_rows: usize) {
        let mut offset = start_rows;
        while offset < total_rows {
            let rows = (total_rows - offset).min(self.chunk_size).max(1);
            let slice = batch.slice(offset, rows);
            let slice_bytes = slice.get_array_memory_size();
            if rows < self.chunk_size {
                self.buffered.push(slice);
                self.buffered_rows += rows;
                self.buffered_bytes += slice_bytes;
            } else {
                self.pending.push_back(slice);
            }
            offset += rows;
        }
    }
}

/// Concatenate batches into row-and-byte-bounded chunks.
///
/// Output batches have at most `chunk_size` rows.  The byte budget is used to
/// decide when to stop accumulating buffered batches and emit them; an input
/// batch that is individually larger than `max_bytes` is emitted as-is so that
/// caller-supplied batch boundaries are preserved.
pub fn chunk_concat_stream_with_bytes(
    stream: SendableRecordBatchStream,
    chunk_size: usize,
    max_bytes: usize,
) -> SendableRecordBatchStream {
    let schema = stream.schema();
    let chunker = ByteBoundedConcatChunker::new(stream, chunk_size, max_bytes);
    Box::pin(RecordBatchStreamAdapter::new(schema, chunker))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{Int32Type, Int64Type};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::{StreamExt, TryStreamExt};
    use lance_datagen::{BatchCount, RowCount, array};

    use crate::datagen::DatafusionDatagenExt;

    #[tokio::test]
    async fn test_chunkers() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("", arrow::datatypes::DataType::Int32, false),
        ]));

        let make_batch = |num_rows: u32| {
            lance_datagen::gen_batch()
                .anon_col(lance_datagen::array::step::<Int32Type>())
                .into_batch_rows(RowCount::from(num_rows as u64))
                .unwrap()
        };

        let batches = vec![make_batch(10), make_batch(5), make_batch(13), make_batch(0)];

        let make_stream = || {
            let stream = futures::stream::iter(
                batches
                    .clone()
                    .into_iter()
                    .map(datafusion_common::Result::Ok),
            )
            .boxed();
            Box::pin(RecordBatchStreamAdapter::new(schema.clone(), stream))
        };

        let chunked = super::chunk_stream(make_stream(), 10)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(chunked.len(), 3);
        assert_eq!(chunked[0].len(), 1);
        assert_eq!(chunked[0][0].num_rows(), 10);
        assert_eq!(chunked[1].len(), 2);
        assert_eq!(chunked[1][0].num_rows(), 5);
        assert_eq!(chunked[1][1].num_rows(), 5);
        assert_eq!(chunked[2].len(), 1);
        assert_eq!(chunked[2][0].num_rows(), 8);

        let chunked = super::chunk_concat_stream(make_stream(), 10)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(chunked.len(), 3);
        assert_eq!(chunked[0].num_rows(), 10);
        assert_eq!(chunked[1].num_rows(), 10);
        assert_eq!(chunked[2].num_rows(), 8);

        let chunked = super::break_stream(make_stream(), 10)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(chunked.len(), 4);
        assert_eq!(chunked[0].num_rows(), 10);
        assert_eq!(chunked[1].num_rows(), 5);
        assert_eq!(chunked[2].num_rows(), 5);
        assert_eq!(chunked[3].num_rows(), 8);
    }

    #[tokio::test]
    async fn test_chunk_concat_stream_with_bytes() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("", arrow::datatypes::DataType::Int32, false),
        ]));

        // 4 batches of 5 rows = 20 rows total, ~80 bytes each.
        let batches = (0..4)
            .map(|_| {
                lance_datagen::gen_batch()
                    .anon_col(lance_datagen::array::step::<Int32Type>())
                    .into_batch_rows(RowCount::from(5))
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let make_stream = || {
            let stream = futures::stream::iter(
                batches
                    .clone()
                    .into_iter()
                    .map(datafusion_common::Result::Ok),
            )
            .boxed();
            Box::pin(RecordBatchStreamAdapter::new(schema.clone(), stream))
        };

        // Row budget of 10 and a generous byte budget -> two 10-row chunks.
        let chunked = super::chunk_concat_stream_with_bytes(make_stream(), 10, 1024 * 1024)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(chunked.len(), 2);
        assert_eq!(chunked[0].num_rows(), 10);
        assert_eq!(chunked[1].num_rows(), 10);

        // A byte budget large enough for several small batches, so the row
        // budget is the limiting factor and batches are concatenated into
        // 10-row chunks.
        let chunked = super::chunk_concat_stream_with_bytes(make_stream(), 10, 1024)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(chunked.len(), 2);
        assert_eq!(chunked[0].num_rows(), 10);
        assert_eq!(chunked[1].num_rows(), 10);
    }

    #[tokio::test]
    async fn test_chunk_concat_stream_with_bytes_splits_oversized_batch() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("", arrow::datatypes::DataType::Int32, false),
        ]));

        // A single batch of 25 rows, larger than the row budget of 10.
        let batch = lance_datagen::gen_batch()
            .anon_col(lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(25))
            .unwrap();
        let stream = futures::stream::iter(vec![datafusion_common::Result::Ok(batch)]).boxed();
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        // A generous byte budget so only the row budget drives splitting.
        let chunked = super::chunk_concat_stream_with_bytes(stream, 10, 1024 * 1024)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        // The 25-row batch is split into 10 + 10 + 5.
        assert_eq!(chunked.len(), 3);
        assert_eq!(chunked[0].num_rows(), 10);
        assert_eq!(chunked[1].num_rows(), 10);
        assert_eq!(chunked[2].num_rows(), 5);
    }

    #[tokio::test]
    async fn test_chunk_concat_stream_with_bytes_fills_remaining_row_capacity() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("", arrow::datatypes::DataType::Int32, false),
        ]));

        let make_batch = |num_rows: u32| {
            lance_datagen::gen_batch()
                .anon_col(lance_datagen::array::step::<Int32Type>())
                .into_batch_rows(RowCount::from(num_rows as u64))
                .unwrap()
        };

        // A 3-row batch buffered first, then a 25-row batch.  The chunker should
        // fill the remaining 7 rows of the first chunk before slicing the rest,
        // giving 10 + 10 + 8 rather than 3 + 10 + 10 + 5.
        let batches = vec![make_batch(3), make_batch(25)];
        let stream = futures::stream::iter(
            batches
                .into_iter()
                .map(datafusion_common::Result::Ok)
                .collect::<Vec<_>>(),
        )
        .boxed();
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        let chunked = super::chunk_concat_stream_with_bytes(stream, 10, 1024 * 1024)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(chunked.len(), 3);
        assert_eq!(chunked[0].num_rows(), 10);
        assert_eq!(chunked[1].num_rows(), 10);
        assert_eq!(chunked[2].num_rows(), 8);
    }

    // Regression for the file-row boundary: a short tail from a split
    // oversized batch must be kept in the buffer so later input fills the
    // remaining chunk_size rows instead of being emitted as a partial chunk.
    // With chunk_size=10 and input lengths [3, 25, 5] the output must be
    // [10, 10, 10, 3], not [10, 10, 13].
    #[tokio::test]
    async fn test_chunk_concat_stream_with_bytes_keeps_short_tail_for_later_input() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("", arrow::datatypes::DataType::Int32, false),
        ]));

        let make_batch = |num_rows: u32| {
            lance_datagen::gen_batch()
                .anon_col(lance_datagen::array::step::<Int32Type>())
                .into_batch_rows(RowCount::from(num_rows as u64))
                .unwrap()
        };

        let batches = vec![make_batch(3), make_batch(25), make_batch(5)];
        let stream = futures::stream::iter(
            batches
                .into_iter()
                .map(datafusion_common::Result::Ok)
                .collect::<Vec<_>>(),
        )
        .boxed();
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        let chunked = super::chunk_concat_stream_with_bytes(stream, 10, 1024 * 1024)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(chunked.len(), 4);
        assert_eq!(chunked[0].num_rows(), 10);
        assert_eq!(chunked[1].num_rows(), 10);
        assert_eq!(chunked[2].num_rows(), 10);
        assert_eq!(chunked[3].num_rows(), 3);
    }

    #[tokio::test]
    async fn test_chunk_concat_stream_with_bytes_emits_oversized_batch_unsplit() {
        use arrow::array::{ArrayRef, LargeBinaryArray};

        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("data", arrow::datatypes::DataType::LargeBinary, false),
        ]));

        // 5 rows, each ~200 KiB raw.  The byte budget is 100 KiB, but a single
        // input batch is emitted as-is rather than sliced by bytes.
        let values: Vec<Vec<u8>> = (0..5).map(|i| vec![i as u8; 200 * 1024]).collect();
        let array: ArrayRef = Arc::new(LargeBinaryArray::from_iter_values(
            values.iter().map(|v| v.as_slice()),
        ));
        let batch = arrow::record_batch::RecordBatch::try_new(schema.clone(), vec![array]).unwrap();

        let stream = futures::stream::iter(vec![datafusion_common::Result::Ok(batch)]).boxed();
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        let chunked = super::chunk_concat_stream_with_bytes(stream, 10, 100 * 1024)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(chunked.len(), 1);
        assert_eq!(chunked[0].num_rows(), 5);
    }

    #[tokio::test]
    async fn test_chunk_concat_stream_with_bytes_honors_byte_budget() {
        use arrow::array::{ArrayRef, LargeBinaryArray};

        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("data", arrow::datatypes::DataType::LargeBinary, false),
        ]));

        // Four 2-row batches, each ~50 KiB raw.  With a 120 KiB byte budget the
        // chunker accumulates two batches (~100 KiB) but cannot fit a third, so
        // it emits and starts a new chunk.  Row budget of 10 is looser and does
        // not drive splitting.
        let batches: Vec<arrow::record_batch::RecordBatch> = (0..4)
            .map(|_| {
                let values: Vec<Vec<u8>> = (0..2).map(|i| vec![i as u8; 25 * 1024]).collect();
                let array: ArrayRef = Arc::new(LargeBinaryArray::from_iter_values(
                    values.iter().map(|v| v.as_slice()),
                ));
                arrow::record_batch::RecordBatch::try_new(schema.clone(), vec![array]).unwrap()
            })
            .collect();

        let stream = futures::stream::iter(
            batches
                .into_iter()
                .map(datafusion_common::Result::Ok)
                .collect::<Vec<_>>(),
        )
        .boxed();
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        let chunked = super::chunk_concat_stream_with_bytes(stream, 10, 120 * 1024)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(chunked.len(), 2);
        assert_eq!(chunked[0].num_rows(), 4);
        assert_eq!(chunked[1].num_rows(), 4);
    }

    #[tokio::test]
    async fn test_chunk_concat_stream_with_bytes_fills_remaining_row_capacity_for_small_batches() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("a", arrow::datatypes::DataType::Int64, false),
        ]));

        // Three 400-row batches with a 1000-row budget should yield 1000 + 200,
        // not 800 + 400, so compaction does not waste the row budget.
        let batches: Vec<arrow::record_batch::RecordBatch> = (0..3)
            .map(|i| {
                let values: Vec<i64> = (0..400).map(|j| (i * 400 + j) as i64).collect();
                let array: arrow::array::ArrayRef =
                    Arc::new(arrow::array::Int64Array::from(values));
                arrow::record_batch::RecordBatch::try_new(schema.clone(), vec![array]).unwrap()
            })
            .collect();

        let stream = futures::stream::iter(
            batches
                .into_iter()
                .map(datafusion_common::Result::Ok)
                .collect::<Vec<_>>(),
        )
        .boxed();
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        let chunked = super::chunk_concat_stream_with_bytes(stream, 1000, 64 * 1024 * 1024)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(chunked.len(), 2);
        assert_eq!(chunked[0].num_rows(), 1000);
        assert_eq!(chunked[1].num_rows(), 200);
    }

    #[tokio::test]
    async fn test_strict_batch_size_stream() {
        let batches = lance_datagen::gen_batch()
            .anon_col(array::step::<Int32Type>())
            .anon_col(array::step::<Int64Type>())
            .into_df_stream(RowCount::from(7), BatchCount::from(10));

        let stream = super::StrictBatchSizeStream::new(batches, 10);

        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(batches.len(), 7);

        for batch in batches {
            assert_eq!(batch.num_rows(), 10);
        }
    }
}
