// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::collections::BinaryHeap;

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    execution::RecordBatchStream,
    physical_plan::{
        metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder, MetricValue, MetricsSet},
        stream::RecordBatchStreamAdapter,
        SendableRecordBatchStream,
    },
};
use datafusion_common::DataFusionError;
use futures::stream::BoxStream;
use futures::{stream, Stream, StreamExt, TryFutureExt, TryStreamExt};
use lance_core::datatypes::Schema;
use lance_core::Result;
use tokio::task::{spawn, spawn_blocking};

fn background_iterator<I: Iterator + Send + 'static>(iter: I) -> impl Stream<Item = I::Item>
where
    I::Item: Send,
{
    stream::unfold(iter, |mut iter| {
        spawn_blocking(|| iter.next().map(|val| (val, iter)))
            .unwrap_or_else(|err| panic!("{}", err))
    })
    .fuse()
}

/// A trait for [BatchRecord] iterators, readers and streams
/// that can be converted to a concrete stream type [SendableRecordBatchStream].
///
/// This also cam read the schema from the first batch
/// and then update the schema to reflect the dictionary columns.
#[async_trait]
pub trait StreamingWriteSource: Send {
    /// Infer the Lance schema from the first batch stream.
    ///
    /// This will peek the first batch to get the dictionaries for dictionary columns.
    ///
    /// NOTE: this does not validate the schema. For example, for appends the schema
    /// should be checked to make sure it matches the existing dataset schema before
    /// writing.
    async fn into_stream_and_schema(self) -> Result<(SendableRecordBatchStream, Schema)>
    where
        Self: Sized,
    {
        let mut stream = self.into_stream();
        let (stream, arrow_schema, schema) = spawn(async move {
            let arrow_schema = stream.schema();
            let mut schema: Schema = Schema::try_from(arrow_schema.as_ref())?;
            let first_batch = stream.try_next().await?;
            if let Some(batch) = &first_batch {
                schema.set_dictionary(batch)?;
            }
            let stream = stream::iter(first_batch.map(Ok)).chain(stream);
            Result::Ok((stream, arrow_schema, schema))
        })
        .await
        .unwrap()?;
        schema.validate()?;
        let adapter = RecordBatchStreamAdapter::new(arrow_schema, stream);
        Ok((Box::pin(adapter), schema))
    }

    /// Returns the arrow schema.
    fn arrow_schema(&self) -> SchemaRef;

    /// Convert to a stream.
    ///
    /// The conversion will be conducted in a background thread.
    fn into_stream(self) -> SendableRecordBatchStream;
}

impl StreamingWriteSource for ArrowArrayStreamReader {
    #[inline]
    fn arrow_schema(&self) -> SchemaRef {
        RecordBatchReader::schema(self)
    }

    #[inline]
    fn into_stream(self) -> SendableRecordBatchStream {
        reader_to_stream(Box::new(self))
    }
}

impl<I> StreamingWriteSource for RecordBatchIterator<I>
where
    Self: Send,
    I: IntoIterator<Item = ::core::result::Result<RecordBatch, ArrowError>> + Send + 'static,
{
    #[inline]
    fn arrow_schema(&self) -> SchemaRef {
        RecordBatchReader::schema(self)
    }

    #[inline]
    fn into_stream(self) -> SendableRecordBatchStream {
        reader_to_stream(Box::new(self))
    }
}

impl<T> StreamingWriteSource for Box<T>
where
    T: StreamingWriteSource,
{
    #[inline]
    fn arrow_schema(&self) -> SchemaRef {
        T::arrow_schema(&**self)
    }

    #[inline]
    fn into_stream(self) -> SendableRecordBatchStream {
        T::into_stream(*self)
    }
}

impl StreamingWriteSource for Box<dyn RecordBatchReader + Send> {
    #[inline]
    fn arrow_schema(&self) -> SchemaRef {
        RecordBatchReader::schema(self)
    }

    #[inline]
    fn into_stream(self) -> SendableRecordBatchStream {
        reader_to_stream(self)
    }
}

impl StreamingWriteSource for SendableRecordBatchStream {
    #[inline]
    fn arrow_schema(&self) -> SchemaRef {
        RecordBatchStream::schema(&**self)
    }

    #[inline]
    fn into_stream(self) -> SendableRecordBatchStream {
        self
    }
}

/// Convert reader to a stream.
///
/// The reader will be called in a background thread.
pub fn reader_to_stream(batches: Box<dyn RecordBatchReader + Send>) -> SendableRecordBatchStream {
    let arrow_schema = batches.arrow_schema();
    let stream = RecordBatchStreamAdapter::new(
        arrow_schema,
        background_iterator(batches).map_err(DataFusionError::from),
    );
    Box::pin(stream)
}

pub trait MetricsExt {
    fn find_count(&self, name: &str) -> Option<Count>;
}

impl MetricsExt for MetricsSet {
    fn find_count(&self, metric_name: &str) -> Option<Count> {
        self.iter().find_map(|m| match m.value() {
            MetricValue::Count { name, count } => {
                if name == metric_name {
                    Some(count.clone())
                } else {
                    None
                }
            }
            _ => None,
        })
    }
}

pub trait ExecutionPlanMetricsSetExt {
    fn new_count(&self, name: &'static str, partition: usize) -> Count;
}

impl ExecutionPlanMetricsSetExt for ExecutionPlanMetricsSet {
    fn new_count(&self, name: &'static str, partition: usize) -> Count {
        let count = Count::new();
        MetricBuilder::new(self)
            .with_partition(partition)
            .build(MetricValue::Count {
                name: Cow::Borrowed(name),
                count: count.clone(),
            });
        count
    }
}

#[derive(Debug, Clone)]
struct Intermediate<K: Ord + Clone, V> {
    idx: usize,
    key: K,
    value: V,
}

impl<K: Ord + Clone, V> Intermediate<K, V> {
    fn new(idx: usize, key: K, value: V) -> Self {
        Self { idx, key, value }
    }
}

impl<K: Ord + Clone, V> PartialEq for Intermediate<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<K: Ord + Clone, V> Eq for Intermediate<K, V> {}

impl<K: Ord + Clone, V> PartialOrd for Intermediate<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord + Clone, V> Ord for Intermediate<K, V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

type KeyStream<K, Item> = BoxStream<'static, Result<(K, Item)>>;

/// A stream that merges multiple streams in order of key.
/// All the stream must have the same schema, and the key must be produced
/// in the same order.
pub struct BatchMergeStream<K: Ord + Clone, Item> {
    streams: Vec<KeyStream<K, Item>>,
    heap: BinaryHeap<std::cmp::Reverse<Intermediate<K, Item>>>,
}

impl<K: Ord + Clone, Item> BatchMergeStream<K, Item> {
    pub async fn try_new(mut streams: Vec<KeyStream<K, Item>>) -> Result<Self> {
        // get the first batch from each stream
        let mut heap = BinaryHeap::with_capacity(streams.len());
        let mut futures = Vec::with_capacity(streams.len());
        for stream in streams.iter_mut() {
            let fut = stream.try_next();
            futures.push(fut);
        }
        let results = futures::future::try_join_all(futures).await?;
        for (i, result) in results.into_iter().enumerate() {
            if let Some((key, batch)) = result {
                heap.push(std::cmp::Reverse(Intermediate::new(i, key, batch)));
            }
        }

        Ok(Self { streams, heap })
    }
}

impl<K: Ord + Clone + Unpin, Item: Unpin> Stream for BatchMergeStream<K, Item> {
    type Item = Result<(K, Item)>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.heap.is_empty() {
            return std::task::Poll::Ready(None);
        }

        let intermediate = this.heap.pop().unwrap().0;

        // consume the stream that produced the value,
        // and push the next value to the heap
        let stream = &mut this.streams[intermediate.idx];
        match stream.poll_next_unpin(cx) {
            std::task::Poll::Ready(Some(Ok((key, value)))) => {
                this.heap.push(std::cmp::Reverse(Intermediate::new(
                    intermediate.idx,
                    key,
                    value,
                )));
                std::task::Poll::Ready(Some(Ok((intermediate.key, intermediate.value))))
            }
            std::task::Poll::Ready(Some(Err(err))) => std::task::Poll::Ready(Some(Err(err))),
            std::task::Poll::Ready(None) => {
                // stream is done, we can just return the value
                std::task::Poll::Ready(Some(Ok((intermediate.key, intermediate.value))))
            }
            std::task::Poll::Pending => {
                // stream is not ready yet
                if !this.heap.is_empty() && this.heap.peek().unwrap().0.key == intermediate.key {
                    // if the next value is the same key, we can just return it
                    std::task::Poll::Ready(Some(Ok((intermediate.key, intermediate.value))))
                } else {
                    // otherwise, we need to wait for the stream to be ready
                    this.heap.push(std::cmp::Reverse(intermediate));
                    std::task::Poll::Pending
                }
            }
        }
    }
}

// Common metrics
pub const IOPS_METRIC: &str = "iops";
pub const REQUESTS_METRIC: &str = "requests";
pub const BYTES_READ_METRIC: &str = "bytes_read";
pub const INDICES_LOADED_METRIC: &str = "indices_loaded";
pub const PARTS_LOADED_METRIC: &str = "parts_loaded";
pub const INDEX_COMPARISONS_METRIC: &str = "index_comparisons";
