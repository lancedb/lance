// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    execution::RecordBatchStream,
    physical_plan::{stream::RecordBatchStreamAdapter, SendableRecordBatchStream},
};
use datafusion_common::DataFusionError;
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
