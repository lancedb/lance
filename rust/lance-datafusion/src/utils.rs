// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{RecordBatchIterator, RecordBatchReader};
use datafusion::physical_plan::{stream::RecordBatchStreamAdapter, SendableRecordBatchStream};
use datafusion_common::DataFusionError;
use futures::{stream, Stream, StreamExt, TryFutureExt, TryStreamExt};
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use tokio::task::spawn_blocking;

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

/// Infer the Lance schema from the first batch.
///
/// This will peek the first batch to get the dictionaries for dictionary columns.
///
/// NOTE: this does not validate the schema. For example, for appends the schema
/// should be checked to make sure it matches the existing dataset schema before
/// writing.
pub async fn peek_reader_schema(
    batches: Box<dyn RecordBatchReader + Send>,
) -> Result<(Box<dyn RecordBatchReader + Send>, Schema)> {
    let arrow_schema = batches.schema();
    let (peekable, schema) = spawn_blocking(move || {
        let mut schema: Schema = Schema::try_from(batches.schema().as_ref())?;
        let mut peekable = batches.peekable();
        if let Some(batch) = peekable.peek() {
            if let Ok(b) = batch {
                schema.set_dictionary(b)?;
            } else {
                return Err(Error::from(batch.as_ref().unwrap_err()));
            }
        }
        Ok((peekable, schema))
    })
    .await
    .unwrap()?;
    schema.validate()?;
    let reader = RecordBatchIterator::new(peekable, arrow_schema);
    Ok((
        Box::new(reader) as Box<dyn RecordBatchReader + Send>,
        schema,
    ))
}

/// Convert reader to a stream.
///
/// The reader will be called in a background thread.
pub fn reader_to_stream(batches: Box<dyn RecordBatchReader + Send>) -> SendableRecordBatchStream {
    let arrow_schema = batches.schema();
    let stream = RecordBatchStreamAdapter::new(
        arrow_schema,
        background_iterator(batches).map_err(DataFusionError::from),
    );
    Box::pin(stream)
}
