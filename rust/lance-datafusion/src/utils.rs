use arrow_array::RecordBatchReader;
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

/// Convert reader to a stream and a schema.
///
/// Will peek the first batch to get the dictionaries for dictionary columns.
///
/// NOTE: this does not validate the schema. For example, for appends the schema
/// should be checked to make sure it matches the existing dataset schema before
/// writing.
pub async fn reader_to_stream(
    batches: Box<dyn RecordBatchReader + Send>,
) -> Result<(SendableRecordBatchStream, Schema)> {
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

    let stream = RecordBatchStreamAdapter::new(
        arrow_schema,
        background_iterator(peekable).map_err(DataFusionError::from),
    );
    let stream = Box::pin(stream) as SendableRecordBatchStream;

    Ok((stream, schema))
}
