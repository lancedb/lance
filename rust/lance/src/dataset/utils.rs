// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Result;
use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::Schema as ArrowSchema;
use datafusion::error::Result as DFResult;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::StreamExt;
use lance_arrow::json::{
    arrow_json_to_lance_json, convert_json_columns, convert_lance_json_to_arrow,
    is_arrow_json_field, is_json_field,
};
use lance_core::ROW_ID;
use lance_table::rowids::{RowIdIndex, RowIdSequence};
use roaring::RoaringTreemap;
use std::borrow::Cow;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

fn extract_row_ids(
    row_ids: &mut CapturedRowIds,
    batch: RecordBatch,
    row_id_idx: usize,
    non_row_id_projection: &[usize],
) -> DFResult<RecordBatch> {
    let row_ids_arr = batch.column(row_id_idx);
    let row_ids_itr = row_ids_arr
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap_or_else(|| {
            panic!(
                "Row ids had an unexpected type: {}",
                row_ids_arr.data_type()
            )
        })
        .values();
    row_ids.capture(row_ids_itr)?;
    Ok(batch.project(non_row_id_projection)?)
}

/// Given a stream that includes a row id column, return a stream that will
/// capture the row id. At completion of the stream, the captured row ids can
/// be received from the returned receiver.
pub fn make_rowid_capture_stream(
    mut target: SendableRecordBatchStream,
    stable_row_ids: bool,
) -> Result<(SendableRecordBatchStream, Receiver<CapturedRowIds>)> {
    let mut row_ids = CapturedRowIds::new(stable_row_ids);

    let (tx, rx) = std::sync::mpsc::channel();

    let schema = target.schema();
    let (row_id_idx, _) = schema
        .column_with_name(ROW_ID)
        .expect("Received a batch without row ids");
    let non_row_ids_cols = (0..schema.fields.len())
        .filter(|col| *col != row_id_idx)
        .collect::<Vec<_>>();
    let output_schema = Arc::new(schema.project(&non_row_ids_cols)?);

    let stream = futures::stream::poll_fn(move |cx| match target.poll_next_unpin(cx) {
        std::task::Poll::Ready(Some(Ok(batch))) => {
            let res = extract_row_ids(&mut row_ids, batch, row_id_idx, &non_row_ids_cols);
            std::task::Poll::Ready(Some(res))
        }
        std::task::Poll::Ready(Some(Err(err))) => std::task::Poll::Ready(Some(Err(err))),
        std::task::Poll::Ready(None) => {
            let row_ids_out = std::mem::take(&mut row_ids);
            tx.send(row_ids_out).unwrap();
            std::task::Poll::Ready(None)
        }
        std::task::Poll::Pending => std::task::Poll::Pending,
    });

    let stream = RecordBatchStreamAdapter::new(output_schema, stream);

    Ok((Box::pin(stream), rx))
}

#[derive(Debug)]
pub enum CapturedRowIds {
    AddressStyle(RoaringTreemap),
    SequenceStyle(RowIdSequence),
}

impl CapturedRowIds {
    pub fn new(stable_row_ids: bool) -> Self {
        if stable_row_ids {
            Self::SequenceStyle(RowIdSequence::new())
        } else {
            Self::AddressStyle(RoaringTreemap::new())
        }
    }

    pub fn capture(&mut self, row_ids: &[u64]) -> DFResult<()> {
        match self {
            Self::AddressStyle(ids) => {
                // Assume they are sorted
                ids.append(row_ids.iter().cloned())
                    .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            }
            Self::SequenceStyle(sequence) => {
                sequence.extend(row_ids.into());
            }
        }
        Ok(())
    }

    pub fn row_id_sequence(&self) -> Option<&RowIdSequence> {
        match self {
            Self::SequenceStyle(sequence) => Some(sequence),
            _ => None,
        }
    }

    pub fn row_addrs(&self, index: Option<&RowIdIndex>) -> Cow<'_, RoaringTreemap> {
        match self {
            Self::AddressStyle(addrs) => Cow::Borrowed(addrs),
            Self::SequenceStyle(sequence) => {
                let mut treemap = RoaringTreemap::new();
                let Some(index) = index else {
                    panic!("RowIdIndex required for sequence style row ids")
                };
                for row_id in sequence.iter() {
                    treemap.insert(index.get(row_id).expect("row id missing from index").into());
                }
                Cow::Owned(treemap)
            }
        }
    }
}

impl Default for CapturedRowIds {
    fn default() -> Self {
        Self::AddressStyle(RoaringTreemap::new())
    }
}

/// Wrap a stream to convert arrow.json to lance.json for writing
///
// FIXME: this is bad, really bad, we need to find a way to remove this.
pub fn wrap_json_stream_for_writing(
    stream: SendableRecordBatchStream,
) -> SendableRecordBatchStream {
    // Check if any fields need conversion
    let needs_conversion = stream
        .schema()
        .fields()
        .iter()
        .any(|f| is_arrow_json_field(f));

    if !needs_conversion {
        return stream;
    }

    // Convert the schema
    let arrow_schema = stream.schema();
    let mut new_fields = Vec::with_capacity(arrow_schema.fields().len());
    for field in arrow_schema.fields() {
        if is_arrow_json_field(field) {
            new_fields.push(Arc::new(arrow_json_to_lance_json(field)));
        } else {
            new_fields.push(Arc::clone(field));
        }
    }
    let converted_schema = Arc::new(ArrowSchema::new_with_metadata(
        new_fields,
        arrow_schema.metadata().clone(),
    ));

    // Convert the stream
    let converted_stream = stream.map(move |batch_result| {
        batch_result.and_then(|batch| {
            convert_json_columns(&batch)
                .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
        })
    });

    Box::pin(RecordBatchStreamAdapter::new(
        converted_schema,
        converted_stream,
    ))
}

/// Wrap a stream to convert lance.json (JSONB) back to arrow.json (strings) for reading
///
// FIXME: this is bad, really bad, we need to find a way to remove this.
pub fn wrap_json_stream_for_reading(
    stream: SendableRecordBatchStream,
) -> SendableRecordBatchStream {
    use lance_arrow::json::ARROW_JSON_EXT_NAME;
    use lance_arrow::ARROW_EXT_NAME_KEY;

    // Check if any fields need conversion
    let needs_conversion = stream.schema().fields().iter().any(|f| is_json_field(f));

    if !needs_conversion {
        return stream;
    }

    // Convert the schema
    let arrow_schema = stream.schema();
    let mut new_fields = Vec::with_capacity(arrow_schema.fields().len());
    for field in arrow_schema.fields() {
        if is_json_field(field) {
            // Convert lance.json (LargeBinary) to arrow.json (Utf8)
            let mut new_field = arrow_schema::Field::new(
                field.name(),
                arrow_schema::DataType::Utf8,
                field.is_nullable(),
            );
            let mut metadata = field.metadata().clone();
            metadata.insert(
                ARROW_EXT_NAME_KEY.to_string(),
                ARROW_JSON_EXT_NAME.to_string(),
            );
            new_field.set_metadata(metadata);
            new_fields.push(new_field);
        } else {
            new_fields.push(field.as_ref().clone());
        }
    }
    let converted_schema = Arc::new(ArrowSchema::new_with_metadata(
        new_fields,
        arrow_schema.metadata().clone(),
    ));

    // Convert the stream
    let converted_stream = stream.map(move |batch_result| {
        batch_result.and_then(|batch| {
            convert_lance_json_to_arrow(&batch).map_err(|e| {
                datafusion::error::DataFusionError::ArrowError(
                    Box::new(arrow_schema::ArrowError::InvalidArgumentError(
                        e.to_string(),
                    )),
                    None,
                )
            })
        })
    });

    Box::pin(RecordBatchStreamAdapter::new(
        converted_schema,
        converted_stream,
    ))
}
