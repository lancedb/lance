// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::borrow::Cow;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use arrow_array::{RecordBatch, UInt64Array};
use datafusion::error::Result as DFResult;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::StreamExt;
use lance_table::rowids::{RowIdIndex, RowIdSequence};
use roaring::RoaringTreemap;

use crate::dataset::ROW_ID;
use crate::Result;

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
/// capture the row ids. At completion of the stream, the captured row ids can
/// be recieved from the returned receiver.
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
