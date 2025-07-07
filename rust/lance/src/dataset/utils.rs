// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::sync::{Arc, RwLock};

use arrow_array::{RecordBatch, UInt64Array};
use datafusion::error::Result as DFResult;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::StreamExt;
use roaring::RoaringTreemap;

use crate::dataset::ROW_ID;
use crate::Result;
use lance_core::ROW_ADDR;

fn extract_row_addrs(
    row_addrs: &mut RoaringTreemap,
    batch: DFResult<RecordBatch>,
) -> DFResult<RecordBatch> {
    let batch = batch?;

    let (row_addr_idx, _) = batch
        .schema()
        .column_with_name(ROW_ADDR)
        .expect("Received a batch without row addresses");
    let row_addrs_arr = batch.column(row_addr_idx);
    let row_addrs_itr = row_addrs_arr
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap_or_else(|| {
            panic!(
                "Row addresses had an unexpected type: {}",
                row_addrs_arr.data_type()
            )
        })
        .values()
        .iter()
        .copied();
    row_addrs.append(row_addrs_itr).map_err(|err| {
        datafusion::error::DataFusionError::Execution(format!(
            "Row addresses did not arrive in sorted order: {}",
            err
        ))
    })?;
    let non_row_addr_cols = (0..batch.num_columns())
        .filter(|col| *col != row_addr_idx)
        .collect::<Vec<_>>();
    Ok(batch.project(&non_row_addr_cols)?)
}

fn extract_row_ids(
    row_ids: &mut RoaringTreemap,
    batch: DFResult<RecordBatch>,
) -> DFResult<RecordBatch> {
    let batch = batch?;

    let (row_id_idx, _) = batch
        .schema()
        .column_with_name(ROW_ID)
        .expect("Received a batch without row ids");
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
        .values()
        .iter()
        .copied();
    row_ids.append(row_ids_itr).map_err(|err| {
        datafusion::error::DataFusionError::Execution(format!(
            "Row ids did not arrive in sorted order: {}",
            err
        ))
    })?;
    let non_row_ids_cols = (0..batch.num_columns())
        .filter(|col| *col != row_id_idx)
        .collect::<Vec<_>>();
    Ok(batch.project(&non_row_ids_cols)?)
}

/// Given a stream that includes a row address column, return a stream that will
/// capture the row addresses in a `RoaringTreemap` and return a stream without the
/// row address column.
pub fn make_rowaddr_capture_stream(
    row_addrs: Arc<RwLock<RoaringTreemap>>,
    target: SendableRecordBatchStream,
) -> Result<SendableRecordBatchStream> {
    let schema = target.schema();
    let stream = target.map(move |batch| {
        let mut row_addrs = row_addrs.write().unwrap();
        extract_row_addrs(&mut row_addrs, batch)
    });

    let (row_addr_idx, _) = schema
        .column_with_name(ROW_ADDR)
        .expect("Received a batch without row addresses");

    let non_row_addr_cols = (0..schema.fields.len())
        .filter(|col| *col != row_addr_idx)
        .collect::<Vec<_>>();

    let schema = Arc::new(schema.project(&non_row_addr_cols)?);

    let stream = RecordBatchStreamAdapter::new(schema, stream);
    Ok(Box::pin(stream))
}

/// Given a stream that includes a row id column, return a stream that will
/// capture the row ids in a `RoaringTreemap` and return a stream without the
/// row id column.
pub fn make_rowid_capture_stream(
    row_ids: Arc<RwLock<RoaringTreemap>>,
    target: SendableRecordBatchStream,
) -> Result<SendableRecordBatchStream> {
    let schema = target.schema();
    let stream = target.map(move |batch| {
        let mut row_ids = row_ids.write().unwrap();
        extract_row_ids(&mut row_ids, batch)
    });

    let (row_id_idx, _) = schema
        .column_with_name(ROW_ID)
        .expect("Received a batch without row ids");

    let non_row_ids_cols = (0..schema.fields.len())
        .filter(|col| *col != row_id_idx)
        .collect::<Vec<_>>();

    let schema = Arc::new(schema.project(&non_row_ids_cols)?);

    let stream = RecordBatchStreamAdapter::new(schema, stream);
    Ok(Box::pin(stream))
}
