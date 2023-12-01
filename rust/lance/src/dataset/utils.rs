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
use std::sync::{Arc, RwLock};

use arrow_array::{RecordBatch, UInt64Array};
use datafusion::error::Result as DFResult;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::StreamExt;
use roaring::RoaringTreemap;

use crate::dataset::ROW_ID;
use crate::Result;

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
