// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use datafusion::{
    execution::SendableRecordBatchStream,
    physical_plan::{stream::RecordBatchStreamAdapter, ExecutionPlan},
};
use datafusion_common::{DataFusionError, Result};
use futures::TryStreamExt;
use lance_datagen::{BatchCount, BatchGeneratorBuilder, RowCount};

use crate::exec::{OneShotExec, RecordBatchExec};

pub trait DatafusionDatagenExt {
    fn into_df_stream(
        self,
        batch_size: RowCount,
        num_batches: BatchCount,
    ) -> SendableRecordBatchStream;

    fn into_df_once_exec(
        self,
        batch_size: RowCount,
        num_batches: BatchCount,
    ) -> Arc<dyn ExecutionPlan>;

    fn into_df_repeat_exec(
        self,
        batch_size: RowCount,
        num_batches: BatchCount,
    ) -> Result<Arc<dyn ExecutionPlan>>;
}

impl DatafusionDatagenExt for BatchGeneratorBuilder {
    fn into_df_stream(
        self,
        batch_size: RowCount,
        num_batches: BatchCount,
    ) -> SendableRecordBatchStream {
        let (stream, schema) = self.into_reader_stream(batch_size, num_batches);
        let stream = stream.map_err(DataFusionError::from);
        Box::pin(RecordBatchStreamAdapter::new(schema, stream))
    }

    fn into_df_once_exec(
        self,
        batch_size: RowCount,
        num_batches: BatchCount,
    ) -> Arc<dyn ExecutionPlan> {
        let stream = self.into_df_stream(batch_size, num_batches);
        Arc::new(OneShotExec::new(stream))
    }

    fn into_df_repeat_exec<'a>(
        self,
        batch_size: RowCount,
        num_batches: BatchCount,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let reader = self.into_reader_rows(batch_size, num_batches);
        let batches = reader
            .collect::<Vec<_>>()
            .into_iter()
            .map(|r| match r {
                Ok(batch) => Ok(batch),
                Err(e) => Err(DataFusionError::Execution(e.to_string())),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(RecordBatchExec::new(batches)?))
    }
}
