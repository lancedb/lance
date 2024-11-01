// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::sync::{Arc, Mutex};

use arrow::array::AsArray;
use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, RecordBatchStream, SendableRecordBatchStream,
};
use futures::{Stream, StreamExt, TryStreamExt};
use lance_core::error::{CloneableResult, Error};
use lance_core::utils::futures::{Capacity, SharedStreamExt};
use lance_core::utils::mask::{RowIdMask, RowIdTreeMap};
use lance_core::{Result, ROW_ID};
use lance_index::prefilter::FilterLoader;
use snafu::{location, Location};

#[derive(Debug, Clone)]
pub enum PreFilterSource {
    /// The prefilter input is an array of row ids that match the filter condition
    FilteredRowIds(Arc<dyn ExecutionPlan>),
    /// The prefilter input is a selection vector from an index query
    ScalarIndexQuery(Arc<dyn ExecutionPlan>),
    /// There is no prefilter
    None,
}

// Utility to convert an input (containing row ids) into a prefilter
pub(crate) struct FilteredRowIdsToPrefilter(pub SendableRecordBatchStream);

#[async_trait]
impl FilterLoader for FilteredRowIdsToPrefilter {
    async fn load(mut self: Box<Self>) -> Result<RowIdMask> {
        let mut allow_list = RowIdTreeMap::new();
        while let Some(batch) = self.0.next().await {
            let batch = batch?;
            let row_ids = batch.column_by_name(ROW_ID).expect(
                "input batch missing row id column even though it is in the schema for the stream",
            );
            let row_ids = row_ids
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("row id column in input batch had incorrect type");
            allow_list.extend(row_ids.iter().flatten())
        }
        Ok(RowIdMask::from_allowed(allow_list))
    }
}

// Utility to convert a serialized selection vector into a prefilter
pub(crate) struct SelectionVectorToPrefilter(pub SendableRecordBatchStream);

#[async_trait]
impl FilterLoader for SelectionVectorToPrefilter {
    async fn load(mut self: Box<Self>) -> Result<RowIdMask> {
        let batch = self
            .0
            .try_next()
            .await?
            .ok_or_else(|| Error::Internal {
                message: "Selection vector source for prefilter did not yield any batches".into(),
                location: location!(),
            })
            .unwrap();
        RowIdMask::from_arrow(batch["result"].as_binary_opt::<i32>().ok_or_else(|| {
            Error::Internal {
                message: format!(
                    "Expected selection vector input to yield binary arrays but got {}",
                    batch["result"].data_type()
                ),
                location: location!(),
            }
        })?)
    }
}

struct InnerState {
    cached: Option<SendableRecordBatchStream>,
    taken: bool,
}

impl std::fmt::Debug for InnerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InnerState")
            .field("cached", &self.cached.is_some())
            .field("taken", &self.taken)
            .finish()
    }
}

/// An execution node that can be used as an input twice
///
/// This can be used to broadcast an input to multiple outputs.
///
/// Note: this is done by caching the results.  If one output is consumed
/// more quickly than the other, this can lead to increased memory usage.
/// The `capacity` parameter can bound this, by blocking the faster output
/// when the cache is full.  Take care not to cause deadlock.
///
/// For example, if both outputs are fed to a HashJoinExec then one side
/// of the join will be fully consumed before the other side is read.  In
/// this case, you should probably use an unbounded capacity.
#[derive(Debug)]
pub struct ReplayExec {
    capacity: Capacity,
    input: Arc<dyn ExecutionPlan>,
    inner_state: Arc<Mutex<InnerState>>,
}

impl ReplayExec {
    pub fn new(capacity: Capacity, input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            capacity,
            input,
            inner_state: Arc::new(Mutex::new(InnerState {
                cached: None,
                taken: false,
            })),
        }
    }
}

impl DisplayAs for ReplayExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "Replay: capacity={:?}", self.capacity)
            }
        }
    }
}

// There's some annoying adapter-work that needs to happen here.  In order
// to share a stream we need its items to be Clone and DataFusionError is
// not Clone.  So we wrap the stream in a CloneableResult.  However, in order
// for that shared stream to be a SendableRecordBatchStream, it needs to be
// using DataFusionError.  So we need to adapt the stream back to a
// SendableRecordBatchStream.
pub struct ShareableRecordBatchStream(pub SendableRecordBatchStream);

impl Stream for ShareableRecordBatchStream {
    type Item = CloneableResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.0.poll_next_unpin(cx) {
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Ready(Some(res)) => {
                std::task::Poll::Ready(Some(CloneableResult::from(res.map_err(Error::from))))
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

pub struct ShareableRecordBatchStreamAdapter<S: Stream<Item = CloneableResult<RecordBatch>> + Unpin>
{
    schema: SchemaRef,
    stream: S,
}

impl<S: Stream<Item = CloneableResult<RecordBatch>> + Unpin> ShareableRecordBatchStreamAdapter<S> {
    pub fn new(schema: SchemaRef, stream: S) -> Self {
        Self { schema, stream }
    }
}

impl<S: Stream<Item = CloneableResult<RecordBatch>> + Unpin> Stream
    for ShareableRecordBatchStreamAdapter<S>
{
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.stream.poll_next_unpin(cx) {
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Ready(Some(res)) => std::task::Poll::Ready(Some(
                res.0
                    .map_err(|e| DataFusionError::External(e.0.to_string().into())),
            )),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl<S: Stream<Item = CloneableResult<RecordBatch>> + Unpin> RecordBatchStream
    for ShareableRecordBatchStreamAdapter<S>
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl ExecutionPlan for ReplayExec {
    fn name(&self) -> &str {
        "ReplayExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        unimplemented!()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let mut inner_state = self.inner_state.lock().unwrap();
        if let Some(cached) = inner_state.cached.take() {
            if inner_state.taken {
                panic!("ReplayExec can only be executed twice");
            }
            inner_state.taken = true;
            Ok(cached)
        } else {
            let input = self.input.execute(partition, context)?;
            let schema = input.schema();
            let input = ShareableRecordBatchStream(input);
            let (to_return, to_cache) = input.boxed().share(self.capacity);
            inner_state.cached = Some(Box::pin(ShareableRecordBatchStreamAdapter {
                schema: schema.clone(),
                stream: to_cache,
            }));
            Ok(Box::pin(ShareableRecordBatchStreamAdapter {
                schema,
                stream: to_return,
            }))
        }
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        self.input.properties()
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use arrow_array::{types::UInt32Type, RecordBatchReader};
    use arrow_schema::SortOptions;
    use datafusion::{
        logical_expr::JoinType,
        physical_expr::expressions::Column,
        physical_plan::{
            joins::SortMergeJoinExec, stream::RecordBatchStreamAdapter, ExecutionPlan,
        },
    };
    use futures::{StreamExt, TryStreamExt};
    use lance_core::utils::futures::Capacity;
    use lance_datafusion::exec::OneShotExec;
    use lance_datagen::{array, BatchCount, RowCount};

    use super::ReplayExec;

    #[tokio::test]
    async fn test_replay() {
        let data = lance_datagen::gen()
            .col("x", array::step::<UInt32Type>())
            .into_reader_rows(RowCount::from(1024), BatchCount::from(16));
        let schema = data.schema();
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(data).map_err(datafusion::error::DataFusionError::from),
        ));

        let input = Arc::new(OneShotExec::new(data));
        let shared = Arc::new(ReplayExec::new(Capacity::Bounded(4), input));

        let joined = Arc::new(
            SortMergeJoinExec::try_new(
                shared.clone(),
                shared,
                vec![(Arc::new(Column::new("x", 0)), Arc::new(Column::new("x", 0)))],
                None,
                JoinType::Inner,
                vec![SortOptions::default()],
                true,
            )
            .unwrap(),
        );

        let mut join_stream = joined
            .execute(0, Arc::new(datafusion::execution::TaskContext::default()))
            .unwrap();

        while let Some(batch) = join_stream.next().await {
            // We don't test much here but shouldn't really need to.  The join and stream sharing
            // are tested on their own.  We just need to make sure they get hooked up correctly
            assert_eq!(batch.unwrap().num_columns(), 2);
        }
    }
}
