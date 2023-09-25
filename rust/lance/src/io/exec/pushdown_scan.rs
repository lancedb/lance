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

use std::collections::HashMap;
use std::{any::Any, sync::Arc};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, Int64Array, PrimitiveArray, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::optimizer::simplify_expressions::{ExprSimplifier, SimplifyContext};
use datafusion::physical_expr::execution_props::ExecutionProps;
use datafusion::physical_expr::intervals::{Interval, IntervalBound, NullableInterval};
use datafusion::physical_plan::ColumnarValue;
use datafusion::prelude::Column;
use datafusion::scalar::ScalarValue;
use datafusion::{
    physical_plan::{
        stream::RecordBatchStreamAdapter, DisplayAs, DisplayFormatType, ExecutionPlan,
        Partitioning, SendableRecordBatchStream,
    },
    prelude::Expr,
};
use futures::{Stream, StreamExt, TryStreamExt};
use lance_arrow::RecordBatchExt;
use snafu::{location, Location};

use crate::dataset::scanner::{DEFAULT_BATCH_READAHEAD, DEFAULT_FRAGMENT_READAHEAD};
use crate::io::ReadBatchParams;
use crate::Error;
use crate::{
    dataset::{
        fragment::{FileFragment, FragmentReader},
        ROW_ID,
    },
    datatypes::Schema,
    format::Fragment,
    Dataset,
};

// TODO:
// [ ] Test with row_id
// [ ] Handle scanning out-of-order
// [ ] Test interval creation
// [ ] Test predicate simplification
// [ ] Hook up into planner
// [ ] Benchmark various scenarios

use super::Planner;

#[derive(Debug, Clone)]
pub struct ScanConfig {
    batch_readahead: usize,
    fragment_readahead: usize,
    with_row_id: bool,
    ordered_output: bool,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            batch_readahead: DEFAULT_BATCH_READAHEAD,
            fragment_readahead: DEFAULT_FRAGMENT_READAHEAD,
            with_row_id: false,
            ordered_output: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LancePushdownScanExec {
    dataset: Arc<Dataset>,
    fragments: Arc<Vec<Fragment>>,
    projection: Arc<Schema>,
    predicate_projection: Arc<Schema>,
    predicate: Expr,
    config: ScanConfig,
}

impl LancePushdownScanExec {
    pub fn try_new(
        dataset: Arc<Dataset>,
        fragments: Arc<Vec<Fragment>>,
        projection: Arc<Schema>,
        predicate: Expr,
        config: ScanConfig,
    ) -> Result<Self> {
        // This should be infallible.
        let columns: Vec<_> = predicate
            .to_columns()
            .unwrap()
            .into_iter()
            .map(|col| col.name)
            .collect();
        let dataset_schema = dataset.schema();
        let predicate_projection = Arc::new(dataset_schema.project(&columns)
            .map_err(|err| Error::invalid_input(format!("Filter predicate '{:?}' references columns {:?}, but some of them were not found in the dataset schema: {}\nInner error: {:?}", predicate, columns, dataset_schema, err), location!()))?);

        Ok(Self {
            dataset,
            fragments,
            projection,
            predicate,
            predicate_projection,
            config,
        })
    }
}

impl ExecutionPlan for LancePushdownScanExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        let schema: ArrowSchema = self.projection.as_ref().into();
        if self.config.with_row_id {
            let mut fields: Vec<Arc<Field>> = schema.fields.to_vec();
            fields.push(Arc::new(Field::new(ROW_ID, DataType::UInt64, false)));
            Arc::new(ArrowSchema::new(fields))
        } else {
            Arc::new(schema)
        }
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
        None
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        todo!()
    }

    fn statistics(&self) -> datafusion::physical_plan::Statistics {
        todo!()
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // To get a stream with a static lifetime, we clone self put it into
        // a stream.
        let state = (self.clone(), 0);
        let fragment_stream = futures::stream::unfold(state, |(exec, fragment_i)| async move {
            if fragment_i == exec.fragments.len() {
                None
            } else {
                let fragment = exec.fragments[fragment_i].clone();
                let exec_ref = exec.clone();
                let res = (exec_ref, fragment);
                let new_state = (exec, fragment_i + 1);
                Some((res, new_state))
            }
        });

        let batch_stream = fragment_stream.map(|(exec, fragment)| async move {
            let frag_scanner = FragmentScanner::open(
                fragment,
                exec.dataset,
                exec.projection,
                exec.predicate_projection,
                exec.predicate,
                exec.config.batch_readahead,
                exec.config.with_row_id,
                exec.config.ordered_output,
            )
            .await?;

            frag_scanner.scan().await
        });

        let batch_stream = batch_stream
            .buffered(self.config.fragment_readahead)
            .try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            batch_stream,
        )))
    }
}

impl DisplayAs for LancePushdownScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let columns = self
                    .projection
                    .fields
                    .iter()
                    .map(|f| f.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "LancePushdownScan: uri={}, projection=[{}], predicate={}, row_id=[{}], ordered={}",
                    self.dataset.data_dir(),
                    columns,
                    self.predicate,
                    self.config.with_row_id,
                    self.config.ordered_output
                )
            }
        }
    }
}

#[derive(Debug)]
struct FragmentScanner {
    fragment: FileFragment,
    projection: Arc<Schema>,
    predicate_projection: Arc<Schema>,
    predicate: Expr,
    batch_readahead: usize,
    ordered_output: bool,
    /// Cache of readers for each projection. For each batch, we scan the
    /// minimum number of columns needed to evaluate the predicate, so which
    /// columns are relevant change from batch to batch. To reduce the cost
    /// of opening readers, we cache them here.
    ///
    /// It's worth noting that the [FileFragment::open()] method caches it's IO
    /// calls, so there shouldn't be a meaningful cost to each additional reader
    /// we open beyond the CPU time required to instantiate them.
    ///
    /// We don't intend to hold this across await points, so we just use the std
    /// Mutex instead of the tokio one.
    reader_cache: Arc<std::sync::Mutex<HashMap<Vec<i32>, Arc<FragmentReader>>>>,
    stats: Option<RecordBatch>,
}

impl FragmentScanner {
    pub async fn open(
        fragment: Fragment,
        dataset: Arc<Dataset>,
        projection: Arc<Schema>,
        predicate_projection: Arc<Schema>,
        predicate: Expr,
        batch_readahead: usize,
        with_row_id: bool,
        ordered_output: bool,
    ) -> Result<Self> {
        let fragment = FileFragment::new(dataset.clone(), fragment);
        let mut filter_reader = fragment.open(&predicate_projection).await?;
        if with_row_id {
            filter_reader.with_row_id();
        }
        // We read the stats off of the filter column reader, since that contains
        // the statistics we need to simplify the predicate.
        let stats = filter_reader.read_page_stats().await?;

        let mut reader_cache = HashMap::new();
        reader_cache.insert(predicate_projection.field_ids(), Arc::new(filter_reader));

        let projection_reader = fragment.open(&projection).await?;
        reader_cache.insert(projection.field_ids(), Arc::new(projection_reader));

        Ok(Self {
            fragment,
            projection,
            predicate_projection,
            predicate,
            batch_readahead,
            ordered_output,
            reader_cache: Arc::new(std::sync::Mutex::new(reader_cache)),
            stats,
        })
    }

    pub async fn scan(self) -> Result<impl Stream<Item = Result<RecordBatch>> + 'static + Send> {
        // This reader was inserted on initialization, so it's guaranteed to be there.
        let num_batches = self
            .reader_cache
            .lock()
            .unwrap()
            .get(&self.predicate_projection.field_ids())
            .unwrap()
            .num_batches();
        let batch_readahead = self.batch_readahead;
        let simplified_predicates = self.simplified_predicates()?;

        let scanner = Arc::new(self);

        Ok(
            futures::stream::iter((0..num_batches).zip(simplified_predicates.into_iter()))
                .map(move |(batch_id, predicate)| {
                    let scanner_ref = scanner.clone();
                    async move { scanner_ref.read_batch(batch_id, predicate).await }
                })
                .buffered(batch_readahead)
                .try_filter_map(|res| futures::future::ready(Ok(res)))
                .boxed(),
        )
    }

    async fn read_batch(&self, batch_id: usize, predicate: Expr) -> Result<Option<RecordBatch>> {
        match predicate {
            Expr::Literal(ScalarValue::Boolean(Some(true))) => {
                // Predicate is always true, we just need to load the projection.

                // TODO: what about the row id column?
                let projection_reader = self
                    .reader_cache
                    .lock()
                    .unwrap()
                    .get(&self.projection.field_ids())
                    .unwrap()
                    .clone();
                let batch = projection_reader.read_batch(batch_id, ..).await?;
                Ok(Some(batch))
            }
            Expr::Literal(ScalarValue::Boolean(Some(false))) => {
                // Predicate is always false, we can skip this batch.
                Ok(None)
            }
            _ => {
                // Predicate is not always true or always false, we need to load
                // the filter columns to evaluate it.

                // 1. Load needed filter columns, which might be a subset of all filter
                //    columns if statistics obviated the need for some columns.
                let columns: Vec<_> = predicate
                    .to_columns()
                    .unwrap()
                    .into_iter()
                    .map(|col| col.name)
                    .collect();
                let predicate_projection =
                    Arc::new(self.fragment.dataset().schema().project(&columns).unwrap());
                let field_ids = predicate_projection.field_ids();
                let maybe_reader = self.reader_cache.lock().unwrap().get(&field_ids).cloned();
                let filter_reader = match maybe_reader {
                    Some(reader) => reader,
                    None => {
                        let filter_reader =
                            Arc::new(self.fragment.open(&predicate_projection).await?);
                        self.reader_cache
                            .lock()
                            .unwrap()
                            .insert(field_ids.clone(), filter_reader.clone());
                        filter_reader
                    }
                };

                let batch = filter_reader.read_batch(batch_id, ..).await?;

                // 2. Evaluate predicate
                let planner = Planner::new(batch.schema());
                let physical_expr = planner.create_physical_expr(&predicate)?;
                let result = physical_expr.evaluate(&batch)?;
                let selection = match result {
                    ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => {
                        ReadBatchParams::RangeFull
                    }
                    ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))) => {
                        // Nothing matched
                        return Ok(None);
                    }
                    ColumnarValue::Array(array) => {
                        let array = array
                            .as_any()
                            .downcast_ref::<arrow_array::BooleanArray>()
                            .unwrap();
                        let selection: UInt32Array = array
                            .iter()
                            .enumerate()
                            .filter_map(|(i, value)| {
                                if value.unwrap_or_default() {
                                    Some(i as u32)
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if selection.is_empty() {
                            // Nothing matched the predicate, so the result is empty.
                            return Ok(None);
                        }

                        ReadBatchParams::Indices(selection)
                    }
                    _ => {
                        return Err(DataFusionError::Internal(format!(
                            "Unexpected result from predicate evaluation: {:?}",
                            result
                        )))
                    }
                };

                // 3. Take the subset of rows that pass the predicate. Some of the
                //    columns in the projection may have already been loaded as a
                //    filter column, so we just load the remaining columns.
                let projection_field_ids = self.projection.field_ids();
                let remaining_fields: Vec<i32> = projection_field_ids
                    .iter()
                    .cloned()
                    .filter(|i| !field_ids.contains(i))
                    .collect();
                let remainder_batch = if !remaining_fields.is_empty() {
                    let maybe_reader = self
                        .reader_cache
                        .lock()
                        .unwrap()
                        .get(&remaining_fields)
                        .cloned();
                    let remainder_reader = match maybe_reader {
                        Some(reader) => reader,
                        None => {
                            let remainder_reader = Arc::new(
                                self.fragment
                                    .open(&self.projection.project_by_ids(&remaining_fields))
                                    .await?,
                            );
                            self.reader_cache
                                .lock()
                                .unwrap()
                                .insert(field_ids.clone(), remainder_reader.clone());
                            remainder_reader
                        }
                    };
                    Some(
                        remainder_reader
                            .read_batch(batch_id, selection.clone())
                            .await?,
                    )
                } else {
                    None
                };

                // 4. If filter columns are in the projection, merge them in
                //    and project the final schema.
                let batch = if let ReadBatchParams::Indices(indices) = selection {
                    batch.take(&indices)?
                } else {
                    batch
                };
                let batch = if let Some(remainder_batch) = remainder_batch {
                    batch.merge(&remainder_batch)?
                } else {
                    batch
                };
                let batch = batch.project_by_schema(&self.projection.as_ref().into())
                    .map_err(|err| Error::Internal {
                        message: format!("Failed to to select schema {} from batch with schema {}\nInner error: {}", self.projection, batch.schema(), err),
                        location: location!()
                    })?;

                Ok(Some(batch))
            }
        }
    }

    /// Parse the statistics into a set of guarantees for each batch.
    fn extract_guarantees<'a>(
        predicate_projection: &'a Schema,
        batch_sizes: &'a [usize],
        stats: &RecordBatch,
    ) -> impl Iterator<Item = Vec<(Expr, NullableInterval)>> + 'a {
        let mut null_counts: HashMap<i32, PrimitiveArray<Int64Type>> = HashMap::new();
        let mut min_values: HashMap<i32, Arc<dyn Array>> = HashMap::new();
        let mut max_values: HashMap<i32, Arc<dyn Array>> = HashMap::new();

        for field_id in predicate_projection.field_ids() {
            let field_stats = stats.column_by_name(&format!("{field_id}"));
            if let Some(field_stats) = field_stats {
                if !matches!(field_stats.data_type(), DataType::Struct(_)) {
                    log::error!(
                        "Invalid statistics: Field stats for field {} is not a struct, but a {}",
                        field_id,
                        field_stats.data_type()
                    );
                    continue;
                }
                let field_stats_ref = field_stats.as_struct();

                if let Some(null_count_col) = field_stats_ref.column_by_name("null_count") {
                    let null_count_col = null_count_col
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    null_counts.insert(field_id, null_count_col.clone());
                }

                if let Some(min_col) = field_stats_ref.column_by_name("min_value") {
                    min_values.insert(field_id, min_col.clone());
                }

                if let Some(max_col) = field_stats_ref.column_by_name("max_value") {
                    max_values.insert(field_id, max_col.clone());
                }
            }
        }

        (0..stats.num_rows())
            .map(move |batch_id| {
                let mut guarantees = Vec::new();
                for field in predicate_projection.fields_pre_order() {
                    let null_count = if field.nullable {
                        let maybe_null_count = null_counts.get(&field.id).map(|arr| arr.value(batch_id));
                        if let Some(null_count) = maybe_null_count {
                            null_count
                        } else {
                            continue
                        }
                    } else {
                        0
                    };

                    let min_value = {
                        let maybe_min_value = min_values.get(&field.id)
                            .map(|arr| ScalarValue::try_from_array(arr, batch_id));
                        match maybe_min_value {
                            Some(Ok(min_value)) => min_value,
                            Some(Err(err)) => {
                                log::error!("Invalid statistics: Failed to convert min_value for field {} to ScalarValue: {}", field.id, err);
                                continue
                            }
                            None => continue
                        }
                    };

                    let max_value = {
                        let maybe_max_value = max_values.get(&field.id)
                            .map(|arr| ScalarValue::try_from_array(arr, batch_id));
                        match maybe_max_value {
                            Some(Ok(max_value)) => max_value,
                            Some(Err(err)) => {
                                log::error!("Invalid statistics: Failed to convert max_value for field {} to ScalarValue: {}", field.id, err);
                                continue
                            }
                            None => continue
                        }
                    };

                    let values = Interval::new(
                        IntervalBound::new_closed(min_value),
                        IntervalBound::new_closed(max_value)
                    );
                    let batch_size = batch_sizes[batch_id];
                    let interval = match (null_count, batch_size) {
                        (0, _) => NullableInterval::NotNull { values },
                        (null_count, batch_size) if null_count == batch_size as i64 => NullableInterval::Null { datatype: field.data_type() },
                        _ => NullableInterval::MaybeNull { values }
                    };
                    let fully_qualified_name = predicate_projection.field_ancestry_by_id(field.id).unwrap()
                        .into_iter()
                        .map(|field| field.name.as_str())
                        .collect::<Vec<&str>>()
                        .join(".");
                    let expr = Expr::Column(Column::from_name(fully_qualified_name));
                    guarantees.push((expr, interval));
                }
                guarantees
            })
    }

    fn simplified_predicates(&self) -> Result<Vec<Expr>> {
        let reader = {
            let cache = self.reader_cache.lock().unwrap();
            cache
                .get(&self.predicate_projection.field_ids())
                .unwrap()
                .clone()
        };
        let num_batches = reader.num_batches();

        if let Some(stats) = &self.stats {
            let batch_sizes: Vec<usize> = (0..num_batches)
                .map(|batch_id| reader.num_rows_in_batch(batch_id))
                .collect();
            let schema =
                Arc::new(ArrowSchema::from(self.predicate_projection.as_ref()).try_into()?);
            let props = ExecutionProps::new();
            let context = SimplifyContext::new(&props).with_schema(schema);
            let mut simplifier = ExprSimplifier::new(context);

            let mut predicates = Vec::with_capacity(num_batches);
            for guarantees in
                Self::extract_guarantees(&self.predicate_projection, &batch_sizes, &stats)
            {
                dbg!(&guarantees);
                simplifier = simplifier.with_guarantees(guarantees);
                predicates.push(simplifier.simplify(self.predicate.clone())?);
            }
            Ok(predicates)
        } else {
            Ok(vec![self.predicate.clone(); num_batches])
        }
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{ArrayRef, Int32Array, RecordBatchIterator, StructArray, UInt64Array};
    use datafusion::prelude::{col, lit, Column, SessionContext};
    use tempfile::tempdir;

    use crate::dataset::WriteParams;

    use super::*;

    #[tokio::test]
    async fn test_empty_result() {
        // Test we can get no results
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let num_rows: usize = 10;
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..num_rows as i32))],
        )
        .unwrap()];
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        let dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let fragments = dataset.fragments().clone();
        let projection = Arc::new(dataset.schema().clone());

        let predicate = col("i").eq(lit(42));

        let exec = LancePushdownScanExec::try_new(
            Arc::new(dataset),
            fragments,
            projection,
            predicate,
            ScanConfig::default(),
        )
        .unwrap();

        let ctx = SessionContext::new();

        let results = exec.execute(0, ctx.task_ctx()).unwrap();
        let results = results.try_collect::<Vec<_>>().await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_nested_filter() {
        // Validate we can filter and project nested columns and they will be
        // merged correctly.
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let field_a = Arc::new(Field::new("a", DataType::Int32, false));
        let field_b = Arc::new(Field::new("b", DataType::Int32, false));
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(
                "x",
                DataType::Struct(vec![field_a.clone(), field_b.clone()].into()),
                false,
            ),
            Field::new(
                "y",
                DataType::Struct(vec![field_a.clone(), field_b.clone()].into()),
                false,
            ),
        ]));

        let array = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StructArray::from(vec![
                    (field_a.clone(), array.clone() as ArrayRef),
                    (field_b.clone(), array.clone() as ArrayRef),
                ])),
                Arc::new(StructArray::from(vec![
                    (field_a.clone(), array.clone() as ArrayRef),
                    (field_b.clone(), array.clone() as ArrayRef),
                ])),
            ],
        )
        .unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let dataset = Arc::new(Dataset::write(batches, test_uri, None).await.unwrap());

        let fragments = dataset.fragments().clone();
        // [x.b, y.a]
        let projection = Arc::new(dataset.schema().clone().project_by_ids(&[2, 4]));

        let predicate = col(Column::from_name("x.a"))
            .lt(lit(8))
            .and(col(Column::from_name("y.b")).gt(lit(3)));

        let exec = LancePushdownScanExec::try_new(
            dataset.clone(),
            fragments.clone(),
            projection,
            predicate.clone(),
            ScanConfig::default(),
        )
        .unwrap();

        let ctx = SessionContext::new();

        let results = exec.execute(0, ctx.task_ctx()).unwrap();
        let results = results.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 4);

        let expected_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("x", DataType::Struct(vec![field_b.clone()].into()), false),
            Field::new("y", DataType::Struct(vec![field_a.clone()].into()), false),
        ]));
        assert_eq!(results[0].schema().as_ref(), expected_schema.as_ref());

        // Also try where projection is same as filter columns
        let projection = Arc::new(dataset.schema().clone().project_by_ids(&[1, 5]));
        let exec = LancePushdownScanExec::try_new(
            dataset.clone(),
            fragments,
            projection,
            predicate,
            ScanConfig::default(),
        )
        .unwrap();
        let results = exec.execute(0, ctx.task_ctx()).unwrap();
        let results = results.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 4);

        let expected_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("x", DataType::Struct(vec![field_a.clone()].into()), false),
            Field::new("y", DataType::Struct(vec![field_b.clone()].into()), false),
        ]));
        assert_eq!(results[0].schema().as_ref(), expected_schema.as_ref());
    }

    #[tokio::test]
    async fn test_with_row_id() {
        // Want to hit all three code paths:
        // Batches: [0..100], [100..200], [200..300]
        // Predicate: s.x >= 150
        // Expected simplification: false, s.x > 150, true
        // Expected result: [150..300]

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let field = Arc::new(Field::new("x", DataType::Int32, false));
        let field_outer = Field::new("s", DataType::Struct(vec![field.clone()].into()), false);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![field_outer]));

        let batches = [0..100, 100..200, 200..300].map(|range| {
            RecordBatch::try_new(
                arrow_schema.clone(),
                vec![Arc::new(StructArray::from(vec![(
                    field.clone(),
                    Arc::new(Int32Array::from_iter_values(range)) as ArrayRef,
                )]))],
            )
        });

        let batches = RecordBatchIterator::new(batches, arrow_schema.clone());

        let write_params = WriteParams {
            max_rows_per_group: 100,
            ..Default::default()
        };
        let dataset = Arc::new(
            Dataset::write(batches, test_uri, Some(write_params))
                .await
                .unwrap(),
        );

        let fragments = dataset.fragments().clone();
        assert_eq!(fragments.len(), 1);
        let fragment = fragments[0].clone();

        let predicate = col(Column::from_name("s.x")).gt_eq(lit(150));

        let schema = Arc::new(dataset.schema().clone());

        let fragment_scanner = FragmentScanner::open(
            fragment,
            dataset.clone(),
            schema.clone(),
            schema.clone(),
            predicate.clone(),
            1,
            true,
            true,
        )
        .await
        .unwrap();

        dbg!(&fragment_scanner);

        let predicates = fragment_scanner.simplified_predicates().unwrap();
        assert_eq!(predicates.len(), 3);
        assert_eq!(&predicates[0], &lit(false));
        assert_eq!(&predicates[1], &predicate);
        assert_eq!(&predicates[2], &lit(true));

        let batch0 = fragment_scanner
            .read_batch(0, predicates[0].clone())
            .await
            .unwrap();
        assert!(batch0.is_none());

        let batch1 = fragment_scanner
            .read_batch(1, predicates[1].clone())
            .await
            .unwrap();
        assert!(batch1.is_some());
        let batch1 = batch1.unwrap();
        let expected_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("s", DataType::Struct(vec![field.clone()].into()), false),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let expected = RecordBatch::try_new(
            expected_schema.clone(),
            vec![
                Arc::new(StructArray::from(vec![(
                    field.clone(),
                    Arc::new(Int32Array::from_iter_values(150..200)) as ArrayRef,
                )])),
                Arc::new(UInt64Array::from_iter_values(150..200)),
            ],
        )
        .unwrap();
        assert_eq!(batch1, expected);

        let batch2 = fragment_scanner
            .read_batch(2, predicates[2].clone())
            .await
            .unwrap();
        assert!(batch2.is_some());
        let batch2 = batch2.unwrap();
        let expected = RecordBatch::try_new(
            expected_schema.clone(),
            vec![
                Arc::new(StructArray::from(vec![(
                    field.clone(),
                    Arc::new(Int32Array::from_iter_values(200..300)) as ArrayRef,
                )])),
                Arc::new(UInt64Array::from_iter_values(200..300)),
            ],
        )
        .unwrap();
        assert_eq!(batch2, expected);
    }

    // property based testing
    // All column types
    // choose random subset to project
    // random range to filter for
    // random over with or without rowid
}
