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
use arrow_select::filter::filter_record_batch;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::col;
use datafusion::optimizer::simplify_expressions::{ExprSimplifier, SimplifyContext};
use datafusion::physical_expr::execution_props::ExecutionProps;
use datafusion::physical_expr::intervals::{Interval, IntervalBound, NullableInterval};
use datafusion::physical_plan::ColumnarValue;
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
use lance_core::io::ReadBatchParams;
use lance_core::ROW_ID_FIELD;
use snafu::{location, Location};

use crate::dataset::scanner::{DEFAULT_BATCH_READAHEAD, DEFAULT_FRAGMENT_READAHEAD};
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

use super::Planner;

#[derive(Debug, Clone)]
pub struct ScanConfig {
    /// Number of batches to read ahead, potentially concurrently. This determines
    /// the amount of concurrent IO a scan may perform as well as the amount of
    /// data that may be buffered.
    pub batch_readahead: usize,
    /// Number of fragments to read ahead, potentially concurrently.
    pub fragment_readahead: usize,
    /// If true, a row id column will be added to the output. This will be the
    /// first column.
    pub with_row_id: bool,
    /// If true, the scan will emit batches that contain deleted rows but have
    /// a null _rowid. This option is only valid if with_row_id is true.
    pub make_deletions_null: bool,
    /// If true (default), the scan will emit batches in the same order as the
    /// fragments. If false, the scan may emit batches in an arbitrary order.
    pub ordered_output: bool,
    /// Whether to use statistics to optimize the scan. This is primarily used
    /// for testing purposes.
    pub use_stats: bool,
    /// The number of rows to read per batch.
    pub batch_size: usize, // TODO: respect batch size.
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            batch_readahead: DEFAULT_BATCH_READAHEAD,
            fragment_readahead: DEFAULT_FRAGMENT_READAHEAD,
            with_row_id: false,
            make_deletions_null: false,
            ordered_output: true,
            use_stats: true,
            batch_size: 1024,
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

        if config.make_deletions_null && !config.with_row_id {
            return Err(DataFusionError::Configuration(
                "make_deletions_null is only valid if with_row_id is true".into(),
            ));
        }

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
                exec.config.clone(),
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
    config: ScanConfig,
    reader: FragmentReader,
    stats: Option<RecordBatch>,
}

impl FragmentScanner {
    pub async fn open(
        fragment: Fragment,
        dataset: Arc<Dataset>,
        projection: Arc<Schema>,
        predicate_projection: Arc<Schema>,
        predicate: Expr,
        config: ScanConfig,
    ) -> Result<Self> {
        let fragment = FileFragment::new(dataset.clone(), fragment);

        // We will call the reader with projections. In order for this to work
        // we must ensure that we open the fragment with the maximal schema.
        let mut reader = fragment.open(dataset.schema()).await?;
        if config.make_deletions_null {
            reader.with_make_deletions_null();
        }

        // We only need the statistics for the predicate projection.
        let stats = reader.read_page_stats(Some(&predicate_projection)).await?;

        Ok(Self {
            fragment,
            projection,
            predicate_projection,
            predicate,
            config,
            reader,
            stats,
        })
    }

    pub async fn scan(self) -> Result<impl Stream<Item = Result<RecordBatch>> + 'static + Send> {
        let batch_readahead = self.config.batch_readahead;
        let simplified_predicates = self.simplified_predicates()?;
        let ordered_output = self.config.ordered_output;

        let scanner = Arc::new(self);

        let stream = futures::stream::iter(simplified_predicates.into_iter().enumerate()).map(
            move |(batch_id, predicate)| {
                let scanner_ref = scanner.clone();
                async move { scanner_ref.read_batch(batch_id, predicate).await }
            },
        );

        let stream = if ordered_output {
            stream
                .buffered(batch_readahead)
                .try_filter_map(|res| futures::future::ready(Ok(res)))
                .boxed()
        } else {
            stream
                .buffer_unordered(batch_readahead)
                .try_filter_map(|res| futures::future::ready(Ok(res)))
                .boxed()
        };

        Ok(stream)
    }

    async fn read_batch(&self, batch_id: usize, predicate: Expr) -> Result<Option<RecordBatch>> {
        match predicate {
            Expr::Literal(ScalarValue::Boolean(Some(true))) => {
                // Predicate is always true, we just need to load the projection.
                let mut projection_reader = self.reader.clone();
                if self.config.with_row_id {
                    projection_reader.with_row_id();
                }
                let batch = projection_reader
                    .read_batch_projected(batch_id, .., &self.projection)
                    .await?;
                let batch = self.final_projection(batch)?;
                Ok(Some(batch))
            }
            Expr::Literal(ScalarValue::Boolean(Some(false))) => {
                // Predicate is always false, we can skip this batch.
                Ok(None)
            }
            _ if !self.config.use_stats => {
                // 1. Load the full projection
                let mut reader = self.reader.clone();
                if self.config.with_row_id {
                    reader.with_row_id();
                }
                let batch = reader.read_batch(batch_id, ..).await?;

                // 2. Evaluate the predicate
                let planner = Planner::new(batch.schema());
                let physical_expr = planner.create_physical_expr(&predicate)?;
                let result = physical_expr.evaluate(&batch)?.into_array(batch.num_rows());

                // 3. Apply the final projection
                let batch = self.final_projection(batch)?;

                // 4. Apply the filter
                let mask = result.as_boolean();
                let batch = filter_record_batch(&batch, mask)?;

                Ok(Some(batch))
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
                let mut reader = self.reader.clone();
                if self.config.with_row_id {
                    reader.with_row_id();
                }
                let batch = reader
                    .read_batch_projected(batch_id, .., &predicate_projection)
                    .await?;

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
                let predicate_field_ids = predicate_projection.field_ids();
                let projection_field_ids = self.projection.field_ids();
                let remaining_fields: Vec<i32> = projection_field_ids
                    .iter()
                    .cloned()
                    .filter(|i| !predicate_field_ids.contains(i))
                    .collect();

                let remainder_batch = if !remaining_fields.is_empty() {
                    let remaining_projection = self.projection.project_by_ids(&remaining_fields);
                    Some(
                        self.reader
                            .read_batch_projected(
                                batch_id,
                                selection.clone(),
                                &remaining_projection,
                            )
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
                    debug_assert_eq!(batch.num_rows(), remainder_batch.num_rows());
                    batch.merge(&remainder_batch)?
                } else {
                    batch
                };

                let batch = self.final_projection(batch)?;

                Ok(Some(batch))
            }
        }
    }

    fn final_projection(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let row_id_column = batch.column_by_name(ROW_ID).cloned();

        if self.projection.fields.is_empty() && row_id_column.is_some() {
            return Ok(RecordBatch::try_new(
                Arc::new(ArrowSchema::new(vec![ROW_ID_FIELD.clone()])),
                vec![row_id_column.unwrap()],
            )?);
        }

        let mut batch = batch
            .project_by_schema(&self.projection.as_ref().into())
            .map_err(|err| Error::Internal {
                message: format!(
                    "Failed to to select schema {} from batch with schema {}\nInner error: {}",
                    self.projection,
                    batch.schema(),
                    err
                ),
                location: location!(),
            })?;

        // Row id wasn't part of the projection, so we need to add it back if it
        // was requested. We always put it at the front.
        if self.config.with_row_id {
            batch = batch.try_with_column_at(0, ROW_ID_FIELD.clone(), row_id_column.unwrap())?;
        }

        Ok(batch)
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
                    let column_path = predicate_projection.field_ancestry_by_id(field.id).unwrap();
                    let mut parts_iter = column_path.into_iter().map(|part| part.name.as_str());
                    let mut expr = col(parts_iter.next().unwrap());
                    for part in parts_iter {
                        expr = expr.field(part);
                    }
                    guarantees.push((expr, interval));
                }
                guarantees
            })
    }

    fn simplified_predicates(&self) -> Result<Vec<Expr>> {
        let num_batches = self.reader.num_batches();

        if let Some(stats) = &self.stats {
            let batch_sizes: Vec<usize> = (0..num_batches)
                .map(|batch_id| self.reader.num_rows_in_batch(batch_id))
                .collect();
            let schema =
                Arc::new(ArrowSchema::from(self.predicate_projection.as_ref()).try_into()?);
            let props = ExecutionProps::new();
            let context = SimplifyContext::new(&props).with_schema(schema);
            let mut simplifier = ExprSimplifier::new(context);

            let mut predicates = Vec::with_capacity(num_batches);
            for guarantees in
                Self::extract_guarantees(&self.predicate_projection, &batch_sizes, stats)
            {
                simplifier = simplifier.with_guarantees(guarantees);
                let simplified_expr = match simplifier.simplify(self.predicate.clone()) {
                    Ok(expr) => expr,
                    Err(err) => {
                        // TODO: this logs on each iteration, but maybe should should
                        // only log once per call of this func?
                        log::debug!("Failed to simplify predicate: {}", err);
                        self.predicate.clone()
                    }
                };

                predicates.push(simplified_expr);
            }
            Ok(predicates)
        } else {
            Ok(vec![self.predicate.clone(); num_batches])
        }
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{
        types::{Int32Type, UInt64Type},
        ArrayRef, DictionaryArray, FixedSizeListArray, Float32Array, Int32Array,
        RecordBatchIterator, StringArray, StructArray, TimestampMicrosecondArray, UInt64Array,
    };
    use arrow_ord::sort::sort_to_indices;
    use arrow_schema::TimeUnit;
    use arrow_select::concat::concat_batches;
    use datafusion::prelude::{col, lit, Column, SessionContext};
    use lance_arrow::{FixedSizeListArrayExt, SchemaExt};
    use tempfile::tempdir;

    use crate::dataset::WriteParams;

    use super::*;

    // TODO: test pushdown with nested column once https://github.com/apache/arrow-datafusion/pull/8256
    // is released.

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

        let predicate = col("x")
            .field("a")
            .lt(lit(8))
            .and(col("y").field("b").gt(lit(3)));

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
        // Want to hit all three code paths: all filtered out, partially filtered
        // out, and none filtered out.
        // Batches: [0..100], [100..200], [200..300]
        // Predicate: s.x >= 150
        // Expected simplification: false, s.x > 150, true
        // Expected result: [150..300]

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let field = Arc::new(Field::new("int", DataType::Int32, false));

        let arrow_schema = Arc::new(ArrowSchema::new(vec![field.clone()]));

        let batches = [0..100, 100..200, 200..300].map(|range| {
            RecordBatch::try_new(
                arrow_schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(range)) as ArrayRef],
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

        let predicate = col("int").gt_eq(lit(150));

        let schema = Arc::new(dataset.schema().clone());

        let fragment_scanner = FragmentScanner::open(
            fragment,
            dataset.clone(),
            schema.clone(),
            schema.clone(),
            predicate.clone(),
            ScanConfig {
                with_row_id: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

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
            ROW_ID_FIELD.clone(),
            field.as_ref().clone(),
        ]));
        let expected = RecordBatch::try_new(
            expected_schema.clone(),
            vec![
                Arc::new(UInt64Array::from_iter_values(150..200)),
                Arc::new(Int32Array::from_iter_values(150..200)),
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
                Arc::new(UInt64Array::from_iter_values(200..300)),
                Arc::new(Int32Array::from_iter_values(200..300)),
            ],
        )
        .unwrap();
        assert_eq!(batch2, expected);
    }

    fn test_data() -> RecordBatch {
        let arrow_schema = ArrowSchema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
                false,
            ),
            Field::new(
                "struct",
                DataType::Struct(vec![Field::new("int", DataType::Int32, false)].into()),
                false,
            ),
            Field::new("str", DataType::Utf8, true),
        ]);

        let num_rows = 9;
        let values: Float32Array = (0..num_rows).flat_map(|i| vec![i as f32; 3]).collect();
        let vector = FixedSizeListArray::try_new_from_values(values, 3).unwrap();

        let struct_int = Int32Array::from_iter_values(0..num_rows);
        let struct_array = StructArray::new(
            vec![Field::new("int", DataType::Int32, false)].into(),
            vec![Arc::new(struct_int) as ArrayRef],
            None,
        );

        let strings = StringArray::from(vec![
            Some("a"),
            None,
            Some("b"),
            Some("c"),
            None,
            Some("d"),
            Some("e"),
            None,
            Some("g"),
        ]);

        RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![
                Arc::new(vector) as ArrayRef,
                Arc::new(struct_array) as ArrayRef,
                Arc::new(strings) as ArrayRef,
            ],
        )
        .unwrap()
    }

    async fn test_dataset() -> Dataset {
        let data = test_data();
        let schema = data.schema().clone();
        let reader = RecordBatchIterator::new(vec![Ok(data)], schema);
        let params = WriteParams {
            max_rows_per_group: 3,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, "memory://test", Some(params))
            .await
            .unwrap();

        dataset.delete("struct.int = 6").await.unwrap();

        dataset
    }

    #[tokio::test]
    async fn test_codepath_options() {
        struct TestCase {
            with_row_id: bool,
            make_deletions_null: bool,
            ordered_output: bool,
            use_stats: bool,
        }

        // This is a little galaxy-brained, but at least it's concise.
        let cases = (0..16).map(|i| TestCase {
            with_row_id: i & 1 != 0 || i & 2 != 0, // If make_deletions_null is true, with_row_id is required
            make_deletions_null: i & 2 != 0,
            ordered_output: i & 4 != 0,
            use_stats: i & 8 != 0,
        });

        let dataset = Arc::new(test_dataset().await);

        // TODO: it seems like nested column references work by accident. We
        // can move the string to a flat column.
        let predicate = col("struct")
            .field("int")
            .gt(lit(4))
            .and(col(Column::from_name("str")).is_not_null());

        for case in cases {
            let config = ScanConfig {
                with_row_id: case.with_row_id,
                make_deletions_null: case.make_deletions_null,
                ordered_output: case.ordered_output,
                use_stats: case.use_stats,
                ..Default::default()
            };
            let scan = LancePushdownScanExec::try_new(
                dataset.clone(),
                dataset.fragments().clone(),
                Arc::new(dataset.schema().clone()),
                predicate.clone(),
                config,
            )
            .unwrap();

            let ctx = SessionContext::new();
            let batches = scan
                .execute(0, ctx.task_ctx())
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let mut result = concat_batches(&batches[0].schema(), &batches).unwrap();

            let mut expected_schema = ArrowSchema::from(dataset.schema());
            if case.with_row_id {
                expected_schema = expected_schema
                    .try_with_column_at(0, ROW_ID_FIELD.clone())
                    .unwrap();
            }
            assert_eq!(result.schema().as_ref(), &expected_schema);

            if !case.ordered_output {
                let indices = sort_to_indices(
                    result.column_by_qualified_name("struct.int").unwrap(),
                    Default::default(),
                    None,
                )
                .unwrap();
                result = result.take(&indices).unwrap();
            }

            if case.make_deletions_null {
                let predicate = col(ROW_ID).is_not_null();
                let planner = Planner::new(result.schema());
                let physical_expr = planner.create_physical_expr(&predicate).unwrap();
                let mask = physical_expr
                    .evaluate(&result)
                    .unwrap()
                    .into_array(result.num_rows());
                result = filter_record_batch(&result, mask.as_boolean()).unwrap();
            }

            assert_eq!(
                result
                    .column_by_qualified_name("struct.int")
                    .unwrap()
                    .as_primitive::<Int32Type>(),
                &Int32Array::from(vec![5, 8])
            );
        }
    }

    async fn pushdown_scan(
        ctx: &SessionContext,
        dataset: Arc<Dataset>,
        projection_indices: Vec<i32>,
        predicate: Expr,
        scan_config: ScanConfig,
    ) -> Result<RecordBatch> {
        let scan = LancePushdownScanExec::try_new(
            dataset.clone(),
            dataset.fragments().clone(),
            Arc::new(dataset.schema().clone().project_by_ids(&projection_indices)),
            predicate,
            scan_config,
        )
        .unwrap();
        let result: Vec<RecordBatch> = scan.execute(0, ctx.task_ctx())?.try_collect().await?;
        Ok(concat_batches(&result[0].schema(), &result)?)
    }

    #[derive(Debug)]
    struct PredicateCombinationsTest {
        projection_indices: Vec<i32>,
        predicate: Expr,
        data: RecordBatch,
    }

    impl PredicateCombinationsTest {
        fn new(projection_indices: Vec<i32>, predicate: Expr) -> Self {
            Self {
                projection_indices,
                predicate,
                data: Self::data(),
            }
        }

        /// Generate the test data
        fn data() -> RecordBatch {
            // Properties for test data:
            // * Want a good mix of data types
            // * Want multiple batches (will do 3 rows per batch when writing dataset)
            let schema = Arc::new(ArrowSchema::new(vec![
                Field::new("int", DataType::Int32, false),
                Field::new("float", DataType::Float32, false),
                Field::new("str", DataType::Utf8, true),
                Field::new(
                    "timestamp",
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    false,
                ),
                Field::new(
                    "str_dict",
                    DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                    true,
                ),
            ]));

            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 1, 3, 6])),
                    Arc::new(Float32Array::from(vec![
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        std::f32::NAN,
                        3.0,
                        6.0,
                    ])),
                    Arc::new(StringArray::from(vec![
                        Some("a"),
                        None,
                        Some("b"),
                        Some("c"),
                        None,
                        Some("d"),
                        Some("e"),
                        Some("f"),
                        Some("g"),
                    ])),
                    Arc::new(TimestampMicrosecondArray::from(vec![
                        1, 2, 3, 4, 5, 6, 1, 3, 6,
                    ])),
                    Arc::new(DictionaryArray::new(
                        // "a", null, "b", "a", "a", "a", null, null, "b"
                        Int32Array::from(vec![0, 1, 2, 0, 0, 0, 1, 1, 2]),
                        Arc::new(StringArray::from(vec![Some("a"), None, Some("b")])),
                    )),
                ],
            )
            .unwrap()
        }

        async fn dataset(uri: &str, data: RecordBatch) -> Dataset {
            let schema = data.schema();
            let reader = RecordBatchIterator::new(vec![Ok(data)], schema.clone());

            let write_params = WriteParams {
                max_rows_per_group: 3,
                ..Default::default()
            };
            Dataset::write(reader, uri, Some(write_params))
                .await
                .unwrap()
        }

        /// Generate the expected result using DataFusion executed on the test data.
        async fn expected_result(&self) -> RecordBatch {
            let ctx = SessionContext::new();
            let schema = self.data.schema();
            let schema_names = schema.field_names();
            let columns = self
                .projection_indices
                .iter()
                .map(|i| schema_names[*i as usize].as_str())
                .collect::<Vec<&str>>();
            let res = ctx
                .read_batch(self.data.clone())
                .unwrap()
                .filter(self.predicate.clone())
                .unwrap()
                .select_columns(&columns)
                .unwrap()
                .collect()
                .await
                .unwrap();
            concat_batches(&res[0].schema(), &res).unwrap()
        }

        async fn run_test(&self) {
            let test_dir = tempdir().unwrap();
            let test_uri = test_dir.path().to_str().unwrap();

            let dataset = Arc::new(Self::dataset(test_uri, self.data.clone()).await);
            let expected = self.expected_result().await;

            let ctx = SessionContext::new();

            let result = pushdown_scan(
                &ctx,
                dataset.clone(),
                self.projection_indices.clone(),
                self.predicate.clone(),
                Default::default(),
            )
            .await
            .unwrap();
            assert_eq!(
                &expected, &result,
                "Failed with use_stats = true: {:#?}",
                self
            );

            // Also validate with use_stats = false
            let result = pushdown_scan(
                &ctx,
                dataset.clone(),
                self.projection_indices.clone(),
                self.predicate.clone(),
                ScanConfig {
                    use_stats: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            assert_eq!(
                &expected, &result,
                "Failed with use_stats = false: {:#?}",
                self
            );
        }
    }

    #[tokio::test]
    async fn test_predicate_combinations() {
        struct TestCase {
            projection_indices: Vec<i32>,
            predicate: Expr,
        }

        let cases = [
            TestCase {
                projection_indices: vec![0, 1, 2, 3, 4],
                predicate: col("int").eq(lit(1)),
            },
            TestCase {
                projection_indices: vec![1, 4],
                predicate: col("int").lt_eq(lit(3)),
            },
            TestCase {
                projection_indices: vec![1],
                predicate: col("float").gt(lit(2.0_f32)),
            },
            TestCase {
                projection_indices: vec![0, 1, 2, 3, 4],
                predicate: col("str").in_list(vec![lit("a"), lit("g")], false),
            },
            TestCase {
                projection_indices: vec![0, 1, 4],
                predicate: col("str").in_list(vec![lit("a"), lit("g")], true),
            },
            TestCase {
                projection_indices: vec![0, 1, 4],
                predicate: col("str").is_null().or(col("int").eq(lit(1))),
            },
            TestCase {
                projection_indices: vec![0, 1, 3],
                predicate: col("timestamp")
                    .lt(lit(ScalarValue::TimestampMicrosecond(Some(3), None))),
            },
            // TODO: I think there's something wrong with how we handle nulls with dictionaries.
            // TestCase {
            //     projection_indices: vec![0, 1, 4],
            //     predicate: col("str_dict").eq(lit("b")).or(col("str_dict").is_null()),
            // },
        ];

        for case in cases {
            let test = PredicateCombinationsTest::new(case.projection_indices, case.predicate);
            test.run_test().await;
        }
    }

    #[tokio::test]
    async fn test_retrieve_just_row_id() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "int",
            DataType::Int32,
            false,
        )]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..50))],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let write_params = WriteParams {
            max_rows_per_group: 10,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        dataset.delete("int >= 10 AND int < 25").await.unwrap();

        let dataset = Arc::new(dataset.clone());
        let ctx = SessionContext::new();

        let result = pushdown_scan(
            &ctx,
            dataset.clone(),
            vec![0],
            col("int").lt(lit(40)),
            ScanConfig {
                with_row_id: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let row_ids = result[ROW_ID].as_primitive::<UInt64Type>();
        assert_eq!(
            row_ids,
            &UInt64Array::from_iter_values((0..10).chain(25..40))
        );
    }
}
