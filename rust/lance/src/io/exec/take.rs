// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use arrow::array::AsArray;
use arrow::compute::{concat_batches, TakeOptions};
use arrow::datatypes::UInt64Type;
use arrow_array::{Array, UInt32Array};
use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use datafusion::common::Statistics;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricValue, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::stream::{FuturesOrdered, Stream, StreamExt, TryStreamExt};
use futures::FutureExt;
use lance_arrow::RecordBatchExt;
use lance_core::datatypes::{Field, OnMissing, Projection};
use lance_core::error::{DataFusionResult, LanceOptionExt};
use lance_core::utils::address::RowAddress;
use lance_core::utils::futures::FinallyStreamExt;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{ROW_ADDR, ROW_ID};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

use crate::dataset::fragment::{FragReadConfig, FragmentReader};
use crate::dataset::rowids::get_row_id_index;
use crate::dataset::Dataset;
use crate::datatypes::Schema;

use super::utils::IoMetrics;

#[derive(Debug, Clone)]
struct TakeStreamMetrics {
    baseline_metrics: BaselineMetrics,
    batches_processed: Count,
    io_metrics: IoMetrics,
}

impl TakeStreamMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let batches_processed = Count::new();
        MetricBuilder::new(metrics)
            .with_partition(partition)
            .build(MetricValue::Count {
                name: Cow::Borrowed("batches_processed"),
                count: batches_processed.clone(),
            });
        Self {
            baseline_metrics: BaselineMetrics::new(metrics, partition),
            batches_processed,
            io_metrics: IoMetrics::new(metrics, partition),
        }
    }
}

struct TakeStream {
    /// The dataset to take from
    dataset: Arc<Dataset>,
    /// The fields to take from the input stream
    fields_to_take: Arc<Schema>,
    /// The output schema, needed for us to merge the new columns
    /// into the input data in the correct order
    output_schema: SchemaRef,
    /// A cache of opened file readers
    ///
    /// This is a map from fragment id to a reader.
    readers_cache: Arc<Mutex<HashMap<u32, Arc<FragmentReader>>>>,
    /// The scan scheduler to use for reading fragments
    scan_scheduler: Arc<ScanScheduler>,
    /// The metrics for the stream
    metrics: TakeStreamMetrics,
}

impl TakeStream {
    fn new(
        dataset: Arc<Dataset>,
        fields_to_take: Arc<Schema>,
        output_schema: SchemaRef,
        scan_scheduler: Arc<ScanScheduler>,
        metrics: &ExecutionPlanMetricsSet,
        partition: usize,
    ) -> Self {
        Self {
            dataset,
            fields_to_take,
            output_schema,
            readers_cache: Arc::new(Mutex::new(HashMap::new())),
            scan_scheduler,
            metrics: TakeStreamMetrics::new(metrics, partition),
        }
    }

    async fn do_open_reader(&self, fragment_id: u32) -> DataFusionResult<Arc<FragmentReader>> {
        let fragment = self
        .dataset
        .get_fragment(fragment_id as usize)
        .ok_or_else(|| {
            DataFusionError::Execution(format!("The input to a take operation specified fragment id {} but this fragment does not exist in the dataset", fragment_id))
        })?;

        let reader = Arc::new(
            fragment
                .open(
                    &self.fields_to_take,
                    FragReadConfig::default().with_scan_scheduler(self.scan_scheduler.clone()),
                )
                .await?,
        );

        let mut readers = self.readers_cache.lock().unwrap();
        readers.insert(fragment_id, reader.clone());
        Ok(reader)
    }

    async fn open_reader(&self, fragment_id: u32) -> DataFusionResult<Arc<FragmentReader>> {
        if let Some(reader) = self
            .readers_cache
            .lock()
            .unwrap()
            .get(&fragment_id)
            .cloned()
        {
            return Ok(reader);
        }

        self.do_open_reader(fragment_id).await
    }

    async fn get_row_addrs(&self, batch: &RecordBatch) -> Result<Arc<dyn Array>> {
        if let Some(row_addr_array) = batch.column_by_name(ROW_ADDR) {
            Ok(row_addr_array.clone())
        } else {
            let row_id_array = batch.column_by_name(ROW_ID).expect_ok()?;

            if let Some(row_id_index) = get_row_id_index(&self.dataset).await? {
                let row_id_array = row_id_array.as_primitive::<UInt64Type>();
                let addresses = row_id_array
                    .values()
                    .iter()
                    .filter_map(|id| row_id_index.get(*id).map(|address| address.into()))
                    .collect::<Vec<u64>>();
                Ok(Arc::new(UInt64Array::from(addresses)))
            } else {
                Ok(row_id_array.clone())
            }
        }
    }

    async fn map_batch(
        self: Arc<Self>,
        batch: RecordBatch,
        batch_number: u32,
    ) -> DataFusionResult<RecordBatch> {
        let compute_timer = self.metrics.baseline_metrics.elapsed_compute().timer();
        let row_addrs_arr = self.get_row_addrs(&batch).await?;
        let row_addrs = row_addrs_arr.as_primitive::<UInt64Type>();

        // Check if the row addresses are already sorted to avoid unnecessary reorders
        let is_sorted = row_addrs.values().windows(2).all(|w| w[0] <= w[1]);

        let sorted_addrs: Arc<dyn Array>;
        let (sorted_addrs, permutation) = if is_sorted {
            (row_addrs, None)
        } else {
            let permutation = arrow::compute::sort_to_indices(&row_addrs_arr, None, None).unwrap();
            sorted_addrs = arrow::compute::take(
                &row_addrs_arr,
                &permutation,
                Some(TakeOptions {
                    check_bounds: false,
                }),
            )
            .unwrap();
            // Calculate the inverse permutation to restore the original order
            let mut inverse_permutation = vec![0; permutation.len()];
            for (i, p) in permutation.values().iter().enumerate() {
                inverse_permutation[*p as usize] = i as u32;
            }
            (
                sorted_addrs.as_primitive::<UInt64Type>(),
                Some(UInt32Array::from(inverse_permutation)),
            )
        };

        let mut futures = FuturesOrdered::new();
        let mut current_offsets = Vec::new();
        let mut current_fragment_id = None;

        for row_addr in sorted_addrs.values() {
            let addr = RowAddress::new_from_u64(*row_addr);

            if Some(addr.fragment_id()) != current_fragment_id {
                // Start a new group
                if let Some(fragment_id) = current_fragment_id {
                    let reader = self.open_reader(fragment_id).await?;
                    let offsets = std::mem::take(&mut current_offsets);
                    futures.push_back(
                        async move { reader.take_as_batch(&offsets, Some(batch_number)).await }
                            .boxed(),
                    );
                }
                current_fragment_id = Some(addr.fragment_id());
            }

            current_offsets.push(addr.row_offset());
        }

        // Handle the last group
        if let Some(fragment_id) = current_fragment_id {
            let reader = self.open_reader(fragment_id).await?;
            futures.push_back(
                async move {
                    reader
                        .take_as_batch(&current_offsets, Some(batch_number))
                        .await
                }
                .boxed(),
            );
        }

        // Stop the compute timer, don't count I/O time
        drop(compute_timer);

        let batches = futures.try_collect::<Vec<_>>().await?;

        if batches.is_empty() {
            return Ok(RecordBatch::new_empty(self.output_schema.clone()));
        }

        let _compute_timer = self.metrics.baseline_metrics.elapsed_compute().timer();
        let schema = batches.first().expect_ok()?.schema();
        let mut new_data = concat_batches(&schema, batches.iter())?;

        // Restore previous order (if addresses were out of order originally)
        if let Some(permutation) = permutation {
            new_data = arrow_select::take::take_record_batch(&new_data, &permutation).unwrap();
        }

        self.metrics
            .baseline_metrics
            .record_output(new_data.num_rows());
        self.metrics.batches_processed.add(1);
        Ok(batch.merge_with_schema(&new_data, self.output_schema.as_ref())?)
    }

    fn apply<S: Stream<Item = Result<RecordBatch>> + Send + 'static>(
        self: Arc<Self>,
        input: S,
    ) -> impl Stream<Item = Result<RecordBatch>> {
        let scan_scheduler = self.scan_scheduler.clone();
        let metrics = self.metrics.clone();
        let batches = input
            .enumerate()
            .map(move |(batch_index, batch)| {
                let batch = batch?;
                let this = self.clone();
                Ok(
                    tokio::task::spawn(this.map_batch(batch, batch_index as u32))
                        .map(|res| res.unwrap()),
                )
            })
            .boxed();
        batches
            .try_buffered(get_num_compute_intensive_cpus())
            .finally(move || {
                metrics.io_metrics.record_final(scan_scheduler.as_ref());
            })
    }
}

#[derive(Debug)]
pub struct TakeExec {
    // The dataset to take from
    dataset: Arc<Dataset>,
    // The desired output projection of the relation (input schema + take schema)
    //
    // This is used to re-calculate output_projection and extra_schema when
    // with_new_children is called.
    output_projection: Projection,
    // The schema of the extra columns to take from the dataset
    schema_to_take: Arc<Schema>,
    // The schema of the output
    output_schema: SchemaRef,
    input: Arc<dyn ExecutionPlan>,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for TakeExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let extra_fields = self
            .schema_to_take
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect::<HashSet<_>>();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let columns = self
                    .output_schema
                    .fields
                    .iter()
                    .map(|f| {
                        let name = f.name();
                        if extra_fields.contains(name) {
                            format!("({})", name)
                        } else {
                            name.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "Take: columns={:?}", columns)
            }
        }
    }
}

impl TakeExec {
    /// Create a [`TakeExec`] node.
    ///
    /// - dataset: the dataset to read from
    /// - input: the upstream [`ExecutionPlan`] to feed data in.
    /// - projection: the desired output projection, can overlap with the input schema if desired
    ///
    /// Returns None if no extra columns are required (everything in the projection exists in the input schema).
    pub fn try_new(
        dataset: Arc<Dataset>,
        input: Arc<dyn ExecutionPlan>,
        projection: Projection,
    ) -> Result<Option<Self>> {
        let original_projection = projection.clone();
        let projection =
            projection.subtract_arrow_schema(input.schema().as_ref(), OnMissing::Ignore)?;
        if projection.is_empty() {
            return Ok(None);
        }

        // We actually need a take so lets make sure we have a row id
        if input.schema().column_with_name(ROW_ADDR).is_none()
            && input.schema().column_with_name(ROW_ID).is_none()
        {
            return Err(DataFusionError::Plan(format!(
                "TakeExec requires the input plan to have a column named '{}' or '{}'",
                ROW_ADDR, ROW_ID
            )));
        }

        // Can't use take if we don't want any fields and we can't use take to add row_id or row_addr
        assert!(
            !projection.with_row_id && !projection.with_row_addr,
            "Take should not be used to insert row_id / row_addr: {:#?}",
            projection
        );

        let output_schema = Arc::new(Self::calculate_output_schema(
            dataset.schema(),
            &input.schema(),
            &projection,
        ));
        let output_arrow = Arc::new(ArrowSchema::from(output_schema.as_ref()));
        let properties = input
            .properties()
            .clone()
            .with_eq_properties(EquivalenceProperties::new(output_arrow.clone()));

        Ok(Some(Self {
            dataset,
            output_projection: original_projection,
            schema_to_take: projection.into_schema_ref(),
            input,
            output_schema: output_arrow,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    /// The output of a take operation will be all columns from the input schema followed
    /// by any new columns from the dataset.
    ///
    /// The output fields will always be added in dataset schema order
    ///
    /// Nested columns in the input schema may have new fields inserted into them.
    ///
    /// If this happens the order of the new nested fields will match the order defined in
    /// the dataset schema.
    fn calculate_output_schema(
        dataset_schema: &Schema,
        input_schema: &ArrowSchema,
        projection: &Projection,
    ) -> Schema {
        // TakeExec doesn't reorder top-level fields and so the first thing we need to do is determine the
        // top-level field order.
        let mut top_level_fields_added = HashSet::with_capacity(input_schema.fields.len());
        let projected_schema = projection.to_schema();

        let mut output_fields =
            Vec::with_capacity(input_schema.fields.len() + projected_schema.fields.len());
        // TakeExec always moves the _rowid to the start of the output schema
        output_fields.extend(input_schema.fields.iter().map(|f| {
            let f = Field::try_from(f.as_ref()).unwrap();
            if let Some(ds_field) = dataset_schema.field(&f.name) {
                top_level_fields_added.insert(ds_field.id);
                // Field is in the dataset, it might have new fields added to it
                if let Some(projected_field) = ds_field.apply_projection(projection) {
                    f.merge_with_reference(&projected_field, ds_field)
                } else {
                    // No new fields added, keep as-is
                    f
                }
            } else {
                // Field not in dataset, not possible to add extra fields, use as-is
                f
            }
        }));

        // Now we add to the end any brand new top-level fields.  These will be added
        // dataset schema order.
        output_fields.extend(
            projected_schema
                .fields
                .into_iter()
                .filter(|f| !top_level_fields_added.contains(&f.id)),
        );
        Schema {
            fields: output_fields,
            metadata: dataset_schema.metadata.clone(),
        }
    }

    /// Get the dataset.
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }
}

impl ExecutionPlan for TakeExec {
    fn name(&self) -> &str {
        "TakeExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // This is an I/O bound operation and wouldn't really benefit from partitioning
        //
        // Plus, if we did that, we would be creating multiple schedulers which could use
        // a lot of RAM.
        vec![false]
    }

    /// This preserves the output schema.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "TakeExec wrong number of children".to_string(),
            ));
        }

        let projection = self.output_projection.clone();

        let plan = Self::try_new(self.dataset.clone(), children[0].clone(), projection)?;

        if let Some(plan) = plan {
            Ok(Arc::new(plan))
        } else {
            // Is this legal or do we need to insert a no-op node?
            Ok(children[0].clone())
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let dataset = self.dataset.clone();
        let schema_to_take = self.schema_to_take.clone();
        let output_schema = self.output_schema.clone();
        let metrics = self.metrics.clone();

        // ScanScheduler::new launches the I/O scheduler in the background.
        // We aren't allowed to do work in `execute` and so we defer creation of the
        // TakeStream until the stream is polled.
        let lazy_take_stream = futures::stream::once(async move {
            let obj_store = dataset.object_store.clone();
            let scheduler_config = SchedulerConfig::max_bandwidth(&obj_store);
            let scan_scheduler = ScanScheduler::new(obj_store, scheduler_config);

            let take_stream = Arc::new(TakeStream::new(
                dataset,
                schema_to_take,
                output_schema,
                scan_scheduler,
                &metrics,
                partition,
            ));
            take_stream.apply(input_stream)
        });
        let output_schema = self.output_schema.clone();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            lazy_take_stream.flatten(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<datafusion::physical_plan::Statistics> {
        Ok(Statistics {
            num_rows: self.input.statistics()?.num_rows,
            ..Statistics::new_unknown(self.schema().as_ref())
        })
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{
        ArrayRef, Float32Array, Int32Array, RecordBatchIterator, StringArray, StructArray,
    };
    use arrow_schema::{DataType, Field, Fields};
    use datafusion::execution::TaskContext;
    use lance_arrow::SchemaExt;
    use lance_core::{datatypes::OnMissing, ROW_ID};
    use lance_datafusion::{datagen::DatafusionDatagenExt, exec::OneShotExec, utils::MetricsExt};
    use lance_datagen::{BatchCount, RowCount};
    use rstest::rstest;
    use tempfile::{tempdir, TempDir};

    use crate::{
        dataset::WriteParams,
        io::exec::{LanceScanConfig, LanceScanExec},
        utils::test::NoContextTestFixture,
    };

    struct TestFixture {
        dataset: Arc<Dataset>,
        _tmp_dir_guard: TempDir,
    }

    async fn test_fixture() -> TestFixture {
        let struct_fields = Fields::from(vec![
            Arc::new(Field::new("x", DataType::Int32, false)),
            Arc::new(Field::new("y", DataType::Int32, false)),
        ]);

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("f", DataType::Float32, false),
            Field::new("s", DataType::Utf8, false),
            Field::new("struct", DataType::Struct(struct_fields.clone()), false),
        ]));

        // Write 3 batches.
        let expected_batches: Vec<RecordBatch> = (0..3)
            .map(|batch_id| {
                let value_range = batch_id * 10..batch_id * 10 + 10;
                let columns: Vec<ArrayRef> = vec![
                    Arc::new(Int32Array::from_iter_values(value_range.clone())),
                    Arc::new(Float32Array::from_iter(
                        value_range.clone().map(|v| v as f32),
                    )),
                    Arc::new(StringArray::from_iter_values(
                        value_range.clone().map(|v| format!("str-{v}")),
                    )),
                    Arc::new(StructArray::new(
                        struct_fields.clone(),
                        vec![
                            Arc::new(Int32Array::from_iter(value_range.clone())),
                            Arc::new(Int32Array::from_iter(value_range)),
                        ],
                        None,
                    )),
                ];
                RecordBatch::try_new(schema.clone(), columns).unwrap()
            })
            .collect();

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let params = WriteParams {
            max_rows_per_file: 10,
            ..Default::default()
        };
        let reader =
            RecordBatchIterator::new(expected_batches.clone().into_iter().map(Ok), schema.clone());
        Dataset::write(reader, test_uri, Some(params))
            .await
            .unwrap();

        TestFixture {
            dataset: Arc::new(Dataset::open(test_uri).await.unwrap()),
            _tmp_dir_guard: test_dir,
        }
    }

    #[tokio::test]
    async fn test_take_schema() {
        let TestFixture { dataset, .. } = test_fixture().await;

        let scan_arrow_schema = ArrowSchema::new(vec![Field::new("i", DataType::Int32, false)]);
        let scan_schema = Arc::new(Schema::try_from(&scan_arrow_schema).unwrap());

        // With row id
        let config = LanceScanConfig {
            with_row_id: true,
            ..Default::default()
        };
        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            scan_schema,
            config,
        ));

        let projection = dataset
            .empty_projection()
            .union_column("s", OnMissing::Error)
            .unwrap();
        let take_exec = TakeExec::try_new(dataset, input, projection)
            .unwrap()
            .unwrap();
        let schema = take_exec.schema();
        assert_eq!(
            schema.fields.iter().map(|f| f.name()).collect::<Vec<_>>(),
            vec!["i", ROW_ID, "s"]
        );
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TakeInput {
        Ids,
        Addrs,
        IdsAndAddrs,
    }

    #[rstest]
    #[tokio::test]
    async fn test_simple_take(
        #[values(TakeInput::Ids, TakeInput::Addrs, TakeInput::IdsAndAddrs)] take_input: TakeInput,
    ) {
        let TestFixture {
            dataset,
            _tmp_dir_guard,
        } = test_fixture().await;

        let scan_schema = Arc::new(dataset.schema().project(&["i"]).unwrap());
        let config = LanceScanConfig {
            with_row_address: take_input != TakeInput::Ids,
            with_row_id: take_input != TakeInput::Addrs,
            ..Default::default()
        };
        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            scan_schema,
            config,
        ));

        let projection = dataset
            .empty_projection()
            .union_column("s", OnMissing::Error)
            .unwrap();
        let take_exec = TakeExec::try_new(dataset, input, projection)
            .unwrap()
            .unwrap();
        let schema = take_exec.schema();

        let mut expected_fields = vec!["i"];
        if take_input != TakeInput::Addrs {
            expected_fields.push(ROW_ID);
        }
        if take_input != TakeInput::Ids {
            expected_fields.push(ROW_ADDR);
        }
        expected_fields.push("s");
        assert_eq!(&schema.field_names(), &expected_fields);

        let mut stream = take_exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();

        while let Some(batch) = stream.try_next().await.unwrap() {
            assert_eq!(&batch.schema().field_names(), &expected_fields);
        }
    }

    #[tokio::test]
    async fn test_take_order() {
        let TestFixture {
            dataset,
            _tmp_dir_guard,
        } = test_fixture().await;

        // Grab all row addresses, shuffle them, and select the first 15 (half of the rows)
        let data = dataset
            .scan()
            .project(&["s"])
            .unwrap()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let indices = UInt64Array::from(vec![8, 13, 1, 7, 4, 5, 12, 9, 10, 2, 11, 6, 3, 0, 28]);
        let data = arrow_select::take::take_record_batch(&data, &indices).unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            ROW_ADDR,
            DataType::UInt64,
            true,
        )]));
        let row_addrs = data.project_by_schema(&schema).unwrap();

        // Split into 3 batches of 5
        let batches = (0..3)
            .map(|i| {
                let start = i * 5;
                row_addrs.slice(start, 5)
            })
            .collect::<Vec<_>>();

        let row_addr_stream = futures::stream::iter(batches.clone().into_iter().map(Ok));
        let row_addr_stream = Box::pin(RecordBatchStreamAdapter::new(schema, row_addr_stream));

        let input = Arc::new(OneShotExec::new(row_addr_stream));

        let projection = dataset
            .empty_projection()
            .union_column("s", OnMissing::Error)
            .unwrap();
        let take_exec = TakeExec::try_new(dataset, input, projection)
            .unwrap()
            .unwrap();

        let stream = take_exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();

        let expected = vec![data.slice(0, 5), data.slice(5, 5), data.slice(10, 5)];

        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(batches.len(), 3);
        for (batch, expected) in batches.into_iter().zip(expected) {
            assert_eq!(batch.schema().field_names(), vec![ROW_ADDR, "s"]);
            let expected = expected.project_by_schema(&batch.schema()).unwrap();
            assert_eq!(batch, expected);
        }

        let metrics = take_exec.metrics().unwrap();
        assert_eq!(metrics.output_rows(), Some(15));
        assert_eq!(metrics.find_count("batches_processed").unwrap().value(), 3);
    }

    #[tokio::test]
    async fn test_take_struct() {
        // When taking fields into an existing struct, the field order should be maintained
        // according the the schema of the struct.
        let TestFixture {
            dataset,
            _tmp_dir_guard,
        } = test_fixture().await;

        let scan_schema = Arc::new(dataset.schema().project(&["struct.y"]).unwrap());

        let config = LanceScanConfig {
            with_row_address: true,
            ..Default::default()
        };
        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            scan_schema,
            config,
        ));

        let projection = dataset
            .empty_projection()
            .union_column("struct.x", OnMissing::Error)
            .unwrap();

        let take_exec = TakeExec::try_new(dataset, input, projection)
            .unwrap()
            .unwrap();

        let expected_schema = ArrowSchema::new(vec![
            Field::new(
                "struct",
                DataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("x", DataType::Int32, false)),
                    Arc::new(Field::new("y", DataType::Int32, false)),
                ])),
                false,
            ),
            Field::new(ROW_ADDR, DataType::UInt64, true),
        ]);
        let schema = take_exec.schema();
        assert_eq!(schema.as_ref(), &expected_schema);

        let mut stream = take_exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();

        while let Some(batch) = stream.try_next().await.unwrap() {
            assert_eq!(batch.schema().as_ref(), &expected_schema);
        }
    }

    #[tokio::test]
    async fn test_take_no_row_addr() {
        let TestFixture { dataset, .. } = test_fixture().await;

        let scan_arrow_schema = ArrowSchema::new(vec![Field::new("i", DataType::Int32, false)]);
        let scan_schema = Arc::new(Schema::try_from(&scan_arrow_schema).unwrap());

        let projection = dataset
            .empty_projection()
            .union_column("s", OnMissing::Error)
            .unwrap();

        // No row address
        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            scan_schema,
            LanceScanConfig::default(),
        ));
        assert!(TakeExec::try_new(dataset, input, projection).is_err());
    }

    #[tokio::test]
    async fn test_with_new_children() -> Result<()> {
        let TestFixture { dataset, .. } = test_fixture().await;

        let config = LanceScanConfig {
            with_row_id: true,
            ..Default::default()
        };

        let input_schema = Arc::new(dataset.schema().project(&["i"])?);
        let projection = dataset
            .empty_projection()
            .union_column("s", OnMissing::Error)
            .unwrap();

        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            input_schema,
            config,
        ));

        assert_eq!(input.schema().field_names(), vec!["i", ROW_ID],);
        let take_exec = TakeExec::try_new(dataset.clone(), input.clone(), projection)?.unwrap();
        assert_eq!(take_exec.schema().field_names(), vec!["i", ROW_ID, "s"],);

        let projection = dataset
            .empty_projection()
            .union_columns(["s", "f"], OnMissing::Error)
            .unwrap();

        let outer_take =
            Arc::new(TakeExec::try_new(dataset, Arc::new(take_exec), projection)?.unwrap());
        assert_eq!(
            outer_take.schema().field_names(),
            vec!["i", ROW_ID, "s", "f"],
        );

        // with_new_children should preserve the output schema.
        let edited = outer_take.with_new_children(vec![input])?;
        assert_eq!(edited.schema().field_names(), vec!["i", ROW_ID, "f", "s"],);
        Ok(())
    }

    #[test]
    fn no_context_take() {
        // These tests ensure we can create nodes and call execute without a tokio Runtime
        // being active.  This is a requirement for proper implementation of a Datafusion foreign
        // table provider.
        let fixture = NoContextTestFixture::new();
        let arc_dasaset = Arc::new(fixture.dataset);

        let input = lance_datagen::gen()
            .col(ROW_ID, lance_datagen::array::step::<UInt64Type>())
            .into_df_exec(RowCount::from(50), BatchCount::from(2));

        let take = TakeExec::try_new(
            arc_dasaset.clone(),
            input,
            arc_dasaset
                .empty_projection()
                .union_column("text", OnMissing::Error)
                .unwrap(),
        )
        .unwrap()
        .unwrap();

        take.execute(0, Arc::new(TaskContext::default())).unwrap();
    }
}
