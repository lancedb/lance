// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The merge insert operation merges a batch of new data into an existing batch of old data.  This can be
//! used to implement a bulk update-or-insert (upsert) or find-or-create operation.  It can also be used to
//! replace a specified region of data with new data (e.g. replace the data for the month of January)
//!
//! The terminology for this operation can be slightly confusing.  We try and stick with the terminology from
//! SQL.  The "target table" is the OLD data that already exists.  The "source table" is the NEW data which is
//! being inserted into the dataset.
//!
//! In order for this operation to work we need to be able to match rows from the source table with rows in the
//! target table.  For example, given a row we need to know if this is a brand new row or matches an existing row.
//!
//! This match condition is currently limited to an key-match.  This means we consider a row to be a match if the
//! key columns are identical in both the source and the target.  This means that you will need some kind of
//! meaningful key column to be able to perform a merge insert.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use arrow_array::{
    cast::AsArray, types::UInt64Type, BooleanArray, RecordBatch, RecordBatchReader, StructArray,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::{
    execution::context::{SessionConfig, SessionContext},
    logical_expr::{Expr, JoinType},
    physical_plan::{
        joins::{HashJoinExec, PartitionMode},
        repartition::RepartitionExec,
        stream::RecordBatchStreamAdapter,
        union::UnionExec,
        ColumnarValue, ExecutionPlan, PhysicalExpr, SendableRecordBatchStream,
    },
    scalar::ScalarValue,
};

use datafusion_physical_expr::expressions::Column;
use futures::{
    stream::{self},
    Stream, StreamExt, TryStreamExt,
};
use lance_core::{
    datatypes::SchemaCompareOptions,
    error::{box_error, InvalidInputSnafu},
    utils::futures::Capacity,
    Error, Result,
};
use lance_datafusion::{
    exec::{execute_plan, LanceExecutionOptions, OneShotExec},
    utils::reader_to_stream,
};
use lance_index::DatasetIndexExt;
use lance_table::format::{Fragment, Index};
use log::info;
use roaring::RoaringTreemap;
use snafu::{location, Location, ResultExt};

use crate::{
    datafusion::dataframe::SessionContextExt,
    dataset::transaction::{Operation, Transaction},
    index::DatasetIndexInternalExt,
    io::{
        commit::commit_transaction,
        exec::{scalar_index::MapIndexExec, utils::ReplayExec, Planner, ProjectionExec, TakeExec},
    },
    Dataset,
};

use super::write_fragments_internal;

// "update if" expressions typically compare fields from the source table to the target table.
// These tables have the same schema and so filter expressions need to differentiate.  To do that
// we wrap the left side and the right side in a struct and make a single "combined schema"
fn combined_schema(schema: &Schema) -> Schema {
    let target = Field::new("target", DataType::Struct(schema.fields.clone()), false);
    let source = Field::new("source", DataType::Struct(schema.fields.clone()), false);
    Schema::new(vec![source, target])
}

// This takes a double-wide table (e.g. the result of the outer join below) and takes the left
// half, puts it into a struct, then takes the right half, and puts that into a struct.  This
// makes the table match the "combined schema" so we can apply an "update if" expression
fn unzip_batch(batch: &RecordBatch, schema: &Schema) -> RecordBatch {
    // The schema of the combined batches will be:
    // target_data_keys, target_data_non_keys, target_data_row_id, source_data_keys, source_data_non_keys
    // The keys and non_keys on both sides will be equal
    let num_fields = batch.num_columns();
    debug_assert_eq!(num_fields % 2, 1);
    let half_num_fields = num_fields / 2;
    let row_id_col = num_fields - 1;

    let source_arrays = batch.columns()[0..half_num_fields].to_vec();
    let source = StructArray::new(schema.fields.clone(), source_arrays, None);

    let target_arrays = batch.columns()[half_num_fields..row_id_col].to_vec();
    let target = StructArray::new(schema.fields.clone(), target_arrays, None);

    let combined_schema = combined_schema(schema);
    RecordBatch::try_new(
        Arc::new(combined_schema),
        vec![Arc::new(source), Arc::new(target)],
    )
    .unwrap()
}

/// Describes how rows should be handled when there is no matching row in the source table
///
/// These are old rows which do not match any new data
#[derive(Debug, Clone, PartialEq)]
pub enum WhenNotMatchedBySource {
    /// Do not delete rows from the target table
    ///
    /// This can be used for a find-or-create or an upsert operation
    Keep,
    /// Delete all rows from target table that don't match a row in the source table
    Delete,
    /// Delete rows from the target table if there is no match AND the expression evaluates to true
    ///
    /// This can be used to replace a region of data with new data
    DeleteIf(Expr),
}

impl WhenNotMatchedBySource {
    /// Create an instance of WhenNotMatchedBySource::DeleteIf from
    /// an SQL filter string
    ///
    /// This will parse the filter string (using the schema of the provided
    /// dataset) and simplify the resulting expression
    pub fn delete_if(dataset: &Dataset, expr: &str) -> Result<Self> {
        let planner = Planner::new(Arc::new(dataset.schema().into()));
        let expr = planner
            .parse_filter(expr)
            .map_err(box_error)
            .context(InvalidInputSnafu)?;
        let expr = planner
            .optimize_expr(expr)
            .map_err(box_error)
            .context(InvalidInputSnafu)?;
        Ok(Self::DeleteIf(expr))
    }
}

/// Describes how rows should be handled when there is a match between the target table and source table
#[derive(Debug, Clone, PartialEq)]
pub enum WhenMatched {
    /// The row is deleted from the target table and a new row is inserted based on the source table
    ///
    /// This can be used to achieve upsert behavior
    UpdateAll,
    /// The row is kept unchanged
    ///
    /// This can be used to achieve find-or-create behavior
    DoNothing,
    /// The row is updated (similar to UpdateAll) only for rows where the expression evaluates to
    /// true
    UpdateIf(Expr),
}

impl WhenMatched {
    pub fn update_if(dataset: &Dataset, expr: &str) -> Result<Self> {
        let dataset_schema: Schema = dataset.schema().into();
        let combined_schema = combined_schema(&dataset_schema);
        let planner = Planner::new(Arc::new(combined_schema));
        let expr = planner
            .parse_filter(expr)
            .map_err(box_error)
            .context(InvalidInputSnafu)?;
        let expr = planner
            .optimize_expr(expr)
            .map_err(box_error)
            .context(InvalidInputSnafu)?;
        Ok(Self::UpdateIf(expr))
    }
}

/// Describes how rows should be handled when there is no matching row in the target table
///
/// These are new rows which do not match any old data
pub enum WhenNotMatched {
    /// The new row is inserted into the target table
    ///
    /// This is used in both find-or-create and upsert operations
    InsertAll,
    /// The new row is ignored
    DoNothing,
}

#[derive(Debug, Clone)]
struct MergeInsertParams {
    // The column(s) to join on
    on: Vec<String>,
    // If true, then update all columns of the old data to the new data when there is a match
    when_matched: WhenMatched,
    // If true, then insert all columns of the new data when there is no match in the old data
    insert_not_matched: bool,
    // Controls whether data that is not matched by the source is deleted or not
    delete_not_matched_by_source: WhenNotMatchedBySource,
}

/// A MergeInsertJob inserts new rows, deletes old rows, and updates existing rows all as
/// part of a single transaction.
pub struct MergeInsertJob {
    // The column to merge the new data into
    dataset: Arc<Dataset>,
    // The parameters controlling how to merge the two streams
    params: MergeInsertParams,
}

/// Build a merge insert operation.
///
/// This operation is similar to SQL's MERGE statement. It allows you to merge
/// new data with existing data.
///
/// Use the [MergeInsertBuilder] to construct an merge insert job. For example:
///
/// ```ignore
/// // find-or-create, insert new rows only
/// let builder = MergeInsertBuilder::new(dataset, vec!["my_key"]);
/// let dataset = builder
///     .build()?
///     .execute(new_data)
///     .await?;
///
/// // upsert, insert or update
/// let builder = MergeInsertBuilder::new(dataset, vec!["my_key"]);
/// let dataset = builder
///     .when_not_matched(WhenNotMatched::UpdateAll)
///     .build()?
///     .execute(new_data)
///     .await?;
///
/// // replace data for month=january
/// let builder = MergeInsertBuilder::new(dataset, vec!["my_key"]);
/// let dataset = builder
///     .when_not_matched(WhenNotMatched::UpdateAll)
///     .when_not_matched_by_source(
///         WhenNotMatchedBySource::DeleteIf(month_eq_jan)
///     )
///     .build()?
///     .execute(new_data)
///     .await?;
/// ```
///
#[derive(Debug, Clone)]
pub struct MergeInsertBuilder {
    dataset: Arc<Dataset>,
    params: MergeInsertParams,
}

impl MergeInsertBuilder {
    /// Creates a new builder
    ///
    /// By default this will build a job that has the same semantics as find-or-create
    ///  - matching rows will be kept as-is
    ///  - new rows in the new data will be inserted
    ///  - rows in the old data that do not match will be left as-is
    ///
    /// Use the methods on this builder to customize that behavior
    pub fn try_new(dataset: Arc<Dataset>, on: Vec<String>) -> Result<Self> {
        if on.is_empty() {
            return Err(Error::invalid_input(
                "A merge insert operation must specify at least one on key",
                location!(),
            ));
        }
        Ok(Self {
            dataset,
            params: MergeInsertParams {
                on,
                when_matched: WhenMatched::DoNothing,
                insert_not_matched: true,
                delete_not_matched_by_source: WhenNotMatchedBySource::Keep,
            },
        })
    }

    /// Specify what should happen when a target row matches a row in the source
    pub fn when_matched(&mut self, behavior: WhenMatched) -> &mut Self {
        self.params.when_matched = behavior;
        self
    }

    /// Specify what should happen when a source row has no match in the target
    ///
    /// These are typically "new rows"
    pub fn when_not_matched(&mut self, behavior: WhenNotMatched) -> &mut Self {
        self.params.insert_not_matched = match behavior {
            WhenNotMatched::DoNothing => false,
            WhenNotMatched::InsertAll => true,
        };
        self
    }

    /// Specify what should happen when a target row has no match in the source
    ///
    /// These are typically "old rows"
    pub fn when_not_matched_by_source(&mut self, behavior: WhenNotMatchedBySource) -> &mut Self {
        self.params.delete_not_matched_by_source = behavior;
        self
    }

    /// Crate a merge insert job
    pub fn try_build(&mut self) -> Result<MergeInsertJob> {
        if !self.params.insert_not_matched
            && self.params.when_matched == WhenMatched::DoNothing
            && self.params.delete_not_matched_by_source == WhenNotMatchedBySource::Keep
        {
            return Err(Error::invalid_input(
                "The merge insert job is not configured to change the data in any way",
                location!(),
            ));
        }
        Ok(MergeInsertJob {
            dataset: self.dataset.clone(),
            params: self.params.clone(),
        })
    }
}

impl MergeInsertJob {
    pub async fn execute_reader(
        self,
        source: Box<dyn RecordBatchReader + Send>,
    ) -> Result<Arc<Dataset>> {
        let stream = reader_to_stream(source);
        self.execute(stream).await
    }

    fn check_compatible_schema(&self, schema: &Schema) -> Result<()> {
        let lance_schema: lance_core::datatypes::Schema = schema.try_into()?;
        lance_schema.check_compatible(
            self.dataset.schema(),
            &SchemaCompareOptions {
                compare_dictionary: true,
                ..Default::default()
            },
        )
    }

    async fn join_key_as_scalar_index(&self) -> Result<Option<Index>> {
        if self.params.on.len() != 1 {
            Ok(None)
        } else {
            let col = &self.params.on[0];
            self.dataset.load_scalar_index_for_column(col).await
        }
    }

    async fn create_indexed_scan_joined_stream(
        &self,
        source: SendableRecordBatchStream,
        index: Index,
    ) -> Result<SendableRecordBatchStream> {
        // This relies on a few non-standard physical operators and so we cannot use the
        // datafusion dataframe API and need to construct the plan manually :'(

        // 1 - Input from user
        let input = Arc::new(OneShotExec::new(source));

        // 2 - Fork/Replay the input
        // Regrettably, this needs to have unbounded capacity, and so we need to fully read
        // the new data into memory.  In the future, we can do better
        let shared_input = Arc::new(ReplayExec::new(Capacity::Unbounded, input));

        // 3 - Use the index to map input to row addresses
        // First, we need to project to the key column
        let schema = shared_input.schema();
        let field = schema.field_with_name(&self.params.on[0])?;
        let key_only_schema =
            lance_core::datatypes::Schema::try_from(&Schema::new(vec![field.clone()]))?;
        let index_mapper_input = Arc::new(ProjectionExec::try_new(
            shared_input.clone(),
            Arc::new(key_only_schema),
        )?);

        // Then we pass the key column into the index mapper
        let index_column = self.params.on[0].clone();
        let index_mapper = Arc::new(MapIndexExec::new(
            self.dataset.clone(),
            index_column.clone(),
            index_mapper_input,
        ));

        // 4 - Take the mapped row addresses
        let mut target = Arc::new(TakeExec::try_new(
            self.dataset.clone(),
            index_mapper,
            Arc::new(self.dataset.schema().clone()),
            num_cpus::get(),
        )?) as Arc<dyn ExecutionPlan>;

        // 5 - Take puts the row id at the beginning.  A full scan (used when there is no scalar
        //     index) puts the row id at the end.  We need to match these up so we reorder the row
        //     id to the end
        let schema = target.schema();
        let fields = schema.fields();
        let mut columns = fields[1..].to_vec();
        columns.push(fields[0].clone());
        let projected_schema = lance_core::datatypes::Schema::try_from(&Schema::new(columns))?;
        target = Arc::new(ProjectionExec::try_new(target, Arc::new(projected_schema))?);

        // 5a - We also need to scan any new unindexed data and union it in
        let unindexed_fragments = self.dataset.unindexed_fragments(&index.name).await?;
        if !unindexed_fragments.is_empty() {
            let unindexed_data = self
                .dataset
                .scan()
                .with_row_id()
                .with_fragments(unindexed_fragments)
                .create_plan()
                .await?;
            let unioned = UnionExec::new(vec![target, unindexed_data]);
            // Enforce only 1 partition.
            target = Arc::new(RepartitionExec::try_new(
                Arc::new(unioned),
                datafusion::physical_plan::Partitioning::RoundRobinBatch(1),
            )?);
        }

        // 6 - Finally, join the input (source table) with the taken data (target table)
        let source_key = Column::new_with_schema(&index_column, shared_input.schema().as_ref())?;
        let target_key = Column::new_with_schema(&index_column, target.schema().as_ref())?;
        let joined = Arc::new(
            HashJoinExec::try_new(
                shared_input,
                target,
                vec![(Arc::new(target_key), Arc::new(source_key))],
                None,
                &JoinType::Full,
                None,
                PartitionMode::CollectLeft,
                true,
            )
            .unwrap(),
        );
        execute_plan(
            joined,
            LanceExecutionOptions {
                use_spilling: true,
                ..Default::default()
            },
        )
    }

    // If the join keys are not indexed then we need to do a full scan of the table
    async fn create_full_table_joined_stream(
        &self,
        source: SendableRecordBatchStream,
    ) -> Result<SendableRecordBatchStream> {
        let session_config = SessionConfig::default().with_target_partitions(1);
        let session_ctx = SessionContext::new_with_config(session_config);
        let schema = source.schema();
        self.check_compatible_schema(&schema)?;
        let existing = session_ctx.read_lance(self.dataset.clone(), true)?;
        let new_data = session_ctx.read_one_shot(source)?;
        let join_cols = self
            .params
            .on
            .iter()
            .map(|c| c.as_str())
            .collect::<Vec<_>>();
        let joined = new_data.join(existing, JoinType::Full, &join_cols, &join_cols, None)?;
        Ok(joined.execute_stream().await?)
    }

    async fn create_joined_stream(
        &self,
        source: SendableRecordBatchStream,
    ) -> Result<SendableRecordBatchStream> {
        // We need to do a full index scan if we're deleting source data
        let can_use_scalar_index = matches!(
            self.params.delete_not_matched_by_source,
            WhenNotMatchedBySource::Keep
        );
        if can_use_scalar_index {
            if let Some(index) = self.join_key_as_scalar_index().await? {
                self.create_indexed_scan_joined_stream(source, index).await
            } else {
                self.create_full_table_joined_stream(source).await
            }
        } else {
            info!("The merge insert operation is configured to delete rows from the target table, this requires a potentially costly full table scan");
            self.create_full_table_joined_stream(source).await
        }
    }

    /// Executes the merge insert job
    ///
    /// This will take in the source, merge it with the existing target data, and insert new
    /// rows, update existing rows, and delete existing rows
    pub async fn execute(self, source: SendableRecordBatchStream) -> Result<Arc<Dataset>> {
        let schema = source.schema();

        let joined = self.create_joined_stream(source).await?;
        let merger = Merger::try_new(self.params, schema.clone())?;
        let deleted_rows = merger.deleted_rows.clone();
        let stream = joined
            .and_then(move |batch| merger.clone().execute_batch(batch))
            .try_flatten();
        let stream = RecordBatchStreamAdapter::new(schema, stream);

        let new_fragments = write_fragments_internal(
            None,
            self.dataset.object_store.clone(),
            &self.dataset.base,
            self.dataset.schema(),
            Box::pin(stream),
            Default::default(),
        )
        .await?;

        // Apply deletions
        let removed_row_ids = Arc::into_inner(deleted_rows).unwrap().into_inner().unwrap();
        let (old_fragments, removed_fragment_ids) =
            Self::apply_deletions(&self.dataset, &removed_row_ids).await?;

        // Commit updated and new fragments
        Self::commit(
            self.dataset,
            removed_fragment_ids,
            old_fragments,
            new_fragments,
        )
        .await
    }

    // Delete a batch of rows by id, returns the fragments modified and the fragments removed
    async fn apply_deletions(
        dataset: &Dataset,
        removed_row_ids: &RoaringTreemap,
    ) -> Result<(Vec<Fragment>, Vec<u64>)> {
        let bitmaps = Arc::new(removed_row_ids.bitmaps().collect::<BTreeMap<_, _>>());

        enum FragmentChange {
            Unchanged,
            Modified(Fragment),
            Removed(u64),
        }

        let mut updated_fragments = Vec::new();
        let mut removed_fragments = Vec::new();

        let mut stream = futures::stream::iter(dataset.get_fragments())
            .map(move |fragment| {
                let bitmaps_ref = bitmaps.clone();
                async move {
                    let fragment_id = fragment.id();
                    if let Some(bitmap) = bitmaps_ref.get(&(fragment_id as u32)) {
                        match fragment.extend_deletions(*bitmap).await {
                            Ok(Some(new_fragment)) => {
                                Ok(FragmentChange::Modified(new_fragment.metadata))
                            }
                            Ok(None) => Ok(FragmentChange::Removed(fragment_id as u64)),
                            Err(e) => Err(e),
                        }
                    } else {
                        Ok(FragmentChange::Unchanged)
                    }
                }
            })
            .buffer_unordered(num_cpus::get() * 4);

        while let Some(res) = stream.next().await.transpose()? {
            match res {
                FragmentChange::Unchanged => {}
                FragmentChange::Modified(fragment) => updated_fragments.push(fragment),
                FragmentChange::Removed(fragment_id) => removed_fragments.push(fragment_id),
            }
        }

        Ok((updated_fragments, removed_fragments))
    }

    // Commit the operation
    async fn commit(
        dataset: Arc<Dataset>,
        removed_fragment_ids: Vec<u64>,
        updated_fragments: Vec<Fragment>,
        new_fragments: Vec<Fragment>,
    ) -> Result<Arc<Dataset>> {
        let operation = Operation::Update {
            removed_fragment_ids,
            updated_fragments,
            new_fragments,
        };
        let transaction = Transaction::new(dataset.manifest.version, operation, None);

        let manifest = commit_transaction(
            dataset.as_ref(),
            dataset.object_store(),
            dataset.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
        )
        .await?;

        let mut dataset = dataset.as_ref().clone();
        dataset.manifest = Arc::new(manifest);

        Ok(Arc::new(dataset))
    }
}

// A sync-safe structure that is shared by all of the "process batch" tasks.
//
// Note: we are not currently using parallelism but this still needs to be sync because it is
//       held across an await boundary (and we might use parallelism someday)
#[derive(Debug, Clone)]
struct Merger {
    // As the merger runs it will update the list of deleted rows
    deleted_rows: Arc<Mutex<RoaringTreemap>>,
    // Physical delete expression, only set if params.delete_not_matched_by_source is DeleteIf
    delete_expr: Option<Arc<dyn PhysicalExpr>>,
    // Physical "when matched update if" expression, only set if params.when_matched is UpdateIf
    match_filter_expr: Option<Arc<dyn PhysicalExpr>>,
    // The parameters controlling the merge
    params: MergeInsertParams,
    // The schema of the dataset, used to recover nullability information
    schema: Arc<Schema>,
}

impl Merger {
    // Creates a new merger with an empty set of deleted rows, compiles expressions, if present
    fn try_new(params: MergeInsertParams, schema: Arc<Schema>) -> Result<Self> {
        let delete_expr = if let WhenNotMatchedBySource::DeleteIf(expr) =
            &params.delete_not_matched_by_source
        {
            let planner = Planner::new(schema.clone());
            let expr = planner.optimize_expr(expr.clone())?;
            let physical_expr = planner.create_physical_expr(&expr)?;
            let data_type = physical_expr.data_type(&schema)?;
            if data_type != DataType::Boolean {
                return Err(Error::invalid_input(format!("Merge insert conditions must be expressions that return a boolean value, received expression ({}) which has data type {}", expr, data_type), location!()));
            }
            Some(physical_expr)
        } else {
            None
        };
        let match_filter_expr = if let WhenMatched::UpdateIf(expr) = &params.when_matched {
            let combined_schema = Arc::new(combined_schema(&schema));
            let planner = Planner::new(combined_schema.clone());
            let expr = planner.optimize_expr(expr.clone())?;
            let match_expr = planner.create_physical_expr(&expr)?;
            let data_type = match_expr.data_type(combined_schema.as_ref())?;
            if data_type != DataType::Boolean {
                return Err(Error::invalid_input(format!("Merge insert conditions must be expressions that return a boolean value, received a 'when matched update if' expression ({}) which has data type {}", expr, data_type), location!()));
            }
            Some(match_expr)
        } else {
            None
        };
        Ok(Self {
            deleted_rows: Arc::new(Mutex::new(RoaringTreemap::new())),
            delete_expr,
            match_filter_expr,
            params,
            schema,
        })
    }

    // Retrieves a bitmap of rows where at least one of the columns in the range
    // col_offset..coll_offset+num_cols is not null.
    //
    fn not_all_null(
        batch: &RecordBatch,
        col_offset: usize,
        num_cols: usize,
    ) -> Result<BooleanArray> {
        // For our purposes we know there is always at least 1 on key
        debug_assert_ne!(num_cols, 0);
        let mut at_least_one_valid = arrow::compute::is_not_null(batch.column(col_offset))?;
        for idx in col_offset + 1..col_offset + num_cols {
            let is_valid = arrow::compute::is_not_null(batch.column(idx))?;
            at_least_one_valid = arrow::compute::or(&at_least_one_valid, &is_valid)?;
        }
        Ok(at_least_one_valid)
    }

    // Since we are performing an
    // outer join below we expect the results to look like:
    //
    // | LEFT KEYS | LEFT PAYLOAD | RIGHT KEYS | RIGHT PAYLOAD |
    // | NULL      | NULL         | NOT NULL   | ************* | <- when not matched
    // | ********* | ************ | ********** | ************* | <- when matched
    // | ********* | ************ | NULL       | NULL          | <- when not matched by source
    //
    // To test which case we are in we check to see if all of LEFT KEYS or RIGHT KEYS are null
    //
    // This returns three selection bitmaps
    //
    //  - The first is true for rows that are in the left side only
    //  - The second is true for rows in both the left and the right
    //  - The third is true for rows in the right side only
    fn extract_selections(
        &self,
        combined_batch: &RecordBatch,
        right_offset: usize,
        num_keys: usize,
    ) -> Result<(BooleanArray, BooleanArray, BooleanArray)> {
        let in_left = Self::not_all_null(combined_batch, 0, num_keys)?;
        let in_right = Self::not_all_null(combined_batch, right_offset, num_keys)?;
        let in_both = arrow::compute::and(&in_left, &in_right)?;
        let left_only = arrow::compute::and(&in_left, &arrow::compute::not(&in_right)?)?;
        let right_only = arrow::compute::and(&arrow::compute::not(&in_left)?, &in_right)?;
        Ok((left_only, in_both, right_only))
    }

    // Given a batch of outer join data, split it into three different batches
    //
    // Process each sub-batch according to the merge insert params
    //
    // Returns 0, 1, or 2 batches
    // Potentially updates (as a side-effect) the deleted rows vec
    async fn execute_batch(
        self,
        batch: RecordBatch,
    ) -> datafusion::common::Result<impl Stream<Item = datafusion::common::Result<RecordBatch>>>
    {
        let num_fields = batch.schema().fields.len();
        // The schema of the combined batches will be:
        // source_keys, source_payload, target_keys, target_payload, row_id
        // The keys and non_keys on both sides will be equal
        debug_assert_eq!(num_fields % 2, 1);
        let row_id_col = num_fields - 1;
        let right_offset = num_fields / 2;
        let num_keys = self.params.on.len();

        let left_cols = Vec::from_iter(0..right_offset);
        let right_cols_with_id = Vec::from_iter(right_offset..num_fields);

        let mut batches = Vec::with_capacity(2);
        let (left_only, in_both, right_only) =
            self.extract_selections(&batch, right_offset, num_keys)?;

        // There is no contention on this mutex.  We're only using it to bypass the rust
        // borrow checker (the stream needs to be `sync` since it crosses an await point)
        let mut deleted_row_ids = self.deleted_rows.lock().unwrap();

        if self.params.when_matched != WhenMatched::DoNothing {
            let mut matched = arrow::compute::filter_record_batch(&batch, &in_both)?;
            if let Some(match_filter) = self.match_filter_expr {
                let unzipped = unzip_batch(&matched, &self.schema);
                let filtered = match_filter.evaluate(&unzipped)?;
                match filtered {
                    ColumnarValue::Array(mask) => {
                        // Some rows matched, filter down and replace those rows
                        matched = arrow::compute::filter_record_batch(&matched, mask.as_boolean())?;
                    }
                    ColumnarValue::Scalar(scalar) => {
                        if let ScalarValue::Boolean(Some(true)) = scalar {
                            // All rows matched, go ahead and replace the whole batch
                        } else {
                            // Nothing matched, replace nothing
                            matched = RecordBatch::new_empty(matched.schema().clone());
                        }
                    }
                }
            }
            // If the filter eliminated all rows then its important we don't try and write
            // the batch at all.  Writing an empty batch currently panics
            if matched.num_rows() > 0 {
                let row_ids = matched.column(row_id_col).as_primitive::<UInt64Type>();
                deleted_row_ids.extend(row_ids.values());
                let matched = matched.project(&left_cols)?;
                // The payload columns of an outer join are always nullable.  We need to restore
                // non-nullable to columns that were originally non-nullable.  This should be safe
                // since the not_matched rows should all be valid on the right_cols
                //
                // Sadly we can't use with_schema because it doesn't let you toggle nullability
                let matched = RecordBatch::try_new(
                    self.schema.clone(),
                    Vec::from_iter(matched.columns().iter().cloned()),
                )?;
                batches.push(Ok(matched));
            }
        }
        if self.params.insert_not_matched {
            let not_matched = arrow::compute::filter_record_batch(&batch, &left_only)?;
            let not_matched = not_matched.project(&left_cols)?;
            // See comment above explaining this schema replacement
            let not_matched = RecordBatch::try_new(
                self.schema.clone(),
                Vec::from_iter(not_matched.columns().iter().cloned()),
            )?;
            batches.push(Ok(not_matched));
        }
        match self.params.delete_not_matched_by_source {
            WhenNotMatchedBySource::Delete => {
                let unmatched = arrow::compute::filter(batch.column(row_id_col), &right_only)?;
                let row_ids = unmatched.as_primitive::<UInt64Type>();
                deleted_row_ids.extend(row_ids.values());
            }
            WhenNotMatchedBySource::DeleteIf(_) => {
                let target_data = batch.project(&right_cols_with_id)?;
                let unmatched = arrow::compute::filter_record_batch(&target_data, &right_only)?;
                let row_id_col = unmatched.num_columns() - 1;
                let to_delete = self.delete_expr.unwrap().evaluate(&unmatched)?;
                match to_delete {
                    ColumnarValue::Array(mask) => {
                        let row_ids = arrow::compute::filter(
                            unmatched.column(row_id_col),
                            mask.as_boolean(),
                        )?;
                        let row_ids = row_ids.as_primitive::<UInt64Type>();
                        deleted_row_ids.extend(row_ids.values());
                    }
                    ColumnarValue::Scalar(scalar) => {
                        if let ScalarValue::Boolean(Some(true)) = scalar {
                            let row_ids = unmatched.column(row_id_col).as_primitive::<UInt64Type>();
                            deleted_row_ids.extend(row_ids.values());
                        }
                    }
                }
            }
            WhenNotMatchedBySource::Keep => {}
        }
        Ok(stream::iter(batches))
    }
}

#[cfg(test)]
mod tests {

    use arrow_array::{types::UInt32Type, RecordBatchIterator, StringArray, UInt32Array};
    use arrow_select::concat::concat_batches;
    use datafusion::common::Column;
    use lance_datagen::{array, BatchCount, RowCount, Seed};
    use lance_index::IndexType;
    use tempfile::tempdir;

    use crate::{
        dataset::{WriteMode, WriteParams},
        index::scalar::ScalarIndexParams,
    };

    use super::*;

    async fn check(
        new_data: RecordBatch,
        mut job: MergeInsertJob,
        keys_from_left: &[u32],
        keys_from_right: &[u32],
    ) {
        let mut dataset = (*job.dataset).clone();
        dataset.restore().await.unwrap();
        job.dataset = Arc::new(dataset);

        let schema = new_data.schema();
        let new_reader = Box::new(RecordBatchIterator::new([Ok(new_data)], schema.clone()));
        let new_stream = reader_to_stream(new_reader);

        let merged_dataset = job.execute(new_stream).await.unwrap();

        let batches = merged_dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let merged = concat_batches(&schema, &batches).unwrap();

        let keyvals = merged
            .column(0)
            .as_primitive::<UInt32Type>()
            .values()
            .iter()
            .zip(
                merged
                    .column(1)
                    .as_primitive::<UInt32Type>()
                    .values()
                    .iter(),
            );
        let mut left_keys = keyvals
            .clone()
            .filter(|(_, &val)| val == 1)
            .map(|(key, _)| key)
            .copied()
            .collect::<Vec<_>>();
        let mut right_keys = keyvals
            .clone()
            .filter(|(_, &val)| val == 2)
            .map(|(key, _)| key)
            .copied()
            .collect::<Vec<_>>();
        left_keys.sort();
        right_keys.sort();
        assert_eq!(left_keys, keys_from_left);
        assert_eq!(right_keys, keys_from_right);
    }

    #[tokio::test]
    async fn test_basic_merge() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::UInt32, false),
            Field::new("value", DataType::UInt32, false),
            Field::new("filterme", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![1, 2, 3, 4, 5, 6])),
                Arc::new(UInt32Array::from(vec![1, 1, 1, 1, 1, 1])),
                Arc::new(StringArray::from(vec!["A", "B", "A", "A", "B", "A"])),
            ],
        )
        .unwrap();

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let batches = RecordBatchIterator::new([Ok(batch)], schema.clone());
        let ds = Arc::new(Dataset::write(batches, test_uri, None).await.unwrap());

        let new_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![4, 5, 6, 7, 8, 9])),
                Arc::new(UInt32Array::from(vec![2, 2, 2, 2, 2, 2])),
                Arc::new(StringArray::from(vec!["A", "B", "C", "A", "B", "C"])),
            ],
        )
        .unwrap();

        // Quick test that no on-keys is not valid and fails
        assert!(MergeInsertBuilder::try_new(ds.clone(), vec![]).is_err());

        let keys = vec!["key".to_string()];
        // find-or-create, no delete
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1, 2, 3, 4, 5, 6], &[7, 8, 9]).await;

        // upsert, no delete
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1, 2, 3], &[4, 5, 6, 7, 8, 9]).await;

        // conditional upsert, no delete
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(
                WhenMatched::update_if(&ds, "source.filterme != target.filterme").unwrap(),
            )
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1, 2, 3, 4, 5], &[6, 7, 8, 9]).await;

        // conditional update, no matches
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_matched(WhenMatched::update_if(&ds, "target.filterme = 'z'").unwrap())
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1, 2, 3, 4, 5, 6], &[]).await;

        // update only, no delete (useful for bulk update)
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1, 2, 3], &[4, 5, 6]).await;

        // Conditional update
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(
                WhenMatched::update_if(&ds, "source.filterme == target.filterme").unwrap(),
            )
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1, 2, 3, 6], &[4, 5]).await;

        // No-op (will raise an error)
        assert!(MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .is_err());

        // find-or-create, with delete all
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[4, 5, 6], &[7, 8, 9]).await;

        // upsert, with delete all
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[], &[4, 5, 6, 7, 8, 9]).await;

        // update only, with delete all (unusual)
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[], &[4, 5, 6]).await;

        // just delete all (not real case, just use delete)
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[4, 5, 6], &[]).await;

        // For the "delete some" tests we use key > 1
        let condition = Expr::gt(
            Expr::Column(Column::new_unqualified("key")),
            Expr::Literal(ScalarValue::UInt32(Some(1))),
        );
        // find-or-create, with delete some
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition.clone()))
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1, 4, 5, 6], &[7, 8, 9]).await;

        // upsert, with delete some
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition.clone()))
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1], &[4, 5, 6, 7, 8, 9]).await;

        // update only, with delete some (unusual)
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition.clone()))
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1], &[4, 5, 6]).await;

        // just delete some (not real case, just use delete)
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition.clone()))
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1, 4, 5, 6], &[]).await;
    }

    #[tokio::test]
    async fn test_indexed_merge_insert() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = lance_datagen::gen()
            .with_seed(Seed::from(1))
            .col("value", array::step::<UInt32Type>())
            .col("key", array::rand_pseduo_uuid_hex());
        let data = data.into_reader_rows(RowCount::from(1024), BatchCount::from(32));
        let schema = data.schema();

        // Create an input dataset with a scalar index on key
        let mut ds = Dataset::write(data, test_uri, None).await.unwrap();
        let index_params = ScalarIndexParams::default();
        ds.create_index(&["key"], IndexType::Scalar, None, &index_params, false)
            .await
            .unwrap();

        // Create some new (unindexed) data
        let data = lance_datagen::gen()
            .with_seed(Seed::from(2))
            .col("value", array::step::<UInt32Type>())
            .col("key", array::rand_pseduo_uuid_hex());
        let data = data.into_reader_rows(RowCount::from(1024), BatchCount::from(8));
        let ds = Dataset::write(
            data,
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let ds = Arc::new(ds);

        let just_index_col = Schema::new(vec![Field::new("key", DataType::Utf8, false)]);

        // Sample 2048 random indices and then paste on a column of 9999999's
        let some_indices = ds
            .sample(2048, &(&just_index_col).try_into().unwrap())
            .await
            .unwrap();
        let some_indices = some_indices.column(0).clone();
        let some_vals = lance_datagen::gen()
            .anon_col(array::fill::<UInt32Type>(9999999))
            .into_batch_rows(RowCount::from(2048))
            .unwrap();
        let some_vals = some_vals.column(0).clone();
        let source_batch =
            RecordBatch::try_new(schema.clone(), vec![some_vals, some_indices]).unwrap();
        // To make things more interesting, lets make the input a stream of four batches
        let source_batches = vec![
            source_batch.slice(0, 512),
            source_batch.slice(512, 512),
            source_batch.slice(1024, 512),
            source_batch.slice(1536, 512),
        ];
        let source = Box::new(RecordBatchIterator::new(
            source_batches.clone().into_iter().map(Ok),
            schema.clone(),
        ));

        // Run merge_insert
        let ds = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
            .unwrap()
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_matched(WhenMatched::UpdateAll)
            .try_build()
            .unwrap()
            .execute_reader(source)
            .await
            .unwrap();

        // Check that the data is as expected
        let updated = ds
            .scan()
            .filter("value = 9999999")
            .unwrap()
            .count_rows()
            .await
            .unwrap();
        assert_eq!(updated, 2048);

        // Make sure we don't use an indexed scan if there is a delete criteria
        let source = Box::new(RecordBatchIterator::new(
            source_batches.clone().into_iter().map(Ok),
            schema.clone(),
        ));
        // Run merge_insert
        let ds = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
            .unwrap()
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
            .try_build()
            .unwrap()
            .execute_reader(source)
            .await
            .unwrap();

        // Check that the data is as expected
        assert_eq!(ds.count_rows(None).await.unwrap(), 2048);

        let source = Box::new(RecordBatchIterator::new(
            source_batches.clone().into_iter().map(Ok),
            schema.clone(),
        ));
        // Run merge_insert one last time.  The index is now completely out of date.  Every
        // row it points to is a deleted row.  Make sure that doesn't break.
        let ds = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
            .unwrap()
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_matched(WhenMatched::UpdateAll)
            .try_build()
            .unwrap()
            .execute_reader(source)
            .await
            .unwrap();

        assert_eq!(ds.count_rows(None).await.unwrap(), 2048);
    }
}
