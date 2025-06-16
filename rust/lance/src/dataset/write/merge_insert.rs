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

use assign_action::merge_insert_action;
use datafusion_expr::col;
use futures::FutureExt;
use std::{
    collections::{BTreeMap, HashSet},
    future::Future,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use arrow_array::{
    cast::AsArray, types::UInt64Type, BooleanArray, RecordBatch, RecordBatchIterator, StructArray,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::{
    execution::{
        context::{SessionConfig, SessionContext},
        memory_pool::MemoryConsumer,
    },
    logical_expr::{self, Expr, JoinConstraint, JoinType},
    physical_plan::{
        joins::{HashJoinExec, PartitionMode},
        projection::ProjectionExec,
        repartition::RepartitionExec,
        stream::RecordBatchStreamAdapter,
        union::UnionExec,
        ColumnarValue, ExecutionPlan, PhysicalExpr, SendableRecordBatchStream,
    },
    prelude::DataFrame,
    scalar::ScalarValue,
};

use lance_arrow::{interleave_batches, RecordBatchExt, SchemaExt};
use lance_datafusion::{
    chunker::chunk_stream, dataframe::DataFrameExt, exec::get_session_context,
    utils::reader_to_stream,
};

use datafusion_physical_expr::expressions::Column;
use futures::{
    future::Either,
    stream::{self},
    Stream, StreamExt, TryFutureExt, TryStreamExt,
};
use lance_core::{
    datatypes::{OnMissing, OnTypeMismatch, SchemaCompareOptions},
    error::{box_error, InvalidInputSnafu},
    utils::{
        backoff::SlotBackoff, futures::Capacity, mask::RowIdTreeMap,
        tokio::get_num_compute_intensive_cpus,
    },
    Error, Result, ROW_ADDR, ROW_ADDR_FIELD, ROW_ID, ROW_ID_FIELD,
};
use lance_datafusion::{
    exec::{execute_plan, LanceExecutionOptions, OneShotExec},
    utils::StreamingWriteSource,
};
use lance_file::version::LanceFileVersion;
use lance_index::{DatasetIndexExt, ScalarIndexCriteria};
use lance_table::format::{Fragment, Index};
use log::info;
use roaring::RoaringTreemap;
use snafu::{location, ResultExt};
use tokio::task::JoinSet;

use crate::{
    datafusion::dataframe::SessionContextExt,
    dataset::{
        fragment::{FileFragment, FragReadConfig},
        transaction::{Operation, Transaction},
        write::open_writer,
    },
    index::DatasetIndexInternalExt,
    io::exec::{
        project, scalar_index::MapIndexExec, utils::ReplayExec, AddRowAddrExec, Planner, TakeExec,
    },
    Dataset,
};

use super::{write_fragments_internal, CommitBuilder, WriteParams};

mod assign_action;
mod exec;
mod logical_plan;

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
#[derive(Debug, Clone, PartialEq, Eq)]
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
            .context(InvalidInputSnafu {
                location: location!(),
            })?;
        let expr = planner
            .optimize_expr(expr)
            .map_err(box_error)
            .context(InvalidInputSnafu {
                location: location!(),
            })?;
        Ok(Self::DeleteIf(expr))
    }
}

impl std::hash::Hash for WhenNotMatchedBySource {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            WhenNotMatchedBySource::Keep | WhenNotMatchedBySource::Delete => {},
            WhenNotMatchedBySource::DeleteIf(_) => {
                format!("{:?}", self).hash(state);
            }
        }
    }
}

impl PartialOrd for WhenNotMatchedBySource {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WhenNotMatchedBySource {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_disc = std::mem::discriminant(self);
        let other_disc = std::mem::discriminant(other);
        
        // Compare discriminants first
        if self_disc != other_disc {
            // Use a simple ordering based on variant pattern
            let self_order = match self {
                WhenNotMatchedBySource::Keep => 0u8,
                WhenNotMatchedBySource::Delete => 1u8,
                WhenNotMatchedBySource::DeleteIf(_) => 2u8,
            };
            let other_order = match other {
                WhenNotMatchedBySource::Keep => 0u8,
                WhenNotMatchedBySource::Delete => 1u8,
                WhenNotMatchedBySource::DeleteIf(_) => 2u8,
            };
            return self_order.cmp(&other_order);
        }
        
        // If same variant, compare contents for complex variants
        if let (WhenNotMatchedBySource::DeleteIf(_), WhenNotMatchedBySource::DeleteIf(_)) = (self, other) {
            format!("{:?}", self).cmp(&format!("{:?}", other))
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

/// Describes how rows should be handled when there is a match between the target table and source table
#[derive(Debug, Clone, PartialEq, Eq)]
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
            .context(InvalidInputSnafu {
                location: location!(),
            })?;
        let expr = planner
            .optimize_expr(expr)
            .map_err(box_error)
            .context(InvalidInputSnafu {
                location: location!(),
            })?;
        Ok(Self::UpdateIf(expr))
    }
}

impl std::hash::Hash for WhenMatched {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            WhenMatched::UpdateAll | WhenMatched::DoNothing => {},
            WhenMatched::UpdateIf(_) => {
                format!("{:?}", self).hash(state);
            }
        }
    }
}

impl PartialOrd for WhenMatched {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WhenMatched {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_disc = std::mem::discriminant(self);
        let other_disc = std::mem::discriminant(other);
        
        // Compare discriminants first
        if self_disc != other_disc {
            // Use a simple ordering based on variant pattern
            let self_order = match self {
                WhenMatched::UpdateAll => 0u8,
                WhenMatched::DoNothing => 1u8,
                WhenMatched::UpdateIf(_) => 2u8,
            };
            let other_order = match other {
                WhenMatched::UpdateAll => 0u8,
                WhenMatched::DoNothing => 1u8,
                WhenMatched::UpdateIf(_) => 2u8,
            };
            return self_order.cmp(&other_order);
        }
        
        // If same variant, compare contents for complex variants
        if let (WhenMatched::UpdateIf(_), WhenMatched::UpdateIf(_)) = (self, other) {
            format!("{:?}", self).cmp(&format!("{:?}", other))
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

/// Describes how rows should be handled when there is no matching row in the target table
///
/// These are new rows which do not match any old data
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WhenNotMatched {
    /// The new row is inserted into the target table
    ///
    /// This is used in both find-or-create and upsert operations
    InsertAll,
    /// The new row is ignored
    DoNothing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MergeInsertParams {
    // The column(s) to join on
    on: Vec<String>,
    // If true, then update all columns of the old data to the new data when there is a match
    when_matched: WhenMatched,
    // If true, then insert all columns of the new data when there is no match in the old data
    insert_not_matched: bool,
    // Controls whether data that is not matched by the source is deleted or not
    delete_not_matched_by_source: WhenNotMatchedBySource,
    conflict_retries: u32,
    retry_timeout: Duration,
}

impl std::hash::Hash for MergeInsertParams {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.on.hash(state);
        self.when_matched.hash(state);
        self.insert_not_matched.hash(state);
        self.delete_not_matched_by_source.hash(state);
        self.conflict_retries.hash(state);
        self.retry_timeout.hash(state);
    }
}

impl PartialOrd for MergeInsertParams {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeInsertParams {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.on.cmp(&other.on)
            .then_with(|| self.when_matched.cmp(&other.when_matched))
            .then_with(|| self.insert_not_matched.cmp(&other.insert_not_matched))
            .then_with(|| self.delete_not_matched_by_source.cmp(&other.delete_not_matched_by_source))
            .then_with(|| self.conflict_retries.cmp(&other.conflict_retries))
            .then_with(|| self.retry_timeout.cmp(&other.retry_timeout))
    }
}

/// A MergeInsertJob inserts new rows, deletes old rows, and updates existing rows all as
/// part of a single transaction.
#[derive(Clone)]
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
                conflict_retries: 10,
                retry_timeout: Duration::from_secs(30),
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

    /// Set number of times to retry the operation if there is contention.
    ///
    /// If this is set > 0, then the operation will keep a copy of the input data
    /// either in memory or on disk (depending on the size of the data) and will
    /// retry the operation if there is contention.
    ///
    /// Default is 10.
    pub fn conflict_retries(&mut self, retries: u32) -> &mut Self {
        self.params.conflict_retries = retries;
        self
    }

    /// Set the timeout used to limit retries.
    ///
    /// This is the maximum time to spend on the operation before giving up. At
    /// least one attempt will be made, regardless of how long it takes to complete.
    /// Subsequent attempts will be cancelled once this timeout is reached. If
    /// the timeout has been reached during the first attempt, the operation
    /// will be cancelled immediately.
    ///
    /// The default is 30 seconds.
    pub fn retry_timeout(&mut self, timeout: Duration) -> &mut Self {
        self.params.retry_timeout = timeout;
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

enum SchemaComparison {
    FullCompatible,
    Subschema,
}

impl MergeInsertJob {
    pub async fn execute_reader(
        self,
        source: impl StreamingWriteSource,
    ) -> Result<(Arc<Dataset>, MergeStats)> {
        let stream = source.into_stream();
        self.execute(stream).await
    }

    fn check_compatible_schema(&self, schema: &Schema) -> Result<SchemaComparison> {
        let lance_schema: lance_core::datatypes::Schema = schema.try_into()?;
        let is_compatible = lance_schema.check_compatible(
            self.dataset.schema(),
            &SchemaCompareOptions {
                compare_dictionary: true,
                ..Default::default()
            },
        );

        fn is_subschema(schema: &Schema, candidate: &Schema) -> bool {
            // Schema::contains() cares about order, but we don't.
            for field in candidate.fields() {
                if !schema
                    .field_with_name(field.name())
                    .map(|f| f.contains(field))
                    .unwrap_or(false)
                {
                    return false;
                }
            }
            true
        }

        if let Err(e) = is_compatible {
            // It might be a subschema
            let dataset_arrow_schema = Schema::from(self.dataset.schema());
            if is_subschema(&dataset_arrow_schema, schema) {
                Ok(SchemaComparison::Subschema)
            } else {
                Err(e)
            }
        } else {
            Ok(SchemaComparison::FullCompatible)
        }
    }

    async fn join_key_as_scalar_index(&self) -> Result<Option<Index>> {
        if self.params.on.len() != 1 {
            // joining on more than one column
            Ok(None)
        } else {
            let col = &self.params.on[0];
            self.dataset
                .load_scalar_index(
                    ScalarIndexCriteria::default()
                        .for_column(col)
                        // Unclear if this would work if the index does not support exact equality
                        .supports_exact_equality(),
                )
                .await
        }
    }

    async fn create_indexed_scan_joined_stream(
        &self,
        source: SendableRecordBatchStream,
        index: Index,
    ) -> Result<SendableRecordBatchStream> {
        // This relies on a few non-standard physical operators and so we cannot use the
        // datafusion dataframe API and need to construct the plan manually :'(
        let schema = source.schema();
        let add_row_addr = match self.check_compatible_schema(&schema)? {
            SchemaComparison::FullCompatible => false,
            SchemaComparison::Subschema => true,
        };

        // 1 - Input from user
        let input = Arc::new(OneShotExec::new(source));

        // 2 - Fork/Replay the input
        // Regrettably, this needs to have unbounded capacity, and so we need to fully read
        // the new data into memory.  In the future, we can do better
        let shared_input = Arc::new(ReplayExec::new(Capacity::Unbounded, input));

        // 3 - Use the index to map input to row addresses
        // First, we need to project to the key column
        let field = schema.field_with_name(&self.params.on[0])?;
        let index_mapper_input = Arc::new(project(
            shared_input.clone(),
            // schema for only the key join column
            &Schema::new(vec![field.clone()]),
        )?);

        // Then we pass the key column into the index mapper
        let index_column = self.params.on[0].clone();
        let mut index_mapper: Arc<dyn ExecutionPlan> = Arc::new(MapIndexExec::new(
            // create index from original data and key column
            self.dataset.clone(),
            index_column.clone(),
            index.name.clone(),
            index_mapper_input,
        ));

        // If requested, add row addresses to the output
        if add_row_addr {
            let pos = index_mapper.schema().fields().len(); // Add to end
            index_mapper = Arc::new(AddRowAddrExec::try_new(
                index_mapper,
                self.dataset.clone(),
                pos,
            )?);
        }

        // 4 - Take the mapped row ids
        let projection = self
            .dataset
            .empty_projection()
            .union_arrow_schema(schema.as_ref(), OnMissing::Error)?;
        let mut target =
            Arc::new(TakeExec::try_new(self.dataset.clone(), index_mapper, projection)?.unwrap())
                as Arc<dyn ExecutionPlan>;

        // 5 - Take puts the row id and row addr at the beginning.  A full scan (used when there is
        //     no scalar index) puts the row id and addr at the end.  We need to match these up so
        //     we reorder those columns at the end.
        let schema = target.schema();
        let mut columns = schema
            .fields()
            .iter()
            .filter(|f| f.name() != ROW_ID && f.name() != ROW_ADDR)
            .cloned()
            .collect::<Vec<_>>();
        columns.push(Arc::new(ROW_ID_FIELD.clone()));
        if add_row_addr {
            columns.push(Arc::new(ROW_ADDR_FIELD.clone()));
        }
        target = Arc::new(project(target, &Schema::new(columns))?);

        let column_names = schema
            .field_names()
            .into_iter()
            .filter(|name| name.as_str() != ROW_ID && name.as_str() != ROW_ADDR)
            .collect::<Vec<_>>();

        // 5a - We also need to scan any new unindexed data and union it in
        let unindexed_fragments = self.dataset.unindexed_fragments(&index.name).await?;
        if !unindexed_fragments.is_empty() {
            let mut builder = self.dataset.scan();
            if add_row_addr {
                builder.with_row_address();
            }
            let unindexed_data = builder
                .with_row_id()
                .with_fragments(unindexed_fragments)
                .project(&column_names)
                .unwrap()
                .create_plan()
                .await?;
            let unioned = UnionExec::new(vec![target, unindexed_data]);
            // Enforce only 1 partition.
            target = Arc::new(RepartitionExec::try_new(
                Arc::new(unioned),
                datafusion::physical_plan::Partitioning::RoundRobinBatch(1),
            )?);
        }

        // We need to prefix the fields in the target with target_ so that we don't have any duplicate
        // field names (DF doesn't support this as of version 44)
        target = Self::prefix_columns_phys(target, "target_");

        // 6 - Finally, join the input (source table) with the taken data (target table)
        let source_key = Column::new_with_schema(&index_column, shared_input.schema().as_ref())?;
        let target_key = Column::new_with_schema(
            &format!("target_{}", index_column),
            target.schema().as_ref(),
        )?;
        let joined = Arc::new(
            HashJoinExec::try_new(
                shared_input,
                target,
                vec![(Arc::new(source_key), Arc::new(target_key))],
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

    fn prefix_columns(df: DataFrame, prefix: &str) -> DataFrame {
        let schema = df.schema();
        let columns = schema
            .fields()
            .iter()
            .map(|f| {
                // Need to "quote" the column name so it gets interpreted case-sensitively
                logical_expr::col(format!("\"{}\"", f.name())).alias(format!(
                    "{}{}",
                    prefix,
                    f.name()
                ))
            })
            .collect::<Vec<_>>();
        df.select(columns).unwrap()
    }

    fn prefix_columns_phys(inp: Arc<dyn ExecutionPlan>, prefix: &str) -> Arc<dyn ExecutionPlan> {
        let schema = inp.schema();
        let exprs = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(idx, f)| {
                let col = Arc::new(Column::new(f.name(), idx)) as Arc<dyn PhysicalExpr>;
                let new_name = format!("{}{}", prefix, f.name());
                (col, new_name)
            })
            .collect::<Vec<_>>();
        Arc::new(ProjectionExec::try_new(exprs, inp).unwrap())
    }

    // If the join keys are not indexed then we need to do a full scan of the table
    async fn create_full_table_joined_stream(
        &self,
        source: SendableRecordBatchStream,
    ) -> Result<SendableRecordBatchStream> {
        let session_config = SessionConfig::default().with_target_partitions(1);
        let session_ctx = SessionContext::new_with_config(session_config);
        let schema = source.schema();
        let new_data = session_ctx.read_one_shot(source)?;
        let join_cols = self
            .params
            .on // columns to join on
            .iter()
            .map(|c| c.as_str())
            .collect::<Vec<_>>(); // vector of strings of col names to join
        let target_cols = self
            .params
            .on
            .iter()
            .map(|c| format!("target_{}", c))
            .collect::<Vec<_>>();
        let target_cols = target_cols.iter().map(|s| s.as_str()).collect::<Vec<_>>();

        match self.check_compatible_schema(&schema)? {
            SchemaComparison::FullCompatible => {
                let existing = session_ctx.read_lance(self.dataset.clone(), true, false)?;
                // We need to rename the columns from the target table so that they don't conflict with the source table
                let existing = Self::prefix_columns(existing, "target_");
                let joined =
                    new_data.join(existing, JoinType::Full, &join_cols, &target_cols, None)?; // full join
                Ok(joined.execute_stream().await?)
            }
            SchemaComparison::Subschema => {
                let existing = session_ctx.read_lance(self.dataset.clone(), true, true)?;
                let columns = schema
                    .field_names()
                    .iter()
                    .map(|s| s.as_str())
                    .chain([ROW_ID, ROW_ADDR])
                    .collect::<Vec<_>>();
                let projected = existing.select_columns(&columns)?;
                // We need to rename the columns from the target table so that they don't conflict with the source table
                let projected = Self::prefix_columns(projected, "target_");
                // We aren't supporting inserts or deletes right now, so we can use inner join
                let join_type = if self.params.insert_not_matched {
                    JoinType::Left
                } else {
                    JoinType::Inner
                };
                let joined = new_data.join(projected, join_type, &join_cols, &target_cols, None)?;
                Ok(joined.execute_stream().await?)
            }
        }
    }

    /// Join the source and target data streams
    ///
    /// If there is a scalar index on the join key, we can use it to do an indexed join.  Otherwise we need to do
    /// a full outer join.
    ///
    /// Datafusion doesn't allow duplicate column names so during this join we rename the columns from target and
    /// prefix them with _target.
    async fn create_joined_stream(
        &self,
        source: SendableRecordBatchStream,
    ) -> Result<SendableRecordBatchStream> {
        // We need to do a full index scan if we're deleting source data
        let can_use_scalar_index = matches!(
            self.params.delete_not_matched_by_source, // this value marks behavior for rows in target that are not matched by the source. Value assigned earlier.
            WhenNotMatchedBySource::Keep
        );
        if can_use_scalar_index {
            // keeping unmatched rows, no deletion
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

    async fn update_fragments(
        dataset: Arc<Dataset>,
        source: SendableRecordBatchStream,
    ) -> Result<(Vec<Fragment>, Vec<Fragment>, Vec<u32>)> {
        // Expected source schema: _rowaddr, updated_cols*
        use datafusion::logical_expr::{col, lit};
        let session_ctx = get_session_context(&LanceExecutionOptions {
            use_spilling: true,
            target_partition: Some(get_num_compute_intensive_cpus().min(8)),
            ..Default::default()
        });
        let mut group_stream = session_ctx
            .read_one_shot(source)?
            .with_column("_fragment_id", col(ROW_ADDR) >> lit(32))?
            .sort(vec![col(ROW_ADDR).sort(true, true)])?
            .group_by_stream(&["_fragment_id"])
            .await?;

        // Can update the fragments in parallel.
        let updated_fragments = Arc::new(Mutex::new(Vec::new()));
        let new_fragments = Arc::new(Mutex::new(Vec::new()));
        let mut tasks = JoinSet::new();
        let task_limit = dataset.object_store().io_parallelism();
        let mut reservation =
            MemoryConsumer::new("MergeInsert").register(session_ctx.task_ctx().memory_pool());

        while let Some((frag_id, batches)) = group_stream.next().await.transpose()? {
            async fn handle_fragment(
                dataset: Arc<Dataset>,
                fragment: FileFragment,
                mut metadata: Fragment,
                mut batches: Vec<RecordBatch>,
                updated_fragments: Arc<Mutex<Vec<Fragment>>>,
                reservation_size: usize,
            ) -> Result<usize> {
                // batches still have _rowaddr
                let write_schema = batches[0].schema().as_ref().without_column(ROW_ADDR);
                let write_schema = dataset.local_schema().project_by_schema(
                    &write_schema,
                    OnMissing::Error,
                    OnTypeMismatch::Error,
                )?;

                let updated_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
                if Some(updated_rows) == metadata.physical_rows {
                    // All rows have been updated and there are no deletions. So we
                    // don't need to merge in existing values.
                    // Also, because we already sorted by row address, the rows
                    // will be in the correct order.

                    let data_storage_version = dataset
                        .manifest()
                        .data_storage_format
                        .lance_file_version()?;
                    let mut writer = open_writer(
                        dataset.object_store(),
                        &write_schema,
                        &dataset.base,
                        data_storage_version,
                    )
                    .await?;

                    // We need to remove rowaddr before writing.
                    batches
                        .iter_mut()
                        .try_for_each(|batch| match batch.drop_column(ROW_ADDR) {
                            Ok(b) => {
                                *batch = b;
                                Ok(())
                            }
                            Err(e) => Err(e),
                        })?;

                    if data_storage_version == LanceFileVersion::Legacy {
                        // Need to match the existing batch size exactly, otherwise
                        // we'll get errors.
                        let reader = fragment
                            .open(
                                dataset.schema(),
                                FragReadConfig::default().with_row_address(true),
                            )
                            .await?;
                        let batch_size = reader.legacy_num_rows_in_batch(0).unwrap();
                        let stream = stream::iter(batches.into_iter().map(Ok));
                        let stream = Box::pin(RecordBatchStreamAdapter::new(
                            Arc::new((&write_schema).into()),
                            stream,
                        ));
                        let mut stream = chunk_stream(stream, batch_size as usize);
                        while let Some(chunk) = stream.next().await {
                            writer.write(&chunk?).await?;
                        }
                    } else {
                        writer.write(batches.as_slice()).await?;
                    }

                    let (_num_rows, data_file) = writer.finish().await?;

                    metadata.files.push(data_file);
                    updated_fragments.lock().unwrap().push(metadata);
                } else {
                    // TODO: we could skip scanning row addresses we don't need.
                    let update_schema = batches[0].schema();
                    let read_columns = update_schema.field_names();
                    let mut updater = fragment
                        .updater(
                            Some(&read_columns),
                            Some((write_schema, dataset.schema().clone())),
                            None,
                        )
                        .await?;

                    // We will use interleave to update the rows. The first batch
                    // will be the original source data, and all subsequent batches
                    // will be updates.
                    let mut source_batches = Vec::with_capacity(batches.len() + 1);
                    source_batches.push(batches[0].clone()); // placeholder for source data
                    for batch in &batches {
                        source_batches.push(batch.drop_column(ROW_ADDR)?);
                    }

                    // This function is here to help rustc with lifetimes.
                    fn get_row_addr_iter(
                        batches: &[RecordBatch],
                    ) -> impl Iterator<Item = (u64, (usize, usize))> + '_ + Send
                    {
                        batches.iter().enumerate().flat_map(|(batch_idx, batch)| {
                            // The index in source batches will be one more.
                            let batch_idx = batch_idx + 1;
                            let row_addrs = batch
                                .column_by_name(ROW_ADDR)
                                .unwrap()
                                .as_any()
                                .downcast_ref::<UInt64Array>()
                                .unwrap();
                            row_addrs
                                .values()
                                .iter()
                                .enumerate()
                                .map(move |(offset, row_addr)| (*row_addr, (batch_idx, offset)))
                        })
                    }
                    let mut updated_row_addr_iter = get_row_addr_iter(&batches).peekable();

                    while let Some(batch) = updater.next().await? {
                        source_batches[0] =
                            batch.project_by_schema(source_batches[1].schema().as_ref())?;

                        let original_row_addrs = batch
                            .column_by_name(ROW_ADDR)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap();
                        let indices = original_row_addrs
                            .values()
                            .into_iter()
                            .enumerate()
                            .map(|(original_offset, row_addr)| {
                                match updated_row_addr_iter.peek() {
                                    Some((updated_row_addr, _))
                                        if *updated_row_addr == *row_addr =>
                                    {
                                        updated_row_addr_iter.next().unwrap().1
                                    }
                                    // If we have passed the next updated row address, something went wrong.
                                    Some((updated_row_addr, _)) => {
                                        debug_assert!(
                                        *updated_row_addr > *row_addr,
                                        "Got updated row address that is not in the original batch"
                                    );
                                        (0, original_offset)
                                    }
                                    _ => (0, original_offset),
                                }
                            })
                            .collect::<Vec<_>>();

                        let updated_batch = interleave_batches(&source_batches, &indices)?;

                        updater.update(updated_batch).await?;
                    }

                    let updated_fragment = updater.finish().await?;
                    updated_fragments.lock().unwrap().push(updated_fragment);
                }
                Ok(reservation_size)
            }

            async fn handle_new_fragments(
                dataset: Arc<Dataset>,
                batches: Vec<RecordBatch>,
                new_fragments: Arc<Mutex<Vec<Fragment>>>,
                reservation_size: usize,
            ) -> Result<usize> {
                // Batches still have _rowaddr (used elsewhere to merge with existing data)
                // We need to remove it before writing to Lance files.
                let num_fields = batches[0].schema().fields().len();
                let mut projection = Vec::with_capacity(num_fields - 1);
                for (i, field) in batches[0].schema().fields().iter().enumerate() {
                    if field.name() != ROW_ADDR {
                        projection.push(i);
                    }
                }
                let write_schema = Arc::new(batches[0].schema().project(&projection).unwrap());

                let batches = batches
                    .into_iter()
                    .map(move |batch| batch.project(&projection));
                let reader = RecordBatchIterator::new(batches, write_schema.clone());
                let stream = reader_to_stream(Box::new(reader));

                let write_schema = dataset.schema().project_by_schema(
                    write_schema.as_ref(),
                    OnMissing::Error,
                    OnTypeMismatch::Error,
                )?;

                let fragments = write_fragments_internal(
                    Some(dataset.as_ref()),
                    dataset.object_store.clone(),
                    &dataset.base,
                    write_schema,
                    stream,
                    Default::default(), // TODO: support write params.
                )
                .await?;

                new_fragments.lock().unwrap().extend(fragments.default.0);
                Ok(reservation_size)
            }
            // We shouldn't need much more memory beyond what is already in the batches.
            let mut memory_size = batches
                .iter()
                .map(|batch| batch.get_array_memory_size())
                .sum();

            loop {
                let have_additional_cpus = tasks.len() < task_limit;
                if have_additional_cpus {
                    if reservation.try_grow(memory_size).is_ok() {
                        break;
                    } else if tasks.is_empty() {
                        // If there are no tasks running, we can bypass the pool limits.
                        // This lets us handle the case where we have a single large batch.
                        memory_size = 0;
                        break;
                    }
                    // If we can't grow the reservation, we will wait for a task to finish
                }

                if let Some(res) = tasks.join_next().await {
                    let size = res??;
                    reservation.shrink(size);
                }
            }

            match frag_id.first() {
                Some(ScalarValue::UInt64(Some(frag_id))) => {
                    let frag_id = *frag_id;
                    let fragment =
                        dataset
                            .get_fragment(frag_id as usize)
                            .ok_or_else(|| Error::Internal {
                                message: format!(
                                    "Got non-existent fragment id from merge result: {}",
                                    frag_id
                                ),
                                location: location!(),
                            })?;
                    let metadata = fragment.metadata.clone();

                    let fut = handle_fragment(
                        dataset.clone(),
                        fragment,
                        metadata,
                        batches,
                        updated_fragments.clone(),
                        memory_size,
                    );
                    tasks.spawn(fut);
                }
                Some(ScalarValue::Null | ScalarValue::UInt64(None)) => {
                    let fut = handle_new_fragments(
                        dataset.clone(),
                        batches,
                        new_fragments.clone(),
                        memory_size,
                    );
                    tasks.spawn(fut);
                }
                _ => {
                    return Err(Error::Internal {
                        message: format!("Got non-fragment id from merge result: {:?}", frag_id),
                        location: location!(),
                    });
                }
            };
        }

        while let Some(res) = tasks.join_next().await {
            let size = res??;
            reservation.shrink(size);
        }
        let mut updated_fragments = Arc::try_unwrap(updated_fragments)
            .unwrap()
            .into_inner()
            .unwrap();

        // We keep track of all fields that are updated so we can prune the indices.
        // We could maybe be more precise since some fields are not modified in some
        // fragments (if they were already null) but this is simpler and good enough
        // for now.
        let mut all_fields_updated = HashSet::new();

        // Collect the updated fragments, and map the field ids. Tombstone old ones
        // as needed.
        for fragment in &mut updated_fragments {
            let updated_fields = fragment.files.last().unwrap().fields.clone();
            all_fields_updated.extend(updated_fields.iter().map(|&f| f as u32));
            for data_file in &mut fragment.files.iter_mut().rev().skip(1) {
                for field in &mut data_file.fields {
                    if updated_fields.contains(field) {
                        // Tombstone these fields
                        *field = -2;
                    }
                }
            }
        }

        let new_fragments = Arc::try_unwrap(new_fragments)
            .unwrap()
            .into_inner()
            .unwrap();

        Ok((
            updated_fragments,
            new_fragments,
            all_fields_updated.into_iter().collect(),
        ))
    }

    /// Executes the merge insert job
    ///
    /// This will take in the source, merge it with the existing target data, and insert new
    /// rows, update existing rows, and delete existing rows
    pub async fn execute(
        mut self,
        source: SendableRecordBatchStream,
    ) -> Result<(Arc<Dataset>, MergeStats)> {
        let start = Instant::now();
        let mut source_iter =
            super::new_source_iter(source, self.params.conflict_retries > 0).await?;

        let mut dataset_ref = self.dataset.clone();
        let max_retries = self.params.conflict_retries;
        let mut backoff = SlotBackoff::default();

        fn timeout_error(retry_timeout: Duration, attempts: u32) -> Error {
            Error::TooMuchWriteContention {
                message: format!(
                    "Attempted {} times, but failed on retry_timeout of {:.3} seconds.",
                    attempts,
                    retry_timeout.as_secs_f32()
                ),
                location: location!(),
            }
        }

        fn maybe_timeout<T>(
            backoff: &SlotBackoff,
            start: Instant,
            retry_timeout: Duration,
            future: impl Future<Output = T>,
        ) -> impl Future<Output = Result<T>> {
            let attempt = backoff.attempt();
            if attempt == 0 {
                // No timeout on first attempt
                Either::Left(future.map(|res| Ok(res)))
            } else {
                let remaining = retry_timeout.saturating_sub(start.elapsed());
                Either::Right(
                    tokio::time::timeout(remaining, future)
                        .map_err(move |_| timeout_error(retry_timeout, attempt + 1)),
                )
            }
        }

        while backoff.attempt() <= max_retries {
            let ds = dataset_ref.clone();
            let execute_fut = self
                .clone()
                .execute_uncommitted_impl(source_iter.next().unwrap());
            let execute_fut =
                maybe_timeout(&backoff, start, self.params.retry_timeout, execute_fut);
            let UncommittedMergeInsert {
                transaction,
                mut stats,
                affected_rows,
            } = execute_fut.await??;
            stats.num_attempts = backoff.attempt() + 1;

            let mut commit_builder = CommitBuilder::new(ds.clone());
            if let Some(affected_rows) = affected_rows {
                commit_builder = commit_builder.with_affected_rows(affected_rows);
            }
            let commit_future = commit_builder.execute(transaction);
            let commit_future =
                maybe_timeout(&backoff, start, self.params.retry_timeout, commit_future);
            match commit_future.await? {
                Ok(ds) => return Ok((Arc::new(ds), stats)),
                Err(Error::RetryableCommitConflict { .. }) => {
                    // Check whether we have exhausted our retries *before*
                    // we sleep.
                    if backoff.attempt() >= max_retries {
                        break;
                    }
                    if start.elapsed() > self.params.retry_timeout {
                        return Err(timeout_error(
                            self.params.retry_timeout,
                            backoff.attempt() + 1,
                        ));
                    }
                    if backoff.attempt() == 0 {
                        // We add 10% buffer here, to allow concurrent writes to complete.
                        // We pass the first attempt's time to the backoff so it's used
                        // as the unit for backoff time slots.
                        // See SlotBackoff implementation for more details on how this works.
                        backoff = backoff.with_unit((start.elapsed().as_millis() * 11 / 10) as u32);
                    }

                    let sleep_fut = tokio::time::sleep(backoff.next_backoff());
                    let sleep_fut =
                        maybe_timeout(&backoff, start, self.params.retry_timeout, sleep_fut);
                    sleep_fut.await?;

                    let mut ds = dataset_ref.as_ref().clone();
                    ds.checkout_latest().await?;

                    dataset_ref = Arc::new(ds);
                    self.dataset = dataset_ref.clone();
                    continue;
                }
                Err(e) => return Err(e),
            };
        }
        Err(Error::TooMuchWriteContention {
            message: format!("Attempted {} retries.", max_retries),
            location: location!(),
        })
    }

    /// Execute the merge insert job without committing the changes.
    ///
    /// Use [`CommitBuilder`] to commit the returned transaction.
    pub async fn execute_uncommitted(
        self,
        source: impl StreamingWriteSource,
    ) -> Result<UncommittedMergeInsert> {
        let stream = source.into_stream();
        self.execute_uncommitted_impl(stream).await
    }

    async fn create_plan(
        self,
        source: SendableRecordBatchStream,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // For upsert, we create a plan:
        // MergeInsertWriteExec
        //   ProjectionExec [action=merge_insert_action(), +keep input columns]
        //     HashJoin [type=left, on=id]
        //       OneShotRead [input]
        //       Scan columns=[id, _rowid, _rowaddr]

        // Goal: we shouldn't manually have to specify which columns to scan.
        //       DataFusion's optimizer should be able to automatically perform
        //       projection pushdown for us.
        // Goal: we shouldn't have to add new branches in this code to handle
        //       indexed vs non-indexed cases. That should be handled by optimizer rules.
        let session_config = SessionConfig::default();
        let session_ctx = SessionContext::new_with_config(session_config);
        // todo!("register planning rules to session ctx");
        let scan = session_ctx.read_lance(self.dataset.clone(), true, true)?;
        let on_cols = self
            .params
            .on
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>();
        let source_df = session_ctx.read_one_shot(source)?;
        // For now, let's create a simple join and handle duplicates
        let source_df_aliased = source_df.alias("source")?;
        let scan_aliased = scan.alias("target")?;
        let df = source_df_aliased
            .join(scan_aliased, JoinType::Left, &on_cols, &on_cols, None)?
            .with_column("action", merge_insert_action(&self.params)?)?;
        dbg!(df.schema());
        // TODO: apply the write node at the end
        
        let (session_state, logical_plan) = df.into_parts();

        let logical_plan = session_state.optimize(&logical_plan)?;

        // TODO: we need to register a planner that can take a logical write and make it a FullSchemaMergeInsertExec
        let mut physical_plan = session_state.create_physical_plan(&logical_plan).await?;

        // Do multiple passes (Maybe datafusion has a method for this, not sure?)
        let optimizers = session_state.physical_optimizers();
        for _ in 0..3 {
            for optimizer in optimizers {
                physical_plan =
                    optimizer.optimize(physical_plan, session_state.config_options())?;
            }
        }

        Ok(physical_plan)
    }

    async fn execute_uncommitted_v2(
        self,
        source: SendableRecordBatchStream,
    ) -> Result<(Transaction, MergeStats)> {
        let plan = self.create_plan(source).await?;

        // Execute the plan
        // Assert that we have exactly one partition since we're designed for single-partition execution
        let partition_count = match plan.properties().output_partitioning() {
            datafusion_physical_expr::Partitioning::RoundRobinBatch(n) => *n,
            datafusion_physical_expr::Partitioning::Hash(_, n) => *n,
            datafusion_physical_expr::Partitioning::UnknownPartitioning(n) => *n,
        };
        
        if partition_count != 1 {
            return Err(Error::invalid_input(
                format!("Expected exactly 1 partition, got {}", partition_count),
                location!(),
            ));
        }

        // Execute partition 0 (the only partition)
        let task_context = Arc::new(datafusion::execution::TaskContext::default());
        let mut stream = plan.execute(0, task_context)?;
        
        // Assert that the execution produces no output (this is a write operation)
        if let Some(batch) = stream.next().await {
            let batch = batch?;
            if batch.num_rows() > 0 {
                return Err(Error::invalid_input(
                    format!("Expected no output from write operation, got {} rows", batch.num_rows()),
                    location!(),
                ));
            }
        }

        // Extract merge stats from the execution plan
        let merge_insert_exec = plan
            .as_any()
            .downcast_ref::<exec::FullSchemaMergeInsertExec>()
            .ok_or_else(|| Error::invalid_input("Expected FullSchemaMergeInsertExec", location!()))?;
        
        let stats = merge_insert_exec
            .merge_stats()
            .ok_or_else(|| Error::invalid_input("Merge stats not available - execution may not have completed", location!()))?;

        let transaction = merge_insert_exec
            .transaction()
            .ok_or_else(|| Error::invalid_input("Transaction not available - execution may not have completed", location!()))?;

        Ok((transaction, stats))
    }

    async fn execute_uncommitted_impl(
        self,
        source: SendableRecordBatchStream,
    ) -> Result<UncommittedMergeInsert> {
        // We are migrating a new plan-based code path. For now, only using this
        // for select supported queries. Right now that is just upsert without
        // any scalar index.
        if self.params.insert_not_matched
            && matches!(self.params.when_matched, WhenMatched::UpdateAll)
            && todo!("there is no scalar index")
        {
            let (transaction, stats) = self.execute_uncommitted_v2(source).await?;
            return Ok(UncommittedMergeInsert {
                transaction,
                affected_rows: None,
                stats,
            });
        }
        // Erase metadata on source / dataset schemas to avoid comparing metadata
        let schema = lance_core::datatypes::Schema::try_from(source.schema().as_ref())?;
        let full_schema = self.dataset.local_schema();
        let is_full_schema = full_schema.compare_with_options(
            &schema,
            &SchemaCompareOptions {
                compare_metadata: false,
                ..Default::default()
            },
        );

        let source_schema = source.schema();
        let joined = self.create_joined_stream(source).await?;
        let merger = Merger::try_new(self.params.clone(), source_schema, !is_full_schema)?;
        let merge_statistics = merger.merge_stats.clone();
        let deleted_rows = merger.deleted_rows.clone();
        let merger_schema = merger.output_schema().clone();
        let stream = joined
            .and_then(move |batch| merger.clone().execute_batch(batch))
            .try_flatten();
        let stream = RecordBatchStreamAdapter::new(merger_schema, stream);

        let (operation, affected_rows) = if !is_full_schema {
            if !matches!(
                self.params.delete_not_matched_by_source,
                WhenNotMatchedBySource::Keep
            ) {
                return Err(Error::NotSupported { source:
                    "Deleting rows from the target table when there is no match in the source table is not supported when the source data has a different schema than the target data".into(), location: location!() });
            }

            // We will have a different commit path here too, as we are modifying
            // fragments rather than writing new ones
            let (updated_fragments, new_fragments, fields_modified) =
                Self::update_fragments(self.dataset.clone(), Box::pin(stream)).await?;

            let operation = Operation::Update {
                removed_fragment_ids: Vec::new(),
                updated_fragments,
                new_fragments,
                fields_modified,
            };
            // We have rewritten the fragments, not just the deletion files, so
            // we can't use affected rows here.
            (operation, None)
        } else {
            let written = write_fragments_internal(
                Some(&self.dataset),
                self.dataset.object_store.clone(),
                &self.dataset.base,
                self.dataset.schema().clone(),
                Box::pin(stream),
                WriteParams::default(),
            )
            .await?;

            assert!(written.blob.is_none());
            let new_fragments = written.default.0;

            // Apply deletions
            let removed_row_ids = Arc::into_inner(deleted_rows).unwrap().into_inner().unwrap();

            let (old_fragments, removed_fragment_ids) =
                Self::apply_deletions(&self.dataset, &removed_row_ids).await?;

            // Commit updated and new fragments
            let operation = Operation::Update {
                removed_fragment_ids,
                updated_fragments: old_fragments,
                new_fragments,
                // On this path we only make deletions against updated_fragments and will not
                // modify any field values.
                fields_modified: vec![],
            };

            let affected_rows = Some(RowIdTreeMap::from(removed_row_ids));
            (operation, affected_rows)
        };

        let stats = Arc::into_inner(merge_statistics)
            .unwrap()
            .into_inner()
            .unwrap();

        let transaction = Transaction::new(
            self.dataset.manifest.version,
            operation,
            /*blobs_op=*/ None,
            None,
        );

        Ok(UncommittedMergeInsert {
            transaction,
            affected_rows,
            stats,
        })
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
            .buffer_unordered(dataset.object_store.io_parallelism());

        while let Some(res) = stream.next().await.transpose()? {
            match res {
                FragmentChange::Unchanged => {}
                FragmentChange::Modified(fragment) => updated_fragments.push(fragment),
                FragmentChange::Removed(fragment_id) => removed_fragments.push(fragment_id),
            }
        }

        Ok((updated_fragments, removed_fragments))
    }
}

/// Merger will store these statistics as it runs (for each batch)
#[derive(Debug, Default, Clone)]
pub struct MergeStats {
    /// Number of inserted rows (for user statistics)
    pub num_inserted_rows: u64,
    /// Number of updated rows (for user statistics)
    pub num_updated_rows: u64,
    /// Number of deleted rows (for user statistics)
    /// Note: This is different from internal references to 'deleted_rows', since we technically "delete" updated rows during processing.
    /// However those rows are not shared with the user.
    pub num_deleted_rows: u64,
    /// Number of attempts performed.
    ///
    /// See [`MergeInsertBuilder::conflict_retries`] for more information.
    pub num_attempts: u32,
}

pub struct UncommittedMergeInsert {
    pub transaction: Transaction,
    pub affected_rows: Option<RowIdTreeMap>,
    pub stats: MergeStats,
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
    // User statistics for merging
    merge_stats: Arc<Mutex<MergeStats>>,
    // Physical "when matched update if" expression, only set if params.when_matched is UpdateIf
    match_filter_expr: Option<Arc<dyn PhysicalExpr>>,
    // The parameters controlling the merge
    params: MergeInsertParams,
    // The schema of the input data, used to recover nullability information
    schema: Arc<Schema>,
    /// Whether the output schema should include a row address column
    with_row_addr: bool,
    /// The output schema of the stream.
    output_schema: Arc<Schema>,
}

impl Merger {
    // Creates a new merger with an empty set of deleted rows, compiles expressions, if present
    fn try_new(
        params: MergeInsertParams,
        schema: Arc<Schema>,
        with_row_addr: bool,
    ) -> Result<Self> {
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
        let output_schema = if with_row_addr {
            Arc::new(schema.try_with_column(ROW_ADDR_FIELD.clone())?)
        } else {
            schema.clone()
        };

        Ok(Self {
            deleted_rows: Arc::new(Mutex::new(RoaringTreemap::new())),
            delete_expr,
            merge_stats: Arc::new(Mutex::new(MergeStats::default())),
            match_filter_expr,
            params,
            schema,
            with_row_addr,
            output_schema,
        })
    }

    fn output_schema(&self) -> &Arc<Schema> {
        &self.output_schema
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
        let mut merge_statistics = self.merge_stats.lock().unwrap();
        let num_fields = batch.schema().fields.len();
        // The schema of the combined batches will be:
        // source_keys, source_payload, target_keys, target_payload, row_id, row_addr?
        // The keys and non_keys on both sides will be equal
        let (row_id_col, row_addr_col, right_offset) = if num_fields % 2 == 1 {
            // No rowaddr
            assert!(!self.with_row_addr);
            (num_fields - 1, None, num_fields / 2)
        } else {
            // Has rowaddr
            assert!(self.with_row_addr);
            (num_fields - 2, Some(num_fields - 1), (num_fields - 2) / 2)
        };

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
                            matched = RecordBatch::new_empty(matched.schema());
                        }
                    }
                }
            }

            merge_statistics.num_updated_rows += matched.num_rows() as u64;

            // If the filter eliminated all rows then its important we don't try and write
            // the batch at all.  Writing an empty batch currently panics
            if matched.num_rows() > 0 {
                let row_ids = matched.column(row_id_col).as_primitive::<UInt64Type>();
                deleted_row_ids.extend(row_ids.values());
                let projection = if let Some(row_addr_col) = row_addr_col {
                    let mut cols = Vec::from_iter(left_cols.iter().cloned());
                    cols.push(row_addr_col);
                    cols
                } else {
                    #[allow(clippy::redundant_clone)]
                    left_cols.clone()
                };
                let matched = matched.project(&projection)?;
                // The payload columns of an outer join are always nullable.  We need to restore
                // non-nullable to columns that were originally non-nullable.  This should be safe
                // since the not_matched rows should all be valid on the right_cols
                //
                // Sadly we can't use with_schema because it doesn't let you toggle nullability
                let matched = RecordBatch::try_new(
                    self.output_schema.clone(),
                    Vec::from_iter(matched.columns().iter().cloned()),
                )?;
                batches.push(Ok(matched));
            }
        }
        if self.params.insert_not_matched {
            let not_matched = arrow::compute::filter_record_batch(&batch, &left_only)?;
            let left_cols_with_id = left_cols
                .into_iter()
                .chain(row_addr_col)
                .collect::<Vec<_>>();
            let not_matched = not_matched.project(&left_cols_with_id)?;
            // See comment above explaining this schema replacement
            let not_matched = RecordBatch::try_new(
                self.output_schema.clone(),
                Vec::from_iter(not_matched.columns().iter().cloned()),
            )?;

            merge_statistics.num_inserted_rows += not_matched.num_rows() as u64;
            batches.push(Ok(not_matched));
        }
        match self.params.delete_not_matched_by_source {
            WhenNotMatchedBySource::Delete => {
                let unmatched = arrow::compute::filter(batch.column(row_id_col), &right_only)?;
                merge_statistics.num_deleted_rows += unmatched.len() as u64;
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
                        merge_statistics.num_deleted_rows += row_ids.len() as u64;
                        deleted_row_ids.extend(row_ids.values());
                    }
                    ColumnarValue::Scalar(scalar) => {
                        if let ScalarValue::Boolean(Some(true)) = scalar {
                            let row_ids = unmatched.column(row_id_col).as_primitive::<UInt64Type>();
                            merge_statistics.num_deleted_rows += row_ids.len() as u64;
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

    use arrow_array::{
        types::UInt32Type, Int64Array, RecordBatchIterator, RecordBatchReader, StringArray,
        UInt32Array,
    };
    use arrow_select::concat::concat_batches;
    use datafusion::common::Column;
    use futures::future::try_join_all;
    use lance_datafusion::{datagen::DatafusionDatagenExt, utils::reader_to_stream};
    use lance_datagen::{array, BatchCount, RowCount, Seed};
    use lance_index::{scalar::ScalarIndexParams, IndexType};
    use lance_io::object_store::ObjectStoreParams;
    use object_store::throttle::ThrottleConfig;
    use roaring::RoaringBitmap;
    use tempfile::tempdir;
    use tokio::sync::{Barrier, Notify};

    use crate::{
        dataset::{builder::DatasetBuilder, InsertBuilder, ReadParams, WriteMode, WriteParams},
        session::Session,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount, ThrottledStoreWrapper},
    };

    use super::*;

    // Used to validate that futures returned are Send.
    fn assert_send<T: Send>(t: T) -> T {
        t
    }

    async fn check(
        new_data: RecordBatch,
        mut job: MergeInsertJob,
        keys_from_left: &[u32],
        keys_from_right: &[u32],
        stats: &[u64],
    ) {
        let mut dataset = (*job.dataset).clone();
        dataset.restore().await.unwrap();
        job.dataset = Arc::new(dataset);

        let schema = new_data.schema();
        let new_reader = Box::new(RecordBatchIterator::new([Ok(new_data)], schema.clone()));
        let new_stream = reader_to_stream(new_reader);

        let (merged_dataset, merge_stats) = job.execute(new_stream).await.unwrap();

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
        assert_eq!(merge_stats.num_inserted_rows, stats[0]);
        assert_eq!(merge_stats.num_updated_rows, stats[1]);
        assert_eq!(merge_stats.num_deleted_rows, stats[2]);
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn test_basic_merge(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
    ) {
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
        let ds = Arc::new(
            Dataset::write(
                batches,
                test_uri,
                Some(WriteParams::with_storage_version(version)),
            )
            .await
            .unwrap(),
        );

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
        check(
            new_batch.clone(),
            job,
            &[1, 2, 3, 4, 5, 6],
            &[7, 8, 9],
            &[3, 0, 0],
        )
        .await;

        // upsert, no delete
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .try_build()
            .unwrap();
        check(
            new_batch.clone(),
            job,
            &[1, 2, 3],
            &[4, 5, 6, 7, 8, 9],
            &[3, 3, 0],
        )
        .await;

        // conditional upsert, no delete
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(
                WhenMatched::update_if(&ds, "source.filterme != target.filterme").unwrap(),
            )
            .try_build()
            .unwrap();
        check(
            new_batch.clone(),
            job,
            &[1, 2, 3, 4, 5],
            &[6, 7, 8, 9],
            &[3, 1, 0],
        )
        .await;

        // conditional update, no matches
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_matched(WhenMatched::update_if(&ds, "target.filterme = 'z'").unwrap())
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1, 2, 3, 4, 5, 6], &[], &[0, 0, 0]).await;

        // update only, no delete (useful for bulk update)
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1, 2, 3], &[4, 5, 6], &[0, 3, 0]).await;

        // Conditional update
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(
                WhenMatched::update_if(&ds, "source.filterme == target.filterme").unwrap(),
            )
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1, 2, 3, 6], &[4, 5], &[0, 2, 0]).await;

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
        check(new_batch.clone(), job, &[4, 5, 6], &[7, 8, 9], &[3, 0, 3]).await;

        // upsert, with delete all
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[], &[4, 5, 6, 7, 8, 9], &[3, 3, 3]).await;

        // update only, with delete all (unusual)
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[], &[4, 5, 6], &[0, 3, 3]).await;

        // just delete all (not real case, just use delete)
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[4, 5, 6], &[], &[0, 0, 3]).await;

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
        check(
            new_batch.clone(),
            job,
            &[1, 4, 5, 6],
            &[7, 8, 9],
            &[3, 0, 2],
        )
        .await;

        // upsert, with delete some
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition.clone()))
            .try_build()
            .unwrap();
        check(
            new_batch.clone(),
            job,
            &[1],
            &[4, 5, 6, 7, 8, 9],
            &[3, 3, 2],
        )
        .await;

        // update only, witxh delete some (unusual)
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition.clone()))
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1], &[4, 5, 6], &[0, 3, 2]).await;

        // just delete some (not real case, just use delete)
        let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition.clone()))
            .try_build()
            .unwrap();
        check(new_batch.clone(), job, &[1, 4, 5, 6], &[], &[0, 0, 2]).await;
    }

    #[tokio::test]
    async fn test_indexed_merge_insert() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = lance_datagen::gen()
            .with_seed(Seed::from(1))
            .col("value", array::step::<UInt32Type>())
            .col("key", array::rand_pseudo_uuid_hex());
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
            .col("key", array::rand_pseudo_uuid_hex());
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
        let (ds, _) = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
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
            .count_rows(Some("value = 9999999".to_string()))
            .await
            .unwrap();
        assert_eq!(updated, 2048);

        // Make sure we don't use an indexed scan if there is a delete criteria
        let source = Box::new(RecordBatchIterator::new(
            source_batches.clone().into_iter().map(Ok),
            schema.clone(),
        ));
        // Run merge_insert
        let (ds, _) = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
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
        let (ds, _) = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
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

    mod subcols {
        use super::*;
        use rstest::rstest;

        struct Fixtures {
            ds: Arc<Dataset>,
            new_data: RecordBatch,
        }

        async fn setup(scalar_index: bool) -> Fixtures {
            let data = lance_datagen::gen()
                .with_seed(Seed::from(1))
                .col("other", array::rand_utf8(4.into(), false))
                .col("value", array::step::<UInt32Type>())
                .col("key", array::rand_pseudo_uuid_hex());
            let batch = data.into_batch_rows(RowCount::from(1024 + 2)).unwrap();
            let batch1 = batch.slice(0, 512);
            let batch2 = batch.slice(512, 512);
            let batch3 = batch.slice(1024, 2);
            let schema = batch.schema();

            let reader = Box::new(RecordBatchIterator::new(
                [Ok(batch1.clone())],
                schema.clone(),
            ));
            let write_params = WriteParams {
                max_rows_per_file: 256,
                max_rows_per_group: 32, // Non-standard group size to hit edge cases
                ..Default::default()
            };
            let mut ds = Dataset::write(reader, "memory://", Some(write_params.clone()))
                .await
                .unwrap();

            if scalar_index {
                let index_params = ScalarIndexParams::default();
                ds.create_index(&["key"], IndexType::Scalar, None, &index_params, false)
                    .await
                    .unwrap();
            }

            // Another two files, not in the scalar index (if there is one)
            let reader = Box::new(RecordBatchIterator::new(
                [Ok(batch2.clone())],
                batch2.schema(),
            ));
            ds.append(reader, Some(write_params)).await.unwrap();

            let ds = Arc::new(ds);

            // New data with only a subset of columns
            let update_schema = Arc::new(schema.project(&[2, 1]).unwrap());
            // Full second file and part of third file. Also two more new rows.
            let indices: Int64Array = (256..512).chain(600..612).chain([712, 715]).collect();
            let keys = arrow::compute::take(batch["key"].as_ref(), &indices, None).unwrap();
            let keys = arrow::compute::concat(&[&keys, &batch3["key"]]).unwrap();
            let num_rows = keys.len();
            let new_data = RecordBatch::try_new(
                update_schema,
                vec![
                    keys,
                    Arc::new((1024..(1024 + num_rows as u32)).collect::<UInt32Array>()),
                ],
            )
            .unwrap();

            Fixtures { ds, new_data }
        }

        #[tokio::test]
        async fn test_delete_not_supported() {
            let Fixtures { ds, new_data } = setup(false).await;

            let reader = Box::new(RecordBatchIterator::new(
                [Ok(new_data.clone())],
                new_data.schema(),
            ));

            // Should reject when_not_matched_by_source_delete as not yet supported
            let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
                .unwrap()
                .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing)
                .try_build()
                .unwrap();
            let res = assert_send(job.execute_reader(reader)).await;
            assert!(
                matches!(
                    &res,
                    &Err(Error::NotSupported { ref source, .. })
                        if source.to_string().contains("Deleting rows from the target table when there is no match in the source table is not supported when the source data has a different schema than the target data"),
                ),
                "Expected NotSupported error, got: {:?}",
                res
            );
        }

        #[tokio::test]
        async fn test_errors_on_bad_schema() {
            let Fixtures { ds, new_data } = setup(false).await;

            // Schema with different names, which should be rejected.
            let bad_schema = Arc::new(Schema::new(vec![
                Field::new("wrong_key", DataType::Utf8, false),
                Field::new("wrong_value", DataType::UInt32, false),
            ]));

            // Should reject when data is not a subschema.
            let bad_batch =
                RecordBatch::try_new(bad_schema.clone(), new_data.columns().to_vec()).unwrap();
            let reader = Box::new(RecordBatchIterator::new([Ok(bad_batch)], bad_schema));

            let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing)
                .try_build()
                .unwrap();
            let res = job.execute_reader(reader).await;
            assert!(
                matches!(
                    &res,
                    &Err(Error::SchemaMismatch { ref difference, .. })
                        if difference.to_string().contains("fields did not match")
                ),
                "Expected SchemaMismatch error, got: {:?}",
                res
            );
        }

        #[rstest]
        #[tokio::test]
        async fn test_merge_insert_subcols(
            #[values(false, true)] scalar_index: bool,
            #[values(false, true)] insert: bool,
        ) {
            let Fixtures { ds, new_data } = setup(scalar_index).await;
            let reader = Box::new(RecordBatchIterator::new(
                [Ok(new_data.clone())],
                new_data.schema(),
            ));
            let fragments_before = ds
                .get_fragments()
                .iter()
                .map(|f| f.metadata().clone())
                .collect::<Vec<_>>();
            let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(if insert {
                    WhenNotMatched::InsertAll
                } else {
                    WhenNotMatched::DoNothing
                })
                .try_build()
                .unwrap();

            let (ds, stats) = job.execute_reader(reader).await.unwrap();

            // Should not rewrite the affected data files
            let fragments_after = ds
                .get_fragments()
                .iter()
                .map(|f| f.metadata().clone())
                .collect::<Vec<_>>();
            assert_eq!(
                fragments_before.iter().map(|f| f.id).collect::<Vec<_>>(),
                fragments_after
                    .iter()
                    .take(fragments_before.len())
                    .map(|f| f.id)
                    .collect::<Vec<_>>()
            );
            // Only the second and third fragment should be different.
            assert_eq!(fragments_before[0], fragments_after[0]);
            assert_ne!(fragments_before[1], fragments_after[1]);
            assert_ne!(fragments_before[2], fragments_after[2]);
            assert_eq!(fragments_before[3], fragments_after[3]);

            let has_added_files = |frag: &Fragment| {
                assert_eq!(frag.files.len(), 2);
                let data_files = &frag.files;
                // Updated columns should be only columns in new data files
                // -2 field ids are tombstoned.
                assert_eq!(&data_files[0].fields, &[0, -2, -2]);
                assert_eq!(&data_files[1].fields, &[2, 1]);
            };
            has_added_files(&fragments_after[1]);
            has_added_files(&fragments_after[2]);

            if insert {
                assert_eq!(fragments_after.len(), 5);
                assert_eq!(stats.num_inserted_rows, 2);
            } else {
                assert_eq!(fragments_after.len(), 4);
                assert_eq!(stats.num_inserted_rows, 0);
            }

            assert_eq!(stats.num_updated_rows, (new_data.num_rows() - 2) as u64);
            assert_eq!(stats.num_deleted_rows, 0);

            let data = ds
                .scan()
                .scan_in_order(true)
                .try_into_batch()
                .await
                .unwrap();
            assert_eq!(data.num_rows(), if insert { 1024 + 2 } else { 1024 });
            assert_eq!(data.num_columns(), 3);

            let values = data
                .column(1)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();
            assert_eq!(values.value(0), 0);
            assert_eq!(values.value(256), 1024);
            assert_eq!(values.value(512), 512);
            assert_eq!(values.value(715), 1024 + new_data.num_rows() as u32 - 3);
            if insert {
                assert_eq!(values.value(1024), 1024 + new_data.num_rows() as u32 - 2);
            }
        }
    }

    // For some reason, Windows isn't able to handle the timeout test. Possibly
    // a performance bug in their timer implementation?
    #[cfg(not(windows))]
    #[rstest::rstest]
    #[case::all_success(Duration::from_secs(100_000))]
    #[case::timeout(Duration::from_millis(200))]
    #[tokio::test]
    async fn test_merge_insert_concurrency(#[case] timeout: Duration) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("value", DataType::UInt32, false),
        ]));
        // To benchmark scaling curve: measure how long to run
        //
        // And vary `concurrency` to see how it scales. Compare this again `main`.
        let concurrency = 10;
        let initial_data = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..concurrency)),
                Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(
                    0,
                    concurrency as usize,
                ))),
            ],
        )
        .unwrap();

        // Increase likelihood of contention by throttling the store
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                // For benchmarking: Increase this to simulate object storage.
                wait_list_per_call: Duration::from_millis(20),
                wait_get_per_call: Duration::from_millis(20),
                wait_put_per_call: Duration::from_millis(20),
                ..Default::default()
            },
        });
        let session = Arc::new(Session::default());

        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(throttled.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                ..Default::default()
            })
            .execute(vec![initial_data])
            .await
            .unwrap();

        // do merge inserts in parallel based on the concurrency. Each will open the dataset,
        // signal they have opened, and then wait for a signal to proceed. Once the signal
        // is received, they will do a merge insert and close the dataset.

        let barrier = Arc::new(Barrier::new(concurrency as usize));
        let mut handles = Vec::new();
        for i in 0..concurrency {
            let session_ref = session.clone();
            let schema_ref = schema.clone();
            let barrier_ref = barrier.clone();
            let throttled_ref = throttled.clone();
            let handle = tokio::task::spawn(async move {
                let dataset = DatasetBuilder::from_uri("memory://")
                    .with_read_params(ReadParams {
                        store_options: Some(ObjectStoreParams {
                            object_store_wrapper: Some(throttled_ref.clone()),
                            ..Default::default()
                        }),
                        session: Some(session_ref.clone()),
                        ..Default::default()
                    })
                    .load()
                    .await
                    .unwrap();
                let dataset = Arc::new(dataset);

                let new_data = RecordBatch::try_new(
                    schema_ref.clone(),
                    vec![
                        Arc::new(UInt32Array::from(vec![i])),
                        Arc::new(UInt32Array::from(vec![1])),
                    ],
                )
                .unwrap();
                let source = Box::new(RecordBatchIterator::new([Ok(new_data)], schema_ref.clone()));

                let job = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
                    .unwrap()
                    .when_matched(WhenMatched::UpdateAll)
                    .when_not_matched(WhenNotMatched::InsertAll)
                    .conflict_retries(100)
                    .retry_timeout(timeout)
                    .try_build()
                    .unwrap();
                barrier_ref.wait().await;

                job.execute_reader(source)
                    .await
                    .map(|(_ds, stats)| stats.num_attempts)
            });
            handles.push(handle);
        }

        let results = try_join_all(handles).await.unwrap();

        for attempts in results.iter() {
            match attempts {
                Ok(attempts) => {
                    assert!(*attempts <= 10, "Attempt count should be <= 10");
                }
                Err(err) => {
                    // If we get an error, it means the task was cancelled
                    // due to timeout. This is expected if the timeout is
                    // set to a low value.
                    assert!(
                        matches!(err, Error::TooMuchWriteContention { message, .. } if message.contains("failed on retry_timeout")),
                        "Expected TooMuchWriteContention error, got: {:?}",
                        err
                    );
                }
            }
        }

        if timeout.as_secs() > 10 {
            dataset.checkout_latest().await.unwrap();
            let batches = dataset.scan().try_into_batch().await.unwrap();

            let values = batches["value"].as_primitive::<UInt32Type>();
            assert!(
                values.values().iter().all(|&v| v == 1),
                "All values should be 1 after merge insert. Got: {:?}",
                values
            );
        }
    }

    #[tokio::test]
    async fn test_merge_insert_large_concurrent() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("value", DataType::UInt32, false),
        ]));
        let num_rows = 10;
        let initial_data = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..num_rows)),
                Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(
                    0,
                    num_rows as usize,
                ))),
            ],
        )
        .unwrap();

        // Adding latency helps ensure we get contention
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_call: Duration::from_millis(10),
                wait_get_per_call: Duration::from_millis(10),
                ..Default::default()
            },
        });
        let session = Arc::new(Session::default());

        let dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(throttled.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                ..Default::default()
            })
            .execute(vec![initial_data])
            .await
            .unwrap();
        let dataset = Arc::new(dataset);

        // Start one merge insert, but don't commit it yet.
        let new_data1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![1])),
                Arc::new(UInt32Array::from(vec![1])),
            ],
        )
        .unwrap();
        let UncommittedMergeInsert {
            transaction: transaction1,
            ..
        } = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_uncommitted(RecordBatchIterator::new(
                vec![Ok(new_data1)],
                schema.clone(),
            ))
            .await
            .unwrap();

        // Setup a "large" merge insert, with many batches
        let new_data2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..1000)),
                Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(2, 1000))),
            ],
        )
        .unwrap();
        let notify = Arc::new(Notify::new());
        let source = RecordBatchIterator::new(
            (0..10)
                .map(|i| {
                    let batch = new_data2.slice(i * 100, 100);
                    if i == 9 {
                        notify.notify_one();
                    }
                    Ok(batch)
                })
                .collect::<Vec<_>>(),
            schema.clone(),
        );
        let dataset2 = DatasetBuilder::from_uri("memory://")
            .with_read_params(ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(throttled.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();
        let job = MergeInsertBuilder::try_new(Arc::new(dataset2), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_reader(source);
        let task = tokio::task::spawn(job);

        // Right as the large merge insert has finished reading the last batch,
        // we will commit the first merge insert. This should trigger a conflict,
        // but we should resolve it automatically.
        notify.notified().await;
        let mut dataset = CommitBuilder::new(dataset)
            .execute(transaction1)
            .await
            .unwrap();

        task.await.unwrap().unwrap();
        dataset.checkout_latest().await.unwrap();

        let batches = dataset.scan().try_into_batch().await.unwrap();
        let values = batches["value"].as_primitive::<UInt32Type>();
        assert!(
            values.values().iter().all(|&v| v == 2),
            "All values should be 1 after merge insert. Got: {:?}",
            values
        );
    }

    #[tokio::test]
    async fn test_merge_insert_updates_indices() {
        let test_dataset = async || {
            let mut dataset = lance_datagen::gen()
                .col("id", array::step::<UInt32Type>())
                .col("value", array::step::<UInt32Type>())
                .col("other_value", array::step::<UInt32Type>())
                .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(20))
                .await
                .unwrap();

            dataset
                .create_index(
                    &["id"],
                    IndexType::BTree,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
            dataset
                .create_index(
                    &["value"],
                    IndexType::BTree,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
            dataset
                .create_index(
                    &["other_value"],
                    IndexType::BTree,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
            Arc::new(dataset)
        };

        let check_indices = async |dataset: &Dataset, id_frags: &[u32], value_frags: &[u32]| {
            let id_index = dataset
                .load_scalar_index(ScalarIndexCriteria::default().with_name("id_idx"))
                .await
                .unwrap();

            if id_frags.is_empty() {
                assert!(id_index.is_none());
            } else {
                let id_index = id_index.unwrap();
                let id_frags_bitmap = RoaringBitmap::from_iter(id_frags.iter().copied());
                assert_eq!(id_index.fragment_bitmap.unwrap(), id_frags_bitmap);
            }

            let value_index = dataset
                .load_scalar_index(ScalarIndexCriteria::default().with_name("value_idx"))
                .await
                .unwrap();

            if value_frags.is_empty() {
                assert!(value_index.is_none());
            } else {
                let value_index = value_index.unwrap();
                let value_frags_bitmap = RoaringBitmap::from_iter(value_frags.iter().copied());
                assert_eq!(value_index.fragment_bitmap.unwrap(), value_frags_bitmap);
            }

            let other_value_index = dataset
                .load_scalar_index(ScalarIndexCriteria::default().with_name("other_value_idx"))
                .await
                .unwrap()
                .unwrap();

            // We should never remove any fragments from the other_value index.
            assert_eq!(other_value_index.fragment_bitmap.as_ref().unwrap().len(), 4);
        };

        let dataset = test_dataset().await;

        // Sanity test on the initial dataset
        check_indices(&dataset, &[0, 1, 2, 3], &[0, 1, 2, 3]).await;

        // Vertical merge insert (full schema), one fragment is deleted and should be removed from
        // the index.
        let merge_insert = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap();

        let (dataset, _) = merge_insert
            .execute_reader(
                lance_datagen::gen()
                    .col("id", array::step_custom::<UInt32Type>(50, 1))
                    .col("value", array::step_custom::<UInt32Type>(50, 1))
                    .col("other_value", array::step_custom::<UInt32Type>(50, 1))
                    .into_df_stream(RowCount::from(40), BatchCount::from(1)),
            )
            .await
            .unwrap();

        // Fragment 3 removed but we don't remove it from the index bitmap since it is
        // fully deleted and can't be searched anymore anyways.
        check_indices(&dataset, &[0, 1, 2, 3], &[0, 1, 2, 3]).await;

        // Now we do the same thing with a partial merge insert (only id and value)
        let dataset = test_dataset().await;

        // Vertical merge insert (full schema), one fragment is deleted and should be removed from
        // the index.
        let merge_insert = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap();

        let (dataset, _) = merge_insert
            .execute_reader(
                lance_datagen::gen()
                    .col("id", array::step_custom::<UInt32Type>(50, 1))
                    .col("value", array::step_custom::<UInt32Type>(50, 1))
                    .into_df_stream(RowCount::from(40), BatchCount::from(1)),
            )
            .await
            .unwrap();

        // Fragment 3 is fully removed.  We could keep it technically but today it is removed
        // which is also fine.  Fragment 2 is partially and must be removed.
        //
        // TODO: We should not be modifying the id_index here.  A merge_insert should not need
        // to rewrite the id field.  However, it seems we are doing that today.  This should be
        // fixed in
        check_indices(&dataset, &[0, 1], &[0, 1]).await;

        // One more test but this time we touch all fragments which causes the index to be removed
        // entirely.
        let dataset = test_dataset().await;

        // Vertical merge insert (full schema), one fragment is deleted and should be removed from
        // the index.
        let merge_insert = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap();

        let (dataset, _) = merge_insert
            .execute_reader(
                lance_datagen::gen()
                    .col("id", array::step_custom::<UInt32Type>(10, 1))
                    .col("value", array::step_custom::<UInt32Type>(10, 1))
                    .into_df_stream(RowCount::from(80), BatchCount::from(1)),
            )
            .await
            .unwrap();

        check_indices(&dataset, &[], &[]).await;
    }

    #[tokio::test]
    async fn test_upsert_concurrent_full_frag() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("value", DataType::UInt32, false),
        ]));
        let initial_data = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![0, 1])),
                Arc::new(UInt32Array::from(vec![0, 0])),
            ],
        )
        .unwrap();

        // Increase likelihood of contention by throttling the store
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_call: Duration::from_millis(5),
                wait_get_per_call: Duration::from_millis(5),
                wait_put_per_call: Duration::from_millis(5),
                ..Default::default()
            },
        });
        let session = Arc::new(Session::default());

        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(throttled.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                ..Default::default()
            })
            .execute(vec![initial_data])
            .await
            .unwrap();

        // Each merge insert will update one row. Combined, they should delete
        // all rows in the first fragment, and it should be dropped.
        let barrier = Arc::new(Barrier::new(2));
        let mut handles = Vec::new();
        for i in 0..2 {
            let new_data = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(UInt32Array::from(vec![i])),
                    Arc::new(UInt32Array::from(vec![1])),
                ],
            )
            .unwrap();
            let source = Box::new(RecordBatchIterator::new([Ok(new_data)], schema.clone()));

            let dataset_ref = Arc::new(dataset.clone());
            let barrier = barrier.clone();
            let handle = tokio::spawn(async move {
                barrier.wait().await;
                MergeInsertBuilder::try_new(dataset_ref, vec!["id".to_string()])
                    .unwrap()
                    .when_matched(WhenMatched::UpdateAll)
                    .when_not_matched(WhenNotMatched::InsertAll)
                    .try_build()
                    .unwrap()
                    .execute_reader(source)
                    .await
                    .unwrap();
            });
            handles.push(handle);
        }
        try_join_all(handles).await.unwrap();

        dataset.checkout_latest().await.unwrap();
        assert!(
            dataset
                .get_fragments()
                .iter()
                .all(|f| f.metadata().num_rows().unwrap() > 0),
            "No fragments should have zero rows after upsert"
        );

        let batches = dataset.scan().try_into_batch().await.unwrap();
        let values = batches["value"].as_primitive::<UInt32Type>();
        assert!(
            values.values().iter().all(|&v| v == 1),
            "All values should be 1 after merge insert. Got: {:?}",
            values
        );
    }

    #[tokio::test]
    async fn test_plan_upsert() {
        let data = lance_datagen::gen()
            .with_seed(Seed::from(1))
            .col("value", array::step::<UInt32Type>())
            .col("key", array::rand_pseudo_uuid_hex());
        let data = data.into_reader_rows(RowCount::from(1024), BatchCount::from(32));
        let _schema = data.schema();

        // Create dataset with initial data
        let ds = Dataset::write(data, "memory://", None).await.unwrap();

        // Create upsert job
        let merge_insert_job = crate::dataset::MergeInsertBuilder::try_new(
            Arc::new(ds),
            vec!["key".to_string()],
        )
        .unwrap()
        .when_matched(crate::dataset::WhenMatched::UpdateAll)
        .try_build()
        .unwrap();

        // Create new data for upsert
        let new_data = lance_datagen::gen()
            .with_seed(Seed::from(2))
            .col("value", array::step::<UInt32Type>())
            .col("key", array::rand_pseudo_uuid_hex());
        let new_data = new_data.into_reader_rows(RowCount::from(512), BatchCount::from(16));
        let new_data_stream = reader_to_stream(Box::new(new_data));

        let plan = merge_insert_job.create_plan(new_data_stream).await.unwrap();

        // Print the plan using DisplayableExecutionPlan
        use datafusion::physical_plan::displayable;
        let display = displayable(plan.as_ref());
        let plan_str = format!("{}", display.indent(false));

        // Assert the plan structure - should contain HashJoin, Projection, and MergeInsertWrite
        assert!(plan_str.contains("HashJoinExec"));
        assert!(plan_str.contains("ProjectionExec"));
        // The exact plan string can vary, so we check for key components
        println!("Plan:\n{}", plan_str);
    }
}
