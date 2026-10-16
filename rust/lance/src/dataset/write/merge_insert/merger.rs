// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{
    InsertedKeyTracker, MergeInsertParams, MergeStats, SourceDedupeBehavior, WhenMatched,
    WhenNotMatchedBySource, create_duplicate_row_error, format_key_values_on_columns,
};
use crate::dataset::utils::CapturedRowIds;
use crate::io::exec::Planner;
use arrow_array::{
    BooleanArray, RecordBatch, StructArray, UInt32Array, cast::AsArray, types::UInt64Type,
};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::take::take_record_batch;
use datafusion::error::DataFusionError;
use datafusion::{
    physical_plan::{ColumnarValue, PhysicalExpr},
    scalar::ScalarValue,
};
use futures::{
    Stream,
    stream::{self},
};
use lance_arrow::SchemaExt;
use lance_core::{Error, ROW_ADDR_FIELD, Result};
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

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

// A sync-safe structure that is shared by all of the "process batch" tasks.
//
// Note: we are not currently using parallelism but this still needs to be sync because it is
//       held across an await boundary (and we might use parallelism someday)
#[derive(Debug, Clone)]
pub(super) struct Merger {
    // As the merger runs it will update the list of deleted rows
    pub(super) deleted_rows: Arc<Mutex<Vec<u64>>>,
    // Shared collection to capture row ids that need to be updated
    pub(super) updating_row_ids: Arc<Mutex<CapturedRowIds>>,
    // Physical delete expression, only set if params.delete_not_matched_by_source is DeleteIf
    delete_expr: Option<Arc<dyn PhysicalExpr>>,
    // User statistics for merging
    pub(super) merge_stats: Arc<Mutex<MergeStats>>,
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
    /// Whether to enable stable row ids
    enable_stable_row_ids: bool,
    /// Set to track processed row IDs to detect duplicates
    processed_row_ids: Arc<Mutex<HashSet<u64>>>,
    /// Set to track non-null keys of rows inserted by FirstSeen mode
    processed_insert_keys: Arc<Mutex<InsertedKeyTracker>>,
}

impl Merger {
    // Creates a new merger with an empty set of deleted rows, compiles expressions, if present
    pub(super) fn try_new(
        params: MergeInsertParams,
        schema: Arc<Schema>,
        with_row_addr: bool,
        enable_stable_row_ids: bool,
    ) -> Result<Self> {
        let delete_expr = if let WhenNotMatchedBySource::DeleteIf(expr) =
            &params.delete_not_matched_by_source
        {
            let planner = Planner::new(schema.clone());
            let expr = planner.optimize_expr(expr.clone())?;
            let physical_expr = planner.create_physical_expr(&expr)?;
            let data_type = physical_expr.data_type(&schema)?;
            if data_type != DataType::Boolean {
                return Err(Error::invalid_input(format!(
                    "Merge insert conditions must be expressions that return a boolean value, received expression ({}) which has data type {}",
                    expr, data_type
                )));
            }
            Some(physical_expr)
        } else {
            None
        };
        let match_filter_expr = match &params.when_matched {
            WhenMatched::UpdateIf(_) | WhenMatched::UpdateIfExpr(_) => {
                let combined_schema = Arc::new(combined_schema(&schema));
                let planner = Planner::new(combined_schema.clone());
                let expr = match &params.when_matched {
                    WhenMatched::UpdateIf(expr_str) => planner.parse_filter(expr_str)?,
                    WhenMatched::UpdateIfExpr(expr) => expr.clone(),
                    _ => unreachable!(),
                };
                let expr = planner.optimize_expr(expr)?;
                let match_expr = planner.create_physical_expr(&expr)?;
                let data_type = match_expr.data_type(combined_schema.as_ref())?;
                if data_type != DataType::Boolean {
                    return Err(Error::invalid_input(format!(
                        "Merge insert conditions must be expressions that return a boolean value, received a 'when matched update if' expression ({}) which has data type {}",
                        expr, data_type
                    )));
                }
                Some(match_expr)
            }
            _ => None,
        };
        let output_schema = if with_row_addr {
            Arc::new(schema.try_with_column(ROW_ADDR_FIELD.clone())?)
        } else {
            schema.clone()
        };

        Ok(Self {
            deleted_rows: Arc::new(Mutex::new(Vec::new())),
            updating_row_ids: Arc::new(Mutex::new(CapturedRowIds::new(enable_stable_row_ids))),
            delete_expr,
            merge_stats: Arc::new(Mutex::new(MergeStats::default())),
            match_filter_expr,
            params,
            schema,
            with_row_addr,
            output_schema,
            enable_stable_row_ids,
            processed_row_ids: Arc::new(Mutex::new(HashSet::new())),
            processed_insert_keys: Arc::new(Mutex::new(InsertedKeyTracker::default())),
        })
    }

    pub(super) fn output_schema(&self) -> &Arc<Schema> {
        &self.output_schema
    }

    // Retrieves a bitmap of rows where at least one of the given columns is
    // not null.
    fn not_all_null(batch: &RecordBatch, cols: &[usize]) -> Result<BooleanArray> {
        // For our purposes we know there is always at least 1 on key
        debug_assert!(!cols.is_empty());
        let mut at_least_one_valid = arrow::compute::is_not_null(batch.column(cols[0]))?;
        for &idx in &cols[1..] {
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
    ) -> Result<(BooleanArray, BooleanArray, BooleanArray)> {
        // The outer join distinguishes its three cases by which side's join
        // keys were NULL-padded: a present row always has non-null keys, while
        // the absent side is filled with NULLs. We therefore test the *key*
        // columns, located by name. They are NOT necessarily the first
        // `num_keys` columns — a partial-schema source can place a payload
        // column (e.g. an all-null vector) at position 0, and checking
        // positions [0, num_keys) there misreads an all-null leading payload
        // column as an absent join side, silently dropping every matched row
        // (https://github.com/lancedb/lancedb/issues/3515). The target half is
        // resolved independently because the indexed path keeps dataset-schema
        // order even when the source fields are reordered. Restrict that lookup
        // to the target half because a valid source field can have the same
        // `target_`-prefixed name.
        let combined_schema = combined_batch.schema();
        let source_key_cols = self
            .params
            .on
            .iter()
            .map(|key| {
                combined_schema.index_of(key).map_err(|_| {
                    Error::internal(format!(
                        "merge insert key column '{}' not found in joined batch",
                        key
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let target_key_cols = self
            .params
            .on
            .iter()
            .map(|key| {
                let target_key = format!("target_{key}");
                combined_schema
                    .fields()
                    .iter()
                    .enumerate()
                    .skip(right_offset)
                    .find_map(|(index, field)| (field.name() == &target_key).then_some(index))
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "merge insert target key column '{}' not found in joined batch",
                            target_key
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        let in_left = Self::not_all_null(combined_batch, &source_key_cols)?;
        let in_right = Self::not_all_null(combined_batch, &target_key_cols)?;
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
    pub(super) async fn execute_batch(
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

        let left_cols = Vec::from_iter(0..right_offset);
        let right_cols_with_id = Vec::from_iter(right_offset..num_fields);

        let mut batches = Vec::with_capacity(2);
        let (left_only, in_both, right_only) = self.extract_selections(&batch, right_offset)?;

        // There is no contention on this mutex.  We're only using it to bypass the rust
        // borrow checker (the stream needs to be `sync` since it crosses an await point)
        let mut deleted_row_ids = self.deleted_rows.lock().unwrap();

        // Each `WhenMatched` variant handles the matched rows (`in_both`)
        // differently.
        let match_filter_expr = self.match_filter_expr;
        match &self.params.when_matched {
            WhenMatched::DoNothing => {}
            WhenMatched::Delete => {
                // Matched rows are removed, not rewritten: record their row ids
                // for the commit to delete and emit no replacement batch. A
                // source with duplicate keys matches the same target row more
                // than once; apply the same `source_dedupe_behavior` policy as
                // updates so a duplicate either aborts (`Fail`) or is skipped
                // and counted once (`FirstSeen`) — the commit deletes the row a
                // single time regardless.
                let matched = arrow::compute::filter_record_batch(&batch, &in_both)?;
                let row_ids = matched.column(row_id_col).as_primitive::<UInt64Type>();

                let mut processed_row_ids = self.processed_row_ids.lock().unwrap();
                for (row_idx, &row_id) in row_ids.values().iter().enumerate() {
                    if processed_row_ids.insert(row_id) {
                        merge_statistics.num_deleted_rows += 1;
                        deleted_row_ids.push(row_id);
                    } else {
                        match self.params.source_dedupe_behavior {
                            SourceDedupeBehavior::Fail => {
                                return Err(create_duplicate_row_error(
                                    &matched,
                                    row_idx,
                                    &self.params.on,
                                ));
                            }
                            SourceDedupeBehavior::FirstSeen => {
                                merge_statistics.num_skipped_duplicates += 1;
                            }
                        }
                    }
                }
            }
            WhenMatched::Fail => {
                // Any matched row aborts the whole operation.
                if let Some(row_idx) = (0..in_both.len()).find(|&i| in_both.value(i)) {
                    return Err(DataFusionError::Execution(format!(
                        "Merge insert failed: found matching row with key values: {}",
                        format_key_values_on_columns(&batch, row_idx, &self.params.on)
                    )));
                }
            }
            WhenMatched::UpdateAll | WhenMatched::UpdateIf(_) | WhenMatched::UpdateIfExpr(_) => {
                let mut matched = arrow::compute::filter_record_batch(&batch, &in_both)?;

                if let Some(match_filter) = match_filter_expr {
                    let unzipped = unzip_batch(&matched, &self.schema);
                    let filtered = match_filter.evaluate(&unzipped)?;
                    match filtered {
                        ColumnarValue::Array(mask) => {
                            // Some rows matched, filter down and replace those rows
                            matched =
                                arrow::compute::filter_record_batch(&matched, mask.as_boolean())?;
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

                    let mut processed_row_ids = self.processed_row_ids.lock().unwrap();
                    let mut keep_indices: Vec<u32> = Vec::with_capacity(matched.num_rows());
                    for (row_idx, &row_id) in row_ids.values().iter().enumerate() {
                        if processed_row_ids.insert(row_id) {
                            keep_indices.push(row_idx as u32);
                        } else {
                            match self.params.source_dedupe_behavior {
                                SourceDedupeBehavior::Fail => {
                                    return Err(create_duplicate_row_error(
                                        &matched,
                                        row_idx,
                                        &self.params.on,
                                    ));
                                }
                                SourceDedupeBehavior::FirstSeen => {
                                    // Skip this duplicate row (don't add to keep_indices)
                                }
                            }
                        }
                    }
                    drop(processed_row_ids);

                    // Filter out duplicate rows if any were skipped
                    let num_skipped = matched.num_rows() - keep_indices.len();
                    if num_skipped > 0 {
                        merge_statistics.num_skipped_duplicates += num_skipped as u64;
                        merge_statistics.num_updated_rows -= num_skipped as u64;

                        let indices = UInt32Array::from(keep_indices);
                        matched = take_record_batch(&matched, &indices)?;
                    }

                    // Only process and write if there are remaining rows after filtering duplicates
                    if matched.num_rows() > 0 {
                        // Get row_ids again after filtering (if any duplicates were removed)
                        let row_ids = matched.column(row_id_col).as_primitive::<UInt64Type>();
                        deleted_row_ids.extend(row_ids.values());
                        if self.enable_stable_row_ids {
                            self.updating_row_ids
                                .lock()
                                .unwrap()
                                .capture(row_ids.values())?;
                        }

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
            }
        }
        if self.params.insert_not_matched {
            let mut not_matched = arrow::compute::filter_record_batch(&batch, &left_only)?;
            if self.params.source_dedupe_behavior == SourceDedupeBehavior::FirstSeen {
                let mut processed_insert_keys = self.processed_insert_keys.lock().unwrap();
                let mut keep_indices = Vec::with_capacity(not_matched.num_rows());
                for row_idx in 0..not_matched.num_rows() {
                    if processed_insert_keys.insert(&not_matched, row_idx, &self.params.on)? {
                        keep_indices.push(row_idx as u32);
                    } else {
                        merge_statistics.num_skipped_duplicates += 1;
                    }
                }
                drop(processed_insert_keys);

                if keep_indices.len() != not_matched.num_rows() {
                    not_matched =
                        take_record_batch(&not_matched, &UInt32Array::from(keep_indices))?;
                }
            }
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
