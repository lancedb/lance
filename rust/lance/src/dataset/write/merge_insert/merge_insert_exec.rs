// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;

use super::MergeInsertParams;

/// Inserts new rows and updates existing rows in the target table.
///
/// This does the actual write.
///
/// This is implemented by moving updated rows to new fragments. This mode
/// is most optimal when updating the full schema.
///
/// It returns a single batch, containing the statistics.
struct FullSchemaMergeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    params: MergeInsertParams,
}

/// Inserts new rows and updates existing rows in the target table.
///
/// This does the actual write.
///
/// This is implemented by doing updates by writing new data files and apending
/// them to fragments in place. This mode is most optimal when updating a subset
/// of columns, particularly when the columns not being updated are expensive to
/// rewrite.
///
/// It returns a single batch, containing the statistics.
struct PartialUpdateMergeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    params: MergeInsertParams,
}
