// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared primary-key helpers for the LSM scanner execution nodes.
//!
//! Centralizes PK column resolution and per-row hashing so that every dedup
//! node ([`super::WithinSourceDedupExec`] and [`super::LsmGlobalPkDedupExec`])
//! resolves and hashes a primary key the same way. The row hash is kept
//! consistent with the variants supported by [`super::compute_pk_hash_from_scalars`]
//! so a single PK produces the same hash regardless of which exec consumes it.

use arrow_array::{Array, RecordBatch};
use datafusion::error::{DataFusionError, Result as DFResult};

/// Resolve the column index of each primary-key column in `batch`.
pub fn resolve_pk_indices(batch: &RecordBatch, pk_columns: &[String]) -> DFResult<Vec<usize>> {
    pk_columns
        .iter()
        .map(|col| {
            batch
                .schema()
                .column_with_name(col)
                .map(|(idx, _)| idx)
                .ok_or_else(|| {
                    DataFusionError::Internal(format!("Primary key column '{}' not found", col))
                })
        })
        .collect()
}

/// Hash a single row's primary key, identified by the `pk_indices` column
/// positions and `row_idx`.
pub fn compute_pk_hash(batch: &RecordBatch, pk_indices: &[usize], row_idx: usize) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for &col_idx in pk_indices {
        let col = batch.column(col_idx);
        let is_null = col.is_null(row_idx);
        is_null.hash(&mut hasher);

        if !is_null {
            if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int32Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int64Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::BinaryArray>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::UInt32Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::UInt64Array>() {
                arr.value(row_idx).hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}
