// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Projection helpers shared across LSM scanner planners.
//!
//! Vector search, point lookup, and the general scan planner all face the
//! same projection-handling concerns:
//!
//! - The active-arm `MemTableScanner` only special-cases `_rowid` in its
//!   `project()` method. Other system columns (`_rowaddr`, `_rowoffset`,
//!   `_row_*_at_version`) are treated as missing data columns and produce
//!   `Column not found in schema` errors.
//! - Cross-LSM-source values for system columns are not meaningful: a
//!   `_rowid` of 5 in the base table and a `_rowid` of 5 in a flushed
//!   memtable refer to different rows, so concatenating them across the
//!   union is misleading.
//!
//! These helpers give every planner a single source of truth:
//!
//! - [`build_scanner_projection`] — what we pass to underlying scanners.
//!   Strips system columns and `_distance` from the user's projection
//!   (they are handled separately or auto-generated) and appends PK
//!   columns for downstream dedup / staleness detection.
//! - [`canonical_output_schema`] — the final output schema. Honors the
//!   user's column order, exposes requested system columns at their
//!   requested position as nullable `UInt64`, and (optionally) ensures
//!   `_distance` is present for KNN.
//! - [`project_to_canonical`] — wraps an exec plan in a [`ProjectionExec`]
//!   that emits `target_schema` exactly. System columns missing from the
//!   source are filled with NULL literals so the not-comparable semantics
//!   are explicit at the output rather than producing garbage values.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{Column, Literal};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::scalar::ScalarValue;
use lance_core::{Result, is_system_column};

/// Column name for distance in vector search results.
pub const DISTANCE_COLUMN: &str = "_distance";

/// Whether a column is auto-managed by LSM scanner planners and therefore
/// must not be forwarded to the underlying scanner's `project()` call.
fn is_auto_managed(col: &str) -> bool {
    col == DISTANCE_COLUMN || is_system_column(col)
}

/// Build the projection list to pass to underlying scanners.
///
/// - Strips system columns (`_rowid`, `_rowaddr`, …) and `_distance`.
///   These are auto-managed (scanner-side `with_row_id`/`with_row_address`
///   flags, KNN `_distance` synthesis, or NULL fill in the canonical
///   projection), so passing them through `scanner.project()` would error
///   in the active-arm `MemTableScanner`.
/// - Appends every PK column not already present, so downstream dedup /
///   staleness detection always has a hash key.
/// - When the user's projection is `None`, defaults to all base-schema
///   data columns.
pub fn build_scanner_projection(
    user_projection: Option<&[String]>,
    base_schema: &SchemaRef,
    pk_columns: &[String],
) -> Vec<String> {
    let mut cols: Vec<String> = if let Some(p) = user_projection {
        p.iter().filter(|c| !is_auto_managed(c)).cloned().collect()
    } else {
        base_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    };

    for pk in pk_columns {
        if !cols.contains(pk) {
            cols.push(pk.clone());
        }
    }

    cols
}

/// Build the canonical output schema, honoring user-specified column order.
///
/// - User-listed columns appear at their original index.
/// - System columns (`_rowid`, `_rowaddr`, …) are kept as nullable
///   `UInt64` at the user's requested position; their values are filled
///   with NULL across LSM sources by [`project_to_canonical`] because
///   per-source values aren't comparable.
/// - When `include_distance` is true, `_distance` is exposed as nullable
///   `Float32` at the user's position, or appended to the end if absent.
/// - PK columns are appended (after the user projection, before any
///   auto-appended `_distance`) when not already present, to match the
///   data flowing through `build_scanner_projection`.
/// - Any user-projected name that isn't in `base_schema` and isn't a
///   recognized system / `_distance` column is silently dropped, matching
///   prior behavior of the per-planner helpers.
pub fn canonical_output_schema(
    user_projection: Option<&[String]>,
    base_schema: &SchemaRef,
    pk_columns: &[String],
    include_distance: bool,
) -> SchemaRef {
    let mut ordered: Vec<String> = if let Some(p) = user_projection {
        p.to_vec()
    } else {
        base_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    };

    for pk in pk_columns {
        if !ordered.contains(pk) {
            ordered.push(pk.clone());
        }
    }

    if include_distance && !ordered.iter().any(|c| c == DISTANCE_COLUMN) {
        ordered.push(DISTANCE_COLUMN.to_string());
    }

    let fields: Vec<Arc<Field>> = ordered
        .iter()
        .filter_map(|name| {
            if name == DISTANCE_COLUMN {
                include_distance
                    .then(|| Arc::new(Field::new(DISTANCE_COLUMN, DataType::Float32, true)))
            } else if is_system_column(name) {
                Some(Arc::new(Field::new(name.clone(), DataType::UInt64, true)))
            } else {
                base_schema
                    .field_with_name(name)
                    .ok()
                    .map(|f| Arc::new(f.clone()))
            }
        })
        .collect();

    Arc::new(Schema::new(fields))
}

/// Wrap `plan` with a [`ProjectionExec`] that emits exactly `target_schema`.
///
/// - Columns present in both source and target are forwarded by name.
/// - System columns and `_distance` present in the target but missing from
///   the source are filled with typed NULL literals (UInt64 / Float32),
///   reflecting the fact that cross-LSM values for these columns are not
///   meaningful or were not produced by every source.
/// - Any other missing target column is an internal error (the planner
///   built a target schema that the source can't satisfy).
pub fn project_to_canonical(
    plan: Arc<dyn ExecutionPlan>,
    target_schema: &SchemaRef,
) -> Result<Arc<dyn ExecutionPlan>> {
    let input_schema = plan.schema();
    let mut project_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
        Vec::with_capacity(target_schema.fields().len());
    for field in target_schema.fields() {
        let name = field.name();
        let expr: Arc<dyn PhysicalExpr> = match input_schema.column_with_name(name) {
            Some((idx, _)) => Arc::new(Column::new(name, idx)),
            None if is_system_column(name) => Arc::new(Literal::new(ScalarValue::UInt64(None))),
            None if name == DISTANCE_COLUMN => Arc::new(Literal::new(ScalarValue::Float32(None))),
            None => {
                return Err(lance_core::Error::internal(format!(
                    "Column '{}' missing from canonical projection source schema (have: {:?})",
                    name,
                    input_schema
                        .fields()
                        .iter()
                        .map(|f| f.name().clone())
                        .collect::<Vec<_>>()
                )));
            }
        };
        project_exprs.push((expr, name.clone()));
    }
    let projection_exec = ProjectionExec::try_new(project_exprs, plan).map_err(|e| {
        lance_core::Error::internal(format!("Failed to build canonical ProjectionExec: {}", e))
    })?;
    Ok(Arc::new(projection_exec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Schema as ArrowSchema;

    fn schema() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("vector", DataType::Float32, true),
        ]))
    }

    #[test]
    fn scanner_projection_strips_system_and_distance() {
        let s = schema();
        let pks = vec!["id".to_string()];
        let user = vec![
            "_distance".to_string(),
            "vector".to_string(),
            "_rowid".to_string(),
            "_rowaddr".to_string(),
        ];
        let cols = build_scanner_projection(Some(&user), &s, &pks);
        assert_eq!(cols, vec!["vector".to_string(), "id".to_string()]);
    }

    #[test]
    fn scanner_projection_default_uses_base_schema() {
        let s = schema();
        let pks = vec!["id".to_string()];
        let cols = build_scanner_projection(None, &s, &pks);
        assert_eq!(
            cols,
            vec!["id".to_string(), "name".to_string(), "vector".to_string()]
        );
    }

    #[test]
    fn canonical_schema_honors_user_order_for_distance() {
        let s = schema();
        let pks = vec!["id".to_string()];
        let user = vec!["_distance".to_string(), "vector".to_string()];
        let out = canonical_output_schema(Some(&user), &s, &pks, true);
        let names: Vec<&str> = out.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["_distance", "vector", "id"]);
        assert_eq!(
            out.field_with_name("_distance").unwrap().data_type(),
            &DataType::Float32
        );
    }

    #[test]
    fn canonical_schema_includes_system_cols_as_nullable_uint64() {
        let s = schema();
        let pks = vec!["id".to_string()];
        let user = vec![
            "vector".to_string(),
            "_rowid".to_string(),
            "_rowaddr".to_string(),
            "_rowoffset".to_string(),
        ];
        let out = canonical_output_schema(Some(&user), &s, &pks, false);
        let names: Vec<&str> = out.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec!["vector", "_rowid", "_rowaddr", "_rowoffset", "id"]
        );
        for sys in ["_rowid", "_rowaddr", "_rowoffset"] {
            let field = out.field_with_name(sys).unwrap();
            assert_eq!(field.data_type(), &DataType::UInt64);
            assert!(field.is_nullable(), "{sys} must be nullable for NULL fill");
        }
    }

    #[test]
    fn canonical_schema_appends_distance_when_missing() {
        let s = schema();
        let pks = vec!["id".to_string()];
        let user = vec!["vector".to_string()];
        let out = canonical_output_schema(Some(&user), &s, &pks, true);
        let names: Vec<&str> = out.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["vector", "id", "_distance"]);
    }

    #[test]
    fn canonical_schema_drops_distance_when_not_requested() {
        let s = schema();
        let pks = vec!["id".to_string()];
        let user = vec!["_distance".to_string(), "vector".to_string()];
        let out = canonical_output_schema(Some(&user), &s, &pks, false);
        let names: Vec<&str> = out.fields().iter().map(|f| f.name().as_str()).collect();
        // _distance dropped because include_distance=false (e.g. point lookup / scan).
        assert_eq!(names, vec!["vector", "id"]);
    }
}
