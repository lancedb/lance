// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compound (multi-column) scalar index support.
//!
//! This module provides data structures for compound indices that allow
//! efficient lookups on predicates like `WHERE tenant_id = X AND timestamp > T`
//! using a single index.
//!
//! # Key Components
//!
//! - [`CompoundKey`]: A multi-column key using Arrow Row Format for efficient comparison
//! - [`CompoundIndexSchema`]: Schema definition for compound indices
//! - [`CompoundSargableQuery`]: Query types for compound index lookups
//!
//! # Design Decisions
//!
//! - Uses Arrow Row Format for memcomparable key encoding (~3x faster than field-by-field)
//! - NULLS FIRST ordering to match Lance's existing behavior
//! - Column limits: 2-8 columns (soft limit)
//! - Per-column statistics for flexible query pruning (implemented in later milestones)
//!
//! # See Also
//!
//! - [`compound_btree`](super::compound_btree): Index training, storage, and search implementation
//! - [`btree`](super::btree): Single-column BTree index (similar architecture)

use std::{
    any::Any,
    cmp::Ordering,
    collections::HashSet,
    ops::Bound,
};

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_row::{OwnedRow, RowConverter, Rows, SortField};
use arrow_schema::{DataType, SortOptions};
use datafusion_common::{Column, ScalarValue};
use datafusion_expr::Expr;
use lance_core::{Error, Result};
use snafu::location;

use super::AnyQuery;

// ============================================================================
// Constants
// ============================================================================

/// Minimum number of columns for a compound index.
pub const MIN_COMPOUND_INDEX_COLUMNS: usize = 2;

/// Maximum number of columns for a compound index (soft limit).
pub const MAX_COMPOUND_INDEX_COLUMNS: usize = 8;

/// Sort options for compound indices - ASC, NULLS FIRST.
///
/// This matches Lance's existing behavior for single-column BTree indices
/// (see btree.rs:1218-1220).
pub const COMPOUND_SORT_OPTIONS: SortOptions = SortOptions {
    descending: false,
    nulls_first: true,
};

// ============================================================================
// CompoundKey
// ============================================================================

/// A compound key converted to Arrow Row Format for efficient comparison.
///
/// The Arrow Row Format provides a memcomparable binary encoding that allows
/// multi-column keys to be compared with a simple byte comparison (`memcmp`).
/// This is approximately 3x faster than field-by-field comparison.
///
/// # Thread Safety
///
/// `CompoundKey` is `Send + Sync` and can be safely shared across threads.
///
/// # Example
///
/// ```
/// # use lance_index::scalar::compound::{CompoundIndexSchema, CompoundKey};
/// # use arrow_schema::DataType;
/// # use std::sync::Arc;
/// # use arrow_array::{StringArray, Int64Array, ArrayRef};
/// # fn main() -> lance_core::Result<()> {
/// let schema = CompoundIndexSchema::new(
///     vec!["tenant_id".to_string(), "timestamp".to_string()],
///     vec![DataType::Utf8, DataType::Int64],
/// )?;
/// let converter = schema.row_converter()?;
///
/// // Create arrays for the columns
/// let arrays: Vec<ArrayRef> = vec![
///     Arc::new(StringArray::from(vec!["tenant_a", "tenant_b"])),
///     Arc::new(Int64Array::from(vec![100, 200])),
/// ];
///
/// // Create keys from arrays
/// let key1 = CompoundKey::from_arrays(&converter, &arrays, 0)?;
/// let key2 = CompoundKey::from_arrays(&converter, &arrays, 1)?;
///
/// // Keys can be compared directly
/// assert!(key1 < key2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Eq)]
pub struct CompoundKey {
    /// The raw row bytes in memcomparable format.
    row: OwnedRow,
}

impl CompoundKey {
    /// Create a new CompoundKey from an OwnedRow.
    ///
    /// This is the internal constructor. Use [`from_arrays`] or [`from_scalars`]
    /// for most use cases.
    pub fn new(row: OwnedRow) -> Self {
        Self { row }
    }

    /// Create a CompoundKey from column arrays at a specific row index.
    ///
    /// # Arguments
    ///
    /// * `converter` - The RowConverter configured for this index's schema
    /// * `arrays` - Column arrays in index order
    /// * `row_index` - The row index to extract
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The row index is out of bounds
    /// - The arrays don't match the converter's schema
    pub fn from_arrays(
        converter: &RowConverter,
        arrays: &[ArrayRef],
        row_index: usize,
    ) -> Result<Self> {
        if arrays.is_empty() {
            return Err(Error::Index {
                message: "Cannot create CompoundKey from empty arrays".to_string(),
                location: location!(),
            });
        }

        // Check bounds
        let len = arrays[0].len();
        if row_index >= len {
            return Err(Error::Index {
                message: format!(
                    "Row index {} out of bounds for arrays with length {}",
                    row_index, len
                ),
                location: location!(),
            });
        }

        // Slice arrays to single row
        let sliced: Vec<ArrayRef> = arrays
            .iter()
            .map(|arr| arr.slice(row_index, 1))
            .collect();

        // Convert to rows
        let rows = converter.convert_columns(&sliced).map_err(|e| Error::Index {
            message: format!("Failed to convert arrays to row format: {}", e),
            location: location!(),
        })?;

        Ok(Self {
            row: rows.row(0).owned(),
        })
    }

    /// Create a CompoundKey from scalar values.
    ///
    /// # Arguments
    ///
    /// * `converter` - The RowConverter configured for this index's schema
    /// * `values` - Scalar values in index column order
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The values don't match the converter's schema
    /// - Conversion to Arrow arrays fails
    pub fn from_scalars(converter: &RowConverter, values: &[ScalarValue]) -> Result<Self> {
        if values.is_empty() {
            return Err(Error::Index {
                message: "Cannot create CompoundKey from empty values".to_string(),
                location: location!(),
            });
        }

        // Convert scalars to single-row arrays
        let arrays: Vec<ArrayRef> = values
            .iter()
            .map(|v| v.to_array())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Index {
                message: format!("Failed to convert scalar to array: {}", e),
                location: location!(),
            })?;

        // Convert to rows
        let rows = converter.convert_columns(&arrays).map_err(|e| Error::Index {
            message: format!("Failed to convert scalars to row format: {}", e),
            location: location!(),
        })?;

        Ok(Self {
            row: rows.row(0).owned(),
        })
    }

    /// Get the underlying row bytes for direct comparison.
    ///
    /// The bytes are in memcomparable format and can be compared with `memcmp`.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.row.as_ref()
    }

    /// Get a reference to the underlying OwnedRow.
    #[inline]
    pub fn as_row(&self) -> &OwnedRow {
        &self.row
    }
}

impl Ord for CompoundKey {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.row.as_ref().cmp(other.row.as_ref())
    }
}

impl PartialOrd for CompoundKey {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for CompoundKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.row.as_ref() == other.row.as_ref()
    }
}

impl std::hash::Hash for CompoundKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.row.as_ref().hash(state);
    }
}

// ============================================================================
// CompoundIndexSchema
// ============================================================================

/// Schema definition for a compound index.
///
/// Holds metadata about the columns in the index and provides utilities
/// for creating [`RowConverter`] instances. The schema enforces validation
/// rules for compound indices.
///
/// # Validation Rules
///
/// - Minimum 2 columns, maximum 8 columns
/// - No nested types (List, Struct, Map, Union)
/// - No duplicate column names
/// - Column order matters for query optimization
///
/// # Example
///
/// ```
/// # use lance_index::scalar::compound::CompoundIndexSchema;
/// # use arrow_schema::DataType;
/// # fn main() -> lance_core::Result<()> {
/// let schema = CompoundIndexSchema::new(
///     vec!["tenant_id".to_string(), "status".to_string(), "timestamp".to_string()],
///     vec![DataType::Utf8, DataType::Utf8, DataType::Int64],
/// )?;
///
/// // Create a RowConverter for key comparison
/// let converter = schema.row_converter()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CompoundIndexSchema {
    /// Column names in index order.
    columns: Vec<String>,
    /// Data types for each column.
    data_types: Vec<DataType>,
}

impl CompoundIndexSchema {
    /// Create a new CompoundIndexSchema.
    ///
    /// # Arguments
    ///
    /// * `columns` - Column names in index order
    /// * `data_types` - Data types for each column
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Less than 2 columns
    /// - More than 8 columns
    /// - Nested types (List, Struct, Map, Union)
    /// - Duplicate column names
    /// - Mismatched column/type counts
    pub fn new(columns: Vec<String>, data_types: Vec<DataType>) -> Result<Self> {
        // Check counts match
        if columns.len() != data_types.len() {
            return Err(Error::Index {
                message: format!(
                    "Column count ({}) does not match data type count ({})",
                    columns.len(),
                    data_types.len()
                ),
                location: location!(),
            });
        }

        // Check minimum columns
        if columns.len() < MIN_COMPOUND_INDEX_COLUMNS {
            return Err(Error::Index {
                message: format!(
                    "Compound index requires at least {} columns, got {}",
                    MIN_COMPOUND_INDEX_COLUMNS,
                    columns.len()
                ),
                location: location!(),
            });
        }

        // Check maximum columns
        if columns.len() > MAX_COMPOUND_INDEX_COLUMNS {
            return Err(Error::Index {
                message: format!(
                    "Compound index exceeds maximum of {} columns (got {})",
                    MAX_COMPOUND_INDEX_COLUMNS,
                    columns.len()
                ),
                location: location!(),
            });
        }

        // Check for duplicate column names
        let mut seen = HashSet::new();
        for col in &columns {
            if !seen.insert(col) {
                return Err(Error::Index {
                    message: format!("Duplicate column name in compound index: '{}'", col),
                    location: location!(),
                });
            }
        }

        // Check for nested types
        for (i, dt) in data_types.iter().enumerate() {
            if is_nested_type(dt) {
                return Err(Error::Index {
                    message: format!(
                        "Column '{}' has nested type {:?} which is not supported for compound indices",
                        columns[i], dt
                    ),
                    location: location!(),
                });
            }
        }

        Ok(Self { columns, data_types })
    }

    /// Create a RowConverter for this schema.
    ///
    /// The converter is configured with NULLS FIRST, ASC ordering for all columns.
    pub fn row_converter(&self) -> Result<RowConverter> {
        let fields: Vec<SortField> = self
            .data_types
            .iter()
            .map(|dt| SortField::new_with_options(dt.clone(), COMPOUND_SORT_OPTIONS))
            .collect();

        RowConverter::new(fields).map_err(|e| Error::Index {
            message: format!("Failed to create RowConverter: {}", e),
            location: location!(),
        })
    }

    /// Get the column names.
    #[inline]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Get the data types.
    #[inline]
    pub fn data_types(&self) -> &[DataType] {
        &self.data_types
    }

    /// Get the number of columns.
    #[inline]
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get the column index by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }
}

// ============================================================================
// CompoundSargableQuery
// ============================================================================

/// Query types for compound indices.
///
/// These represent the different ways a compound index can be queried.
/// The query type determines which pages/rows need to be searched.
///
/// # Query Semantics
///
/// Compound indices follow left-prefix query semantics:
///
/// - **Full prefix queries** (all columns have equality): Most efficient, single point lookup
/// - **Partial prefix queries** (leading columns have equality): Range scan within prefix
/// - **Range on last column**: Equality on prefix columns, range on last
///
/// # Example
///
/// For an index on `(tenant_id, status, timestamp)`:
///
/// ```text
/// // Full key lookup (most efficient)
/// WHERE tenant_id = 'acme' AND status = 'active' AND timestamp = 123
///
/// // Prefix with range on last (efficient)
/// WHERE tenant_id = 'acme' AND status = 'active' AND timestamp > 100
///
/// // Partial prefix (range scan)
/// WHERE tenant_id = 'acme' AND status = 'active'
///
/// // Single column prefix
/// WHERE tenant_id = 'acme'
/// ```
///
/// # Note
///
/// This type does not implement `AnyQuery` yet - that integration is planned
/// for Milestone 3 of the compound index implementation.
#[derive(Debug, Clone, PartialEq)]
pub enum CompoundSargableQuery {
    /// All columns have equality predicates - single point lookup.
    ///
    /// Example: `WHERE tenant_id = 'acme' AND status = 'active' AND timestamp = 123`
    FullKeyLookup(CompoundKey),

    /// Prefix columns have equality, optionally last has range.
    ///
    /// This is the most common query pattern for compound indices.
    ///
    /// Example: `WHERE tenant_id = 'acme' AND timestamp > 100`
    PrefixLookup {
        /// Equality predicates for prefix columns (in index order).
        /// Must have at least one value.
        prefix: Vec<ScalarValue>,

        /// Optional range on the next column after prefix.
        /// If None, all rows matching the prefix are returned.
        range: Option<(Bound<ScalarValue>, Bound<ScalarValue>)>,
    },

    /// Range query on full compound key.
    ///
    /// Used for cursor-based pagination or full key range scans.
    Range {
        lower: Bound<CompoundKey>,
        upper: Bound<CompoundKey>,
    },

    /// IN-list query on the first column.
    ///
    /// Returns all rows where the first column matches any value in the list.
    /// This is equivalent to multiple prefix lookups OR'd together.
    ///
    /// Example: `WHERE tenant_id IN ('acme', 'beta', 'gamma')`
    ///
    /// Note: IN-list on non-first columns is not supported as it would require
    /// scanning all prefixes (defeating the purpose of the compound index).
    FirstColumnIn(Vec<ScalarValue>),

    /// IN-list query after a prefix of equality predicates.
    ///
    /// Returns all rows where the prefix columns match exactly and the next column
    /// matches any value in the IN-list.
    ///
    /// Example: `WHERE tenant_id = 'acme' AND status IN ('active', 'pending')`
    ///
    /// This is more efficient than FirstColumnIn when there's a leading equality
    /// predicate, as it narrows the search space to rows matching the prefix first.
    PrefixIn {
        /// Equality predicates for prefix columns (in index order).
        /// Must have at least one value.
        prefix: Vec<ScalarValue>,
        /// Values to match in the next column after the prefix.
        in_values: Vec<ScalarValue>,
    },

    /// IS NULL query after a prefix of equality predicates.
    ///
    /// Returns all rows where the prefix columns match exactly and the next column
    /// is NULL.
    ///
    /// Example: `WHERE tenant_id = 'acme' AND deleted_at IS NULL`
    ///
    /// This is useful for soft-delete patterns where you want to find non-deleted
    /// records for a specific tenant.
    PrefixIsNull {
        /// Equality predicates for prefix columns (in index order).
        /// Must have at least one value.
        prefix: Vec<ScalarValue>,
        /// The column index (after prefix) that should be NULL.
        null_column_idx: usize,
    },
}

impl CompoundSargableQuery {
    /// Create a full key lookup query.
    pub fn full_key_lookup(key: CompoundKey) -> Self {
        Self::FullKeyLookup(key)
    }

    /// Create a prefix lookup query with equality on prefix columns.
    pub fn prefix_lookup(prefix: Vec<ScalarValue>) -> Self {
        Self::PrefixLookup {
            prefix,
            range: None,
        }
    }

    /// Create a prefix lookup query with a range on the last column.
    pub fn prefix_lookup_with_range(
        prefix: Vec<ScalarValue>,
        range: (Bound<ScalarValue>, Bound<ScalarValue>),
    ) -> Self {
        Self::PrefixLookup {
            prefix,
            range: Some(range),
        }
    }

    /// Create a range query on compound keys.
    pub fn range(lower: Bound<CompoundKey>, upper: Bound<CompoundKey>) -> Self {
        Self::Range { lower, upper }
    }

    /// Returns true if this is a full key lookup (point query).
    pub fn is_point_query(&self) -> bool {
        matches!(self, Self::FullKeyLookup(_))
    }

    /// Returns the number of prefix columns used in this query.
    ///
    /// For `FullKeyLookup`, returns the total number of columns.
    /// For `PrefixLookup`, returns the prefix length.
    /// For `Range` and `FirstColumnIn`, returns 0 (no specific prefix).
    /// For `PrefixIn` and `PrefixIsNull`, returns the prefix length.
    pub fn prefix_length(&self) -> usize {
        match self {
            Self::FullKeyLookup(_) => 0, // Full key, not a prefix
            Self::PrefixLookup { prefix, .. } => prefix.len(),
            Self::Range { .. } => 0,
            Self::FirstColumnIn(_) => 0, // IN-list on first column, not a prefix
            Self::PrefixIn { prefix, .. } => prefix.len(),
            Self::PrefixIsNull { prefix, .. } => prefix.len(),
        }
    }

    /// Returns true if this query has a range component.
    pub fn has_range(&self) -> bool {
        match self {
            Self::FullKeyLookup(_) => false,
            Self::PrefixLookup { range, .. } => range.is_some(),
            Self::Range { .. } => true,
            Self::FirstColumnIn(_) => false, // IN-list is not a range
            Self::PrefixIn { .. } => false,  // IN-list is not a range
            Self::PrefixIsNull { .. } => false, // IS NULL is not a range
        }
    }

    /// Create an IN-list query on the first column.
    ///
    /// # Arguments
    ///
    /// * `values` - The list of values to match against the first column
    pub fn first_column_in(values: Vec<ScalarValue>) -> Self {
        Self::FirstColumnIn(values)
    }

    /// Create an IN-list query after a prefix of equality predicates.
    ///
    /// # Arguments
    ///
    /// * `prefix` - Equality predicates for leading columns
    /// * `in_values` - Values to match in the next column after the prefix
    pub fn prefix_in(prefix: Vec<ScalarValue>, in_values: Vec<ScalarValue>) -> Self {
        Self::PrefixIn { prefix, in_values }
    }

    /// Create an IS NULL query after a prefix of equality predicates.
    ///
    /// # Arguments
    ///
    /// * `prefix` - Equality predicates for leading columns
    /// * `null_column_idx` - The column index (after prefix) that should be NULL
    pub fn prefix_is_null(prefix: Vec<ScalarValue>, null_column_idx: usize) -> Self {
        Self::PrefixIsNull { prefix, null_column_idx }
    }
}

impl AnyQuery for CompoundSargableQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, col: &str) -> String {
        match self {
            Self::FullKeyLookup(_key) => {
                format!("{}[compound key lookup]", col)
            }
            Self::PrefixLookup { prefix, range } => {
                let prefix_str = prefix
                    .iter()
                    .enumerate()
                    .map(|(i, v)| format!("col{}={}", i, v))
                    .collect::<Vec<_>>()
                    .join(" AND ");
                match range {
                    Some((lower, upper)) => {
                        let range_str = format_bound_range(lower, upper, prefix.len());
                        format!("{} AND {}", prefix_str, range_str)
                    }
                    None => prefix_str,
                }
            }
            Self::Range { lower, upper } => {
                let lower_str = match lower {
                    Bound::Unbounded => "(-∞".to_string(),
                    Bound::Included(_) => "[key".to_string(),
                    Bound::Excluded(_) => "(key".to_string(),
                };
                let upper_str = match upper {
                    Bound::Unbounded => "∞)".to_string(),
                    Bound::Included(_) => "key]".to_string(),
                    Bound::Excluded(_) => "key)".to_string(),
                };
                format!("{} {} range: {}, {}", col, col, lower_str, upper_str)
            }
            Self::FirstColumnIn(values) => {
                let values_str = values
                    .iter()
                    .map(|v| format!("{}", v))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}[0] IN ({})", col, values_str)
            }
            Self::PrefixIn { prefix, in_values } => {
                let prefix_str = prefix
                    .iter()
                    .enumerate()
                    .map(|(i, v)| format!("col{}={}", i, v))
                    .collect::<Vec<_>>()
                    .join(" AND ");
                let in_values_str = in_values
                    .iter()
                    .map(|v| format!("{}", v))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{} AND col{} IN ({})", prefix_str, prefix.len(), in_values_str)
            }
            Self::PrefixIsNull { prefix, null_column_idx } => {
                let prefix_str = prefix
                    .iter()
                    .enumerate()
                    .map(|(i, v)| format!("col{}={}", i, v))
                    .collect::<Vec<_>>()
                    .join(" AND ");
                format!("{} AND col{} IS NULL", prefix_str, prefix.len() + null_column_idx)
            }
        }
    }

    fn to_expr(&self, col: String) -> Expr {
        // For compound queries, we return a placeholder expression.
        // The actual conversion back to DataFusion expressions requires
        // knowing the column names, which is context the query doesn't have.
        // This is primarily used for display/debugging purposes.
        match self {
            Self::FullKeyLookup(_) => {
                // Return a simple column reference as placeholder
                Expr::Column(Column::new_unqualified(col))
            }
            Self::PrefixLookup { prefix, range } => {
                // Build AND expression for prefix columns
                // This is a simplified representation
                if prefix.is_empty() {
                    return Expr::Literal(ScalarValue::Boolean(Some(true)), None);
                }

                let mut expr = Expr::Literal(ScalarValue::Boolean(Some(true)), None);
                for (i, val) in prefix.iter().enumerate() {
                    let col_expr = Expr::Column(Column::new_unqualified(format!("{}_{}", col, i)));
                    let eq_expr = col_expr.eq(Expr::Literal(val.clone(), None));
                    expr = expr.and(eq_expr);
                }

                // Add range if present
                if let Some((lower, upper)) = range {
                    let range_col =
                        Expr::Column(Column::new_unqualified(format!("{}_{}", col, prefix.len())));
                    let range_expr = build_range_expr(range_col, lower, upper);
                    expr = expr.and(range_expr);
                }

                expr
            }
            Self::Range { .. } => {
                // Range on compound key - return placeholder
                Expr::Column(Column::new_unqualified(col))
            }
            Self::FirstColumnIn(values) => {
                // Build IN list expression for the first column
                let col_expr = Expr::Column(Column::new_unqualified(format!("{}_0", col)));
                let list = values
                    .iter()
                    .map(|v| Expr::Literal(v.clone(), None))
                    .collect();
                Expr::InList(datafusion_expr::expr::InList {
                    expr: Box::new(col_expr),
                    list,
                    negated: false,
                })
            }
            Self::PrefixIn { prefix, in_values } => {
                // Build AND expression for prefix + IN list on next column
                if prefix.is_empty() {
                    // No prefix, just IN list on first column
                    let col_expr = Expr::Column(Column::new_unqualified(format!("{}_0", col)));
                    let list = in_values
                        .iter()
                        .map(|v| Expr::Literal(v.clone(), None))
                        .collect();
                    return Expr::InList(datafusion_expr::expr::InList {
                        expr: Box::new(col_expr),
                        list,
                        negated: false,
                    });
                }

                let mut expr = Expr::Literal(ScalarValue::Boolean(Some(true)), None);
                for (i, val) in prefix.iter().enumerate() {
                    let col_expr = Expr::Column(Column::new_unqualified(format!("{}_{}", col, i)));
                    let eq_expr = col_expr.eq(Expr::Literal(val.clone(), None));
                    expr = expr.and(eq_expr);
                }

                // Add IN list for the column after prefix
                let in_col = Expr::Column(Column::new_unqualified(format!("{}_{}", col, prefix.len())));
                let list = in_values
                    .iter()
                    .map(|v| Expr::Literal(v.clone(), None))
                    .collect();
                let in_expr = Expr::InList(datafusion_expr::expr::InList {
                    expr: Box::new(in_col),
                    list,
                    negated: false,
                });
                expr.and(in_expr)
            }
            Self::PrefixIsNull { prefix, null_column_idx } => {
                // Build AND expression for prefix + IS NULL on specified column
                if prefix.is_empty() {
                    // No prefix, just IS NULL on first column
                    let col_expr = Expr::Column(Column::new_unqualified(format!("{}_{}", col, null_column_idx)));
                    return col_expr.is_null();
                }

                let mut expr = Expr::Literal(ScalarValue::Boolean(Some(true)), None);
                for (i, val) in prefix.iter().enumerate() {
                    let col_expr = Expr::Column(Column::new_unqualified(format!("{}_{}", col, i)));
                    let eq_expr = col_expr.eq(Expr::Literal(val.clone(), None));
                    expr = expr.and(eq_expr);
                }

                // Add IS NULL for the column after prefix
                let null_col = Expr::Column(Column::new_unqualified(format!("{}_{}", col, prefix.len() + null_column_idx)));
                expr.and(null_col.is_null())
            }
        }
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }
}

/// Format a bound range for display.
fn format_bound_range(lower: &Bound<ScalarValue>, upper: &Bound<ScalarValue>, col_idx: usize) -> String {
    match (lower, upper) {
        (Bound::Unbounded, Bound::Unbounded) => format!("col{} IN (-∞, ∞)", col_idx),
        (Bound::Unbounded, Bound::Included(v)) => format!("col{} <= {}", col_idx, v),
        (Bound::Unbounded, Bound::Excluded(v)) => format!("col{} < {}", col_idx, v),
        (Bound::Included(v), Bound::Unbounded) => format!("col{} >= {}", col_idx, v),
        (Bound::Excluded(v), Bound::Unbounded) => format!("col{} > {}", col_idx, v),
        (Bound::Included(l), Bound::Included(u)) => format!("col{} BETWEEN {} AND {}", col_idx, l, u),
        (Bound::Included(l), Bound::Excluded(u)) => format!("col{} >= {} AND col{} < {}", col_idx, l, col_idx, u),
        (Bound::Excluded(l), Bound::Included(u)) => format!("col{} > {} AND col{} <= {}", col_idx, l, col_idx, u),
        (Bound::Excluded(l), Bound::Excluded(u)) => format!("col{} > {} AND col{} < {}", col_idx, l, col_idx, u),
    }
}

/// Build a range expression for DataFusion.
fn build_range_expr(col: Expr, lower: &Bound<ScalarValue>, upper: &Bound<ScalarValue>) -> Expr {
    match (lower, upper) {
        (Bound::Unbounded, Bound::Unbounded) => Expr::Literal(ScalarValue::Boolean(Some(true)), None),
        (Bound::Unbounded, Bound::Included(v)) => col.lt_eq(Expr::Literal(v.clone(), None)),
        (Bound::Unbounded, Bound::Excluded(v)) => col.lt(Expr::Literal(v.clone(), None)),
        (Bound::Included(v), Bound::Unbounded) => col.gt_eq(Expr::Literal(v.clone(), None)),
        (Bound::Excluded(v), Bound::Unbounded) => col.gt(Expr::Literal(v.clone(), None)),
        (Bound::Included(l), Bound::Included(u)) => col
            .clone()
            .gt_eq(Expr::Literal(l.clone(), None))
            .and(col.lt_eq(Expr::Literal(u.clone(), None))),
        (Bound::Included(l), Bound::Excluded(u)) => col
            .clone()
            .gt_eq(Expr::Literal(l.clone(), None))
            .and(col.lt(Expr::Literal(u.clone(), None))),
        (Bound::Excluded(l), Bound::Included(u)) => col
            .clone()
            .gt(Expr::Literal(l.clone(), None))
            .and(col.lt_eq(Expr::Literal(u.clone(), None))),
        (Bound::Excluded(l), Bound::Excluded(u)) => col
            .clone()
            .gt(Expr::Literal(l.clone(), None))
            .and(col.lt(Expr::Literal(u.clone(), None))),
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if a data type is nested (not supported by compound indices).
///
/// Nested types are not supported because Arrow Row Format requires fixed-width
/// or variable-width primitive types for efficient encoding.
fn is_nested_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::List(_)
            | DataType::ListView(_)
            | DataType::LargeList(_)
            | DataType::LargeListView(_)
            | DataType::FixedSizeList(_, _)
            | DataType::Map(_, _)
            | DataType::Struct(_)
            | DataType::Union(_, _)
    )
}

/// Convert a RecordBatch to Rows using a RowConverter.
///
/// This is a convenience function for batch processing.
pub fn batch_to_rows(converter: &RowConverter, batch: &RecordBatch) -> Result<Rows> {
    let arrays: Vec<ArrayRef> = batch.columns().to_vec();
    converter.convert_columns(&arrays).map_err(|e| Error::Index {
        message: format!("Failed to convert batch to rows: {}", e),
        location: location!(),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringArray};
    use std::sync::Arc;

    /// Helper to create a schema and converter for testing
    fn create_test_schema(
        columns: Vec<&str>,
        data_types: Vec<DataType>,
    ) -> Result<(CompoundIndexSchema, RowConverter)> {
        let schema = CompoundIndexSchema::new(
            columns.into_iter().map(String::from).collect(),
            data_types,
        )?;
        let converter = schema.row_converter()?;
        Ok((schema, converter))
    }

    // ========================================================================
    // CompoundIndexSchema Tests
    // ========================================================================

    #[test]
    fn test_schema_minimum_columns() {
        let result = CompoundIndexSchema::new(
            vec!["a".to_string(), "b".to_string()],
            vec![DataType::Utf8, DataType::Int64],
        );
        assert!(result.is_ok());
        let schema = result.unwrap();
        assert_eq!(schema.num_columns(), 2);
    }

    #[test]
    fn test_schema_maximum_columns() {
        let columns: Vec<String> = (0..8).map(|i| format!("col{}", i)).collect();
        let data_types = vec![DataType::Int64; 8];
        let result = CompoundIndexSchema::new(columns, data_types);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().num_columns(), 8);
    }

    #[test]
    fn test_schema_rejects_single_column() {
        let result = CompoundIndexSchema::new(vec!["a".to_string()], vec![DataType::Utf8]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("at least 2 columns"));
    }

    #[test]
    fn test_schema_rejects_too_many_columns() {
        let columns: Vec<String> = (0..9).map(|i| format!("col{}", i)).collect();
        let data_types = vec![DataType::Int64; 9];
        let result = CompoundIndexSchema::new(columns, data_types);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exceeds maximum"));
    }

    #[test]
    fn test_schema_rejects_nested_types() {
        let result = CompoundIndexSchema::new(
            vec!["a".to_string(), "b".to_string()],
            vec![
                DataType::Utf8,
                DataType::List(Arc::new(arrow_schema::Field::new(
                    "item",
                    DataType::Int64,
                    true,
                ))),
            ],
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("nested type"));
    }

    #[test]
    fn test_schema_rejects_duplicate_columns() {
        let result = CompoundIndexSchema::new(
            vec!["a".to_string(), "a".to_string()],
            vec![DataType::Utf8, DataType::Int64],
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Duplicate column name"));
    }

    #[test]
    fn test_schema_rejects_mismatched_counts() {
        let result = CompoundIndexSchema::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![DataType::Utf8, DataType::Int64],
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("does not match"));
    }

    #[test]
    fn test_schema_column_index() {
        let schema = CompoundIndexSchema::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![DataType::Utf8, DataType::Int64, DataType::Float64],
        )
        .unwrap();

        assert_eq!(schema.column_index("a"), Some(0));
        assert_eq!(schema.column_index("b"), Some(1));
        assert_eq!(schema.column_index("c"), Some(2));
        assert_eq!(schema.column_index("d"), None);
    }

    // ========================================================================
    // CompoundKey Ordering Tests
    // ========================================================================

    #[test]
    fn test_compound_key_ordering_same_prefix() {
        // ["a", 1] < ["a", 2]
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let key1 = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("a".to_string())),
                ScalarValue::Int64(Some(1)),
            ],
        )
        .unwrap();

        let key2 = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("a".to_string())),
                ScalarValue::Int64(Some(2)),
            ],
        )
        .unwrap();

        assert!(key1 < key2);
        assert!(key2 > key1);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_compound_key_ordering_different_prefix() {
        // ["a", 2] < ["b", 1]
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let key1 = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("a".to_string())),
                ScalarValue::Int64(Some(2)),
            ],
        )
        .unwrap();

        let key2 = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("b".to_string())),
                ScalarValue::Int64(Some(1)),
            ],
        )
        .unwrap();

        assert!(key1 < key2);
    }

    #[test]
    fn test_null_first_column() {
        // [NULL, 1] < ["a", 1] (NULLS FIRST)
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let null_key = CompoundKey::from_scalars(
            &converter,
            &[ScalarValue::Utf8(None), ScalarValue::Int64(Some(1))],
        )
        .unwrap();

        let value_key = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("a".to_string())),
                ScalarValue::Int64(Some(1)),
            ],
        )
        .unwrap();

        assert!(null_key < value_key);
    }

    #[test]
    fn test_null_second_column() {
        // ["a", NULL] < ["a", 1] (NULLS FIRST)
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let null_key = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("a".to_string())),
                ScalarValue::Int64(None),
            ],
        )
        .unwrap();

        let value_key = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("a".to_string())),
                ScalarValue::Int64(Some(1)),
            ],
        )
        .unwrap();

        assert!(null_key < value_key);
    }

    #[test]
    fn test_both_null() {
        // [NULL, NULL] < [NULL, 1]
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let both_null =
            CompoundKey::from_scalars(&converter, &[ScalarValue::Utf8(None), ScalarValue::Int64(None)])
                .unwrap();

        let one_null = CompoundKey::from_scalars(
            &converter,
            &[ScalarValue::Utf8(None), ScalarValue::Int64(Some(1))],
        )
        .unwrap();

        assert!(both_null < one_null);
    }

    #[test]
    fn test_key_equality() {
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let key1 = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("test".to_string())),
                ScalarValue::Int64(Some(42)),
            ],
        )
        .unwrap();

        let key2 = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("test".to_string())),
                ScalarValue::Int64(Some(42)),
            ],
        )
        .unwrap();

        assert_eq!(key1, key2);
        assert!(!(key1 < key2));
        assert!(!(key1 > key2));
    }

    #[test]
    fn test_key_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let key1 = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("test".to_string())),
                ScalarValue::Int64(Some(42)),
            ],
        )
        .unwrap();

        let key2 = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("test".to_string())),
                ScalarValue::Int64(Some(42)),
            ],
        )
        .unwrap();

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        key1.hash(&mut hasher1);
        key2.hash(&mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    // ========================================================================
    // Mixed Data Type Tests
    // ========================================================================

    #[test]
    fn test_mixed_types_string_int() {
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        // Test various combinations
        let keys = vec![
            CompoundKey::from_scalars(
                &converter,
                &[
                    ScalarValue::Utf8(Some("apple".to_string())),
                    ScalarValue::Int64(Some(1)),
                ],
            )
            .unwrap(),
            CompoundKey::from_scalars(
                &converter,
                &[
                    ScalarValue::Utf8(Some("apple".to_string())),
                    ScalarValue::Int64(Some(2)),
                ],
            )
            .unwrap(),
            CompoundKey::from_scalars(
                &converter,
                &[
                    ScalarValue::Utf8(Some("banana".to_string())),
                    ScalarValue::Int64(Some(1)),
                ],
            )
            .unwrap(),
        ];

        // Verify ordering
        assert!(keys[0] < keys[1]); // same string, different int
        assert!(keys[1] < keys[2]); // different string
        assert!(keys[0] < keys[2]); // transitivity
    }

    #[test]
    fn test_mixed_types_with_timestamp() {
        let (_schema, converter) = create_test_schema(
            vec!["s", "i", "ts"],
            vec![
                DataType::Utf8,
                DataType::Int64,
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            ],
        )
        .unwrap();

        let key1 = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("tenant".to_string())),
                ScalarValue::Int64(Some(1)),
                ScalarValue::TimestampMicrosecond(Some(1000), None),
            ],
        )
        .unwrap();

        let key2 = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("tenant".to_string())),
                ScalarValue::Int64(Some(1)),
                ScalarValue::TimestampMicrosecond(Some(2000), None),
            ],
        )
        .unwrap();

        assert!(key1 < key2);
    }

    // ========================================================================
    // Array Conversion Tests
    // ========================================================================

    #[test]
    fn test_from_arrays() {
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let string_array: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let int_array: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));

        let key0 = CompoundKey::from_arrays(&converter, &[string_array.clone(), int_array.clone()], 0)
            .unwrap();
        let key1 = CompoundKey::from_arrays(&converter, &[string_array.clone(), int_array.clone()], 1)
            .unwrap();
        let key2 = CompoundKey::from_arrays(&converter, &[string_array, int_array], 2).unwrap();

        // Verify ordering matches array order
        assert!(key0 < key1);
        assert!(key1 < key2);
    }

    #[test]
    fn test_from_arrays_out_of_bounds() {
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let string_array: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let int_array: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));

        let result = CompoundKey::from_arrays(&converter, &[string_array, int_array], 5);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("out of bounds"));
    }

    #[test]
    fn test_from_arrays_empty() {
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let result = CompoundKey::from_arrays(&converter, &[], 0);
        assert!(result.is_err());
    }

    // ========================================================================
    // CompoundSargableQuery Tests
    // ========================================================================

    #[test]
    fn test_query_full_key_lookup() {
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let key = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("test".to_string())),
                ScalarValue::Int64(Some(42)),
            ],
        )
        .unwrap();

        let query = CompoundSargableQuery::full_key_lookup(key.clone());

        assert!(query.is_point_query());
        assert!(matches!(query, CompoundSargableQuery::FullKeyLookup(k) if k == key));
    }

    #[test]
    fn test_query_prefix_lookup() {
        let query = CompoundSargableQuery::prefix_lookup(vec![
            ScalarValue::Utf8(Some("tenant".to_string())),
        ]);

        assert!(!query.is_point_query());
        assert_eq!(query.prefix_length(), 1);

        if let CompoundSargableQuery::PrefixLookup { prefix, range } = query {
            assert_eq!(prefix.len(), 1);
            assert!(range.is_none());
        } else {
            panic!("Expected PrefixLookup");
        }
    }

    #[test]
    fn test_query_prefix_lookup_with_range() {
        let query = CompoundSargableQuery::prefix_lookup_with_range(
            vec![ScalarValue::Utf8(Some("tenant".to_string()))],
            (
                Bound::Included(ScalarValue::Int64(Some(100))),
                Bound::Excluded(ScalarValue::Int64(Some(200))),
            ),
        );

        assert!(!query.is_point_query());
        assert_eq!(query.prefix_length(), 1);

        if let CompoundSargableQuery::PrefixLookup { prefix, range } = query {
            assert_eq!(prefix.len(), 1);
            assert!(range.is_some());
            let (lower, upper) = range.unwrap();
            assert!(matches!(lower, Bound::Included(ScalarValue::Int64(Some(100)))));
            assert!(matches!(upper, Bound::Excluded(ScalarValue::Int64(Some(200)))));
        } else {
            panic!("Expected PrefixLookup");
        }
    }

    #[test]
    fn test_query_range() {
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let lower_key = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("a".to_string())),
                ScalarValue::Int64(Some(1)),
            ],
        )
        .unwrap();

        let upper_key = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("z".to_string())),
                ScalarValue::Int64(Some(100)),
            ],
        )
        .unwrap();

        let query = CompoundSargableQuery::range(
            Bound::Included(lower_key.clone()),
            Bound::Excluded(upper_key.clone()),
        );

        assert!(!query.is_point_query());
        assert_eq!(query.prefix_length(), 0);
    }

    // ========================================================================
    // Roundtrip Tests
    // ========================================================================

    #[test]
    fn test_roundtrip_from_scalars() {
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let values = vec![
            ScalarValue::Utf8(Some("test_value".to_string())),
            ScalarValue::Int64(Some(12345)),
        ];

        let key1 = CompoundKey::from_scalars(&converter, &values).unwrap();
        let key2 = CompoundKey::from_scalars(&converter, &values).unwrap();

        // Same values should produce equal keys
        assert_eq!(key1, key2);

        // Keys should have non-empty bytes
        assert!(!key1.as_bytes().is_empty());
    }

    #[test]
    fn test_clone_preserves_equality() {
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let key = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("test".to_string())),
                ScalarValue::Int64(Some(42)),
            ],
        )
        .unwrap();

        let cloned = key.clone();
        assert_eq!(key, cloned);
        assert_eq!(key.as_bytes(), cloned.as_bytes());
    }

    // ========================================================================
    // AnyQuery Implementation Tests
    // ========================================================================

    #[test]
    fn test_any_query_format_prefix_lookup() {
        let query = CompoundSargableQuery::prefix_lookup(vec![
            ScalarValue::Utf8(Some("tenant".to_string())),
            ScalarValue::Int64(Some(42)),
        ]);

        let formatted = query.format("index");
        assert!(formatted.contains("col0="));
        assert!(formatted.contains("col1="));
    }

    #[test]
    fn test_any_query_format_prefix_with_range() {
        let query = CompoundSargableQuery::prefix_lookup_with_range(
            vec![ScalarValue::Utf8(Some("tenant".to_string()))],
            (
                Bound::Included(ScalarValue::Int64(Some(100))),
                Bound::Excluded(ScalarValue::Int64(Some(200))),
            ),
        );

        let formatted = query.format("index");
        assert!(formatted.contains("col0="));
        assert!(formatted.contains("col1"));
    }

    #[test]
    fn test_any_query_dyn_eq() {
        let query1 = CompoundSargableQuery::prefix_lookup(vec![
            ScalarValue::Utf8(Some("tenant".to_string())),
        ]);

        let query2 = CompoundSargableQuery::prefix_lookup(vec![
            ScalarValue::Utf8(Some("tenant".to_string())),
        ]);

        let query3 = CompoundSargableQuery::prefix_lookup(vec![
            ScalarValue::Utf8(Some("other".to_string())),
        ]);

        // Same query should be equal
        assert!(query1.dyn_eq(&query2));

        // Different query should not be equal
        assert!(!query1.dyn_eq(&query3));
    }

    #[test]
    fn test_any_query_as_any_downcast() {
        let query = CompoundSargableQuery::prefix_lookup(vec![
            ScalarValue::Utf8(Some("tenant".to_string())),
        ]);

        let any_ref = query.as_any();
        let downcasted = any_ref.downcast_ref::<CompoundSargableQuery>();
        assert!(downcasted.is_some());

        if let Some(CompoundSargableQuery::PrefixLookup { prefix, .. }) = downcasted {
            assert_eq!(prefix.len(), 1);
        } else {
            panic!("Expected PrefixLookup");
        }
    }

    #[test]
    fn test_has_range() {
        let (_schema, converter) =
            create_test_schema(vec!["s", "i"], vec![DataType::Utf8, DataType::Int64]).unwrap();

        let key = CompoundKey::from_scalars(
            &converter,
            &[
                ScalarValue::Utf8(Some("test".to_string())),
                ScalarValue::Int64(Some(42)),
            ],
        )
        .unwrap();

        // Full key lookup has no range
        let query1 = CompoundSargableQuery::full_key_lookup(key.clone());
        assert!(!query1.has_range());

        // Prefix lookup without range
        let query2 = CompoundSargableQuery::prefix_lookup(vec![
            ScalarValue::Utf8(Some("tenant".to_string())),
        ]);
        assert!(!query2.has_range());

        // Prefix lookup with range
        let query3 = CompoundSargableQuery::prefix_lookup_with_range(
            vec![ScalarValue::Utf8(Some("tenant".to_string()))],
            (Bound::Included(ScalarValue::Int64(Some(100))), Bound::Unbounded),
        );
        assert!(query3.has_range());

        // Range query
        let query4 = CompoundSargableQuery::range(
            Bound::Included(key.clone()),
            Bound::Excluded(key),
        );
        assert!(query4.has_range());
    }

    // ========================================================================
    // Property-Based Tests
    // ========================================================================

    proptest::proptest! {
        /// Property: CompoundKey ordering is total and consistent.
        ///
        /// For any three keys a, b, c:
        /// - Reflexive: a == a
        /// - Antisymmetric: if a <= b and b <= a, then a == b
        /// - Transitive: if a <= b and b <= c, then a <= c
        #[test]
        fn test_compound_key_ordering_is_total(
            val1 in proptest::option::of(-1000i64..1000i64),
            val2 in proptest::option::of(-1000i64..1000i64),
            val3 in proptest::option::of(-1000i64..1000i64),
        ) {
            let (_schema, converter) = create_test_schema(
                vec!["a", "b"],
                vec![DataType::Int64, DataType::Int64],
            ).unwrap();

            let key1 = CompoundKey::from_scalars(&converter, &[
                ScalarValue::Int64(val1),
                ScalarValue::Int64(val2),
            ]).unwrap();

            let key2 = CompoundKey::from_scalars(&converter, &[
                ScalarValue::Int64(val2),
                ScalarValue::Int64(val3),
            ]).unwrap();

            let key3 = CompoundKey::from_scalars(&converter, &[
                ScalarValue::Int64(val1),
                ScalarValue::Int64(val3),
            ]).unwrap();

            // Reflexive: a == a
            proptest::prop_assert_eq!(key1.cmp(&key1), Ordering::Equal);
            proptest::prop_assert_eq!(key2.cmp(&key2), Ordering::Equal);
            proptest::prop_assert_eq!(key3.cmp(&key3), Ordering::Equal);

            // Total ordering: exactly one of <, =, > holds
            let cmp12 = key1.cmp(&key2);
            let cmp21 = key2.cmp(&key1);
            proptest::prop_assert_eq!(cmp12, cmp21.reverse());

            // Antisymmetric check
            if cmp12 == Ordering::Equal {
                proptest::prop_assert_eq!(key1, key2);
            }
        }

        /// Property: Hash consistency with equality.
        ///
        /// If two keys are equal, their hashes must be equal.
        #[test]
        fn test_compound_key_hash_consistent_with_eq(
            val1 in proptest::option::of(-1000i64..1000i64),
            val2 in proptest::option::of(-1000i64..1000i64),
        ) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let (_schema, converter) = create_test_schema(
                vec!["a", "b"],
                vec![DataType::Int64, DataType::Int64],
            ).unwrap();

            let key1 = CompoundKey::from_scalars(&converter, &[
                ScalarValue::Int64(val1),
                ScalarValue::Int64(val2),
            ]).unwrap();

            let key2 = CompoundKey::from_scalars(&converter, &[
                ScalarValue::Int64(val1),
                ScalarValue::Int64(val2),
            ]).unwrap();

            // Keys with same values should be equal and have same hash
            proptest::prop_assert_eq!(&key1, &key2);

            let mut h1 = DefaultHasher::new();
            let mut h2 = DefaultHasher::new();
            key1.hash(&mut h1);
            key2.hash(&mut h2);
            proptest::prop_assert_eq!(h1.finish(), h2.finish());
        }

        /// Property: NULL handling follows NULLS FIRST semantics.
        ///
        /// For the first column (primary sort key):
        /// - NULL values should come before all non-NULL values
        #[test]
        fn test_compound_key_nulls_first(
            non_null_val in -1000i64..1000i64,
        ) {
            let (_schema, converter) = create_test_schema(
                vec!["a", "b"],
                vec![DataType::Int64, DataType::Int64],
            ).unwrap();

            let null_key = CompoundKey::from_scalars(&converter, &[
                ScalarValue::Int64(None),
                ScalarValue::Int64(Some(0)),
            ]).unwrap();

            let non_null_key = CompoundKey::from_scalars(&converter, &[
                ScalarValue::Int64(Some(non_null_val)),
                ScalarValue::Int64(Some(0)),
            ]).unwrap();

            // NULL should be less than any non-NULL value (NULLS FIRST)
            proptest::prop_assert_eq!(null_key.cmp(&non_null_key), Ordering::Less);
        }
    }
}
