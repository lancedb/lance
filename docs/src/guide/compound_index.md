# Compound (Multi-Column) Scalar Index

Compound indices enable efficient lookups on multi-column predicates using a single index structure. They are particularly useful for queries that filter on multiple columns simultaneously.

## Overview

A compound index is a B-tree scalar index that spans multiple columns. Instead of creating separate indices on each column and performing intersection at query time, a compound index stores rows sorted by the combined key of all indexed columns.

### When to Use Compound Indices

**Good use cases:**

- Multi-tenant datasets with time-series data: `WHERE tenant_id = 'acme' AND timestamp > T`
- Status + time filtering: `WHERE status = 'active' AND created_at BETWEEN X AND Y`
- Hierarchical data: `WHERE region = 'us-west' AND department = 'engineering'`

**Less ideal use cases:**

- Single-column equality queries (use a regular scalar index)
- Queries that don't follow the leftmost prefix rule (see below)
- Highly selective queries on non-first columns only

### Leftmost Prefix Rule

Compound indices follow the **leftmost prefix rule**: queries can efficiently use the index only if they include predicates on a contiguous prefix of the indexed columns, starting from the first column.

**Example:** For a compound index on `(tenant_id, status, timestamp)`:

| Query | Uses Index? | Notes |
|-------|-------------|-------|
| `tenant_id = X` | ✅ Yes | First column only (prefix lookup) |
| `tenant_id = X AND status = Y` | ✅ Yes | First two columns |
| `tenant_id = X AND status = Y AND timestamp > T` | ✅ Yes | All columns with range on last |
| `status = Y` | ❌ No | Skips first column |
| `tenant_id = X AND timestamp > T` | ⚠️ Partial | Only uses `tenant_id`, ignores `timestamp` |
| `tenant_id IN ('a', 'b', 'c')` | ✅ Yes | IN-list on first column supported |
| `tenant_id = X AND status IN ('a', 'b')` | ✅ Yes | Prefix + IN-list on next column |
| `tenant_id = X AND deleted_at IS NULL` | ✅ Yes | Prefix + IS NULL on next column |

## Column Ordering

The order of columns in a compound index significantly impacts query performance. Follow these guidelines:

1. **Equality columns first**: Columns used with `=` should come before columns used with ranges (`>`, `<`, `BETWEEN`)
2. **High selectivity first**: Columns that narrow down results more should come earlier
3. **Frequently filtered columns first**: Columns used in most queries should be at the beginning

### Example

For a multi-tenant SaaS application with queries like:

```sql
WHERE tenant_id = ? AND status = ? AND created_at > ?
```

The optimal column order is: `(tenant_id, status, created_at)`

- `tenant_id`: Always filtered by equality, high selectivity
- `status`: Usually filtered by equality, lower selectivity
- `created_at`: Filtered by range, comes last

## Query Patterns

### Full Key Lookup

All columns have equality predicates:

```sql
WHERE tenant_id = 'acme' AND status = 'active' AND timestamp = 123
```

This is the most efficient query pattern - it locates a single point in the index.

### Prefix Lookup

Equality on a prefix of columns:

```sql
WHERE tenant_id = 'acme'
WHERE tenant_id = 'acme' AND status = 'active'
```

Returns all rows matching the prefix.

### Prefix + Range

Equality on prefix, range on the next column:

```sql
WHERE tenant_id = 'acme' AND timestamp > 1000
WHERE tenant_id = 'acme' AND status = 'active' AND timestamp BETWEEN 1000 AND 2000
```

This is the most common real-world pattern for time-series queries.

### IN-list on First Column

```sql
WHERE tenant_id IN ('acme', 'beta', 'gamma')
```

Efficiently retrieves rows for multiple values of the first column.

### Prefix + IN-list

Equality on a prefix, then IN-list on the next column:

```sql
WHERE tenant_id = 'acme' AND status IN ('active', 'pending')
```

This pattern is useful when you have a fixed tenant but want to filter by multiple statuses. It's more efficient than multiple separate queries because:
- The prefix narrows the search space first
- IN-list values are checked within that narrowed space

### Prefix + IS NULL

Equality on a prefix, then NULL check on the next column:

```sql
WHERE tenant_id = 'acme' AND deleted_at IS NULL
```

This is a common pattern for soft-delete implementations where you want to find non-deleted records for a specific tenant. The index can efficiently:
- Seek to the tenant prefix
- Use page-level null count statistics for pruning

## Limitations

- **Maximum columns**: 8 (soft limit)
- **Minimum columns**: 2 (use regular scalar index for single column)
- **IN-list positioning**: Supported on first column, or after a prefix of equality predicates
- **IS NULL positioning**: Supported on first column, or after a prefix of equality predicates
- **OR conditions**: Not directly supported - may require multiple queries
- **LIKE/pattern matching**: Not supported

## Performance Characteristics

### Space Overhead

Compound indices store the full value of each indexed column plus row IDs. For a 3-column index on 1M rows:

- Storage ≈ (size of col1 + size of col2 + size of col3 + 8 bytes) × num_rows
- Additional per-page statistics for query pruning

### Query Performance

- **Point lookups**: O(log N) - similar to single-column B-tree
- **Prefix lookups**: O(log N + K) where K is the number of matching rows
- **Range queries**: Efficient page pruning reduces I/O

### Comparison: Compound vs Multiple Single-Column Indices

| Scenario | Compound Index | Multiple Indices |
|----------|----------------|------------------|
| Multi-column equality | Single seek | Multiple seeks + intersection |
| Storage | Single index | N separate indices |
| Update cost | One index to update | N indices to update |
| Single-column queries | Only efficient for first column | Each column optimized |

## Best Practices

1. **Analyze your query patterns** before creating compound indices
2. **Start with the most common query** and order columns accordingly
3. **Monitor page pruning effectiveness** - if many pages are scanned, column order may be suboptimal
4. **Consider query-specific indices** for very different query patterns
5. **Limit the number of columns** - more columns means more storage and update overhead
