# BTree Index

The BTree index is a two-level structure that provides efficient range queries and sorted access. 
It strikes a balance between an expensive memory structure containing all values 
and an expensive disk structure that can't be efficiently searched.

The upper layers of the BTree are designed to be cached in memory and stored in a 
BTree structure (`page_lookup.lance`), while the leaves are searched using sub-indices 
(`page_data.lance`, currently just a flat file). 
This design enables efficient memory usage - for example, with 1 billion values, 
the index can store 256K leaves of size 4K each, requiring only a few MiB of memory 
(depending on data type) for the BTree metadata while narrowing any search to just 4K values.

## Index Details

```protobuf
%%% proto.message.BTreeIndexDetails %%%
```

## Storage Layout

The BTree index consists of two files:

1. `page_lookup.lance` - The BTree structure mapping value ranges to page numbers
2. `page_data.lance` - The actual sub-indices (flat file) containing sorted values and row IDs

### Page Lookup File Schema (BTree Structure)

| Column       | Type       | Nullable | Description                                              |
|--------------|------------|----------|----------------------------------------------------------|
| `min`        | {DataType} | true     | Minimum value in the page (forms BTree keys)             |
| `max`        | {DataType} | true     | Maximum value in the page (for range pruning)            |
| `null_count` | UInt32     | false    | Number of null values in the page                        |
| `page_idx`   | UInt32     | false    | Page number pointing to the sub-index in page_data.lance |

### Schema Metadata

| Key | Type | Description |
|-----|------|-------------|
| `batch_size` | String | Number of rows per page (default: "4096") |

### Page Data File Schema (Sub-indices)

| Column   | Type       | Nullable | Description                                       |
|----------|------------|----------|---------------------------------------------------|
| `values` | {DataType} | true     | Sorted values from the indexed column (flat file) |
| `ids`    | UInt64     | false    | Row IDs corresponding to each value               |

## Accelerated Queries

The BTree index provides exact results for the following query types:

| Query Type | Description               | Operation                                                                   |
|------------|---------------------------|-----------------------------------------------------------------------------|
| **Equals** | `column = value`          | BTree lookup to find relevant pages, then search within sub-indices         |
| **Range**  | `column BETWEEN a AND b`  | BTree traversal for pages overlapping the range, then search each sub-index |
| **IsIn**   | `column IN (v1, v2, ...)` | Multiple BTree lookups, union results from all matching sub-indices         |
| **IsNull** | `column IS NULL`          | Returns rows from all pages where null_count > 0                            |

## Skipping the Sub-Index Search for Unselective Queries

A BTree search has two stages: an in-memory page lookup that selects which pages
might contain matches, followed by a per-page sub-index search that reads each
matching page from `page_data.lance` and returns row IDs. When the page lookup
already matches a large fraction of the index, the second stage rarely refines
the result enough to repay its I/O — the caller is usually better off doing a
normal scan with a recheck on the predicate.

When this triggers, `BTreeIndex::search` returns the `Indeterminate` search
result (semantically: "every row could match, recheck downstream") without
loading any pages. The query planner then falls back to a filtered scan.

The decision is gated by **two thresholds**, both of which must be exceeded
for the skip to fire:

| Environment variable                    | Default     | Effect                                                                                                                         |
|-----------------------------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------|
| `LANCE_BTREE_SKIP_ROW_THRESHOLD`        | `1000000`   | Absolute candidate-row threshold. Estimated as `matched_pages * batch_size`. Set to `0` to disable the skip entirely.          |
| `LANCE_BTREE_SKIP_FRACTION_THRESHOLD`   | `0.15`      | Fractional threshold: `matched_pages / total_pages`. Set to `1.0` (or higher) to disable the skip entirely.                     |

Both checks have to pass before the skip is taken — the matched portion has to
be both absolutely large *and* a meaningful slice of the dataset. In particular,
small indices never trip the skip (with a 4096-row batch size the row threshold
alone requires the index to span ~244 pages or more).

!!! note
    These defaults assume that the cost of loading many btree pages is
    comparable to or worse than the cost of a fallback scan. That holds for
    cold object storage with high per-page latency, but on warm or local
    storage btree page reads often coalesce into a single sequential read and
    are cheaper than the fallback. Tune (or disable) the thresholds for your
    storage class — see the [BTree Index performance guide](../../../../guide/performance.md#btree-index)
    for end-to-end guidance.
