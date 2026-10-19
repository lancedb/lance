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
2. `page_data.lance` - The actual sub-indices (flat file) containing sorted values and row addresses

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
| `ids`    | UInt64     | false    | Physical row address (`fragment_id << 32 \| offset`) corresponding to each value. A segment persisted before format version 1 (`IndexMetadata::index_version`) stores a row id here instead. |

## Stable Row ID Reverse Lookup

An ordinary BTree may index the logical `_rowid` column of a table with stable row
IDs enabled. Its `IndexMetadata.fields` contains the single reserved field ID `-3`.
The field resolves to a non-nullable `UInt64` logical column even when it is absent
from the table's user schema. Each fragment supplies that column from its inline
row-ID sequence or its hidden row-lineage column, as specified in
[Row ID and Lineage](../../table/row_id_lineage.md#inline-and-spilled-storage).

This index uses the existing BTree storage layout and address-domain format
version 1. The `values` column contains stable row IDs; the `ids` column contains
their physical row addresses. Page lookup extrema are over the stable-ID values.
No new index type, file, encoding, field ID, or address representation is introduced.
Version 0 must not be used for this reverse lookup: its payload denotes row IDs,
not the physical addresses needed to locate them.

The index has an ordinary logical index name and segment UUIDs. Multiple segments
may contribute to one logical index. A segment's fragment coverage identifies
where its entries are valid; it is not a statement that every ID between its
minimum and maximum is present. Missing or incomplete index coverage does not
mean that the corresponding rows are deleted. A reader must also resolve the
uncovered fragments' logical `_rowid` column before declaring an ID absent.

Resolution is bound to a dataset snapshot. A reader translates candidate addresses
through that snapshot's applicable fragment-reuse mappings, then applies effective
coverage and deletion visibility. Obsolete entries may remain in immutable index
files, including several candidates for a stable ID whose physical row moved.
Only one distinct live address may remain for a given stable ID within a branch
lineage; conflicting live addresses are an error. Compaction and index maintenance
use the same address-remapping and coverage rules as other address-domain indices.

Index creation, automatic maintenance, and consolidation scheduling are writer
policies rather than additional file formats. A writer must not claim coverage for
new or changed rows unless the published index entries and mappings support it.
Readers that do not support this BTree format version must treat the segment as
unsupported and use the ordinary uncovered-data path, subject to the table's other
feature flags. An existing version-0 segment keeps its original interpretation.

## Accelerated Queries

The BTree index provides exact results for the following query types:

| Query Type | Description               | Operation                                                                   |
|------------|---------------------------|-----------------------------------------------------------------------------|
| **Equals** | `column = value`          | BTree lookup to find relevant pages, then search within sub-indices         |
| **Range**  | `column BETWEEN a AND b`  | BTree traversal for pages overlapping the range, then search each sub-index |
| **IsIn**   | `column IN (v1, v2, ...)` | Multiple BTree lookups, union results from all matching sub-indices         |
| **IsNull** | `column IS NULL`          | Returns rows from all pages where null_count > 0                            |
