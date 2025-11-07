# Row ID and Lineage Specification

## Overview

Lance provides row identification and lineage tracking capabilities.
Row addressing enables efficient random access to rows within the table through a physical location encoding.
Stable row IDs provide persistent identifiers that remain constant throughout a row's lifetime, even as its physical location changes.
Row version tracking records when rows were created and last modified, enabling incremental processing, change data capture, and time-travel queries.

## Row ID Styles

Lance uses two different styles of row IDs:

### Row Address

Row address is the physical location of a row in the table, represented as a 64-bit identifier composed of two 32-bit values:

```
row_address = (fragment_id << 32) | local_row_offset
```

This addressing scheme enables efficient random access: given a row address, the fragment and offset are extracted with bit operations.
Row addresses change when data is reorganized through compaction or updates.

Row address is currently the primary form of identifier used for indexing purposes.
Secondary indices (vector indices, scalar indices, full-text search indices) reference rows by their row addresses.

!!! note
      Work to support stable row IDs in indices is in progress.

### Stable Row ID

Stable Row ID is a unique auto-incrementing u64 identifier assigned to each row that remains constant throughout the row's lifetime, 
even when the row's physical location (row address) changes.
See the next section for more details.

!!! warning
      Historically, "row ID" was used to mean row address interchangeably. 
      With the introduction of stable row IDs, 
      there could be places in code and documentation that mix the terms "row ID" and "row address" or "row ID" and "stable row ID".
      Please raise a PR if you find any place incorrect or confusing.

## Stable Row ID

### Row ID Assignment

Row IDs are assigned using a monotonically increasing `next_row_id` counter stored in the manifest.

**Assignment Protocol:**

1. Writer reads the current `next_row_id` from the manifest at the read version
2. Writer assigns row IDs sequentially starting from `next_row_id` for new rows
3. Writer updates `next_row_id` in the new manifest to `next_row_id + num_new_rows`
4. If commit fails due to conflict, writer rebases:
   - Re-reads the new `next_row_id` from the latest version
   - Reassigns row IDs to new rows using the updated counter
   - Retries commit

This protocol mirrors fragment ID assignment and ensures row IDs are unique across all table versions.

### Row ID Behavior on Updates

When a row is updated, it is typically assigned a new row ID rather than reusing the old one.
This avoids the complexity of updating secondary indices that may reference the old values.

**Update Workflow:**

1. Original row with ID `R` exists at address `(F1, O1)`
2. Update operation creates new row with ID `R'` at address `(F2, O2)`
3. Deletion vector marks row ID `R` as deleted in fragment `F1`
4. Secondary indices referencing old row ID `R` are invalidated through fragment bitmap updates
5. New row ID `R'` requires index rebuild for affected columns

This approach ensures secondary indices do not reference stale data.

### Row ID Sequences

#### Storage Format

Row ID sequences are stored using the `RowIdSequence` protobuf message.
The sequence is partitioned into segments, each encoded optimally based on the data pattern.

<details>
<summary>RowIdSequence protobuf message</summary>

```protobuf
%%% proto.message.RowIdSequence %%%
```

</details>

#### Segment Encodings

Each segment uses one of five encodings optimized for different data patterns:

##### Range (Contiguous Values)

For sorted, contiguous values with no gaps.
Example: Row IDs `[100, 101, 102, 103, 104]` → `Range{start: 100, end: 105}`.
Used for new fragments where row IDs are assigned sequentially.

<details>
<summary>Range protobuf message</summary>

```protobuf
%%% proto.message.Range %%%
```

</details>

##### Range with Holes (Sparse Deletions)

For sorted values with few gaps.
Example: Row IDs `[100, 101, 103, 104]` (missing 102) → `RangeWithHoles{start: 100, end: 105, holes: [102]}`.
Used for fragments with sparse deletions where maintaining the range is efficient.

<details>
<summary>RangeWithHoles protobuf message</summary>

```protobuf
%%% proto.message.RangeWithHoles %%%
```

</details>

##### Range with Bitmap (Dense Deletions)

For sorted values with many gaps.
The bitmap encodes 8 values per byte, with the most significant bit representing the first value.
Used for fragments with dense deletion patterns.

<details>
<summary>RangeWithBitmap protobuf message</summary>

```protobuf
%%% proto.message.RangeWithBitmap %%%
```

</details>

##### Sorted Array (Sparse Values)

For sorted but non-contiguous values, stored as an `EncodedU64Array`.
Used for merged fragments or fragments after compaction.

##### Unsorted Array (General Case)

For unsorted values, stored as an `EncodedU64Array`.
Rare; most operations maintain sorted order.

#### Encoded U64 Arrays

The `EncodedU64Array` message supports bitpacked encoding to minimize storage.
The implementation selects the most compact encoding based on the value range, choosing between base + 16-bit offsets, base + 32-bit offsets, or full 64-bit values.

<details>
<summary>EncodedU64Array protobuf message</summary>

```protobuf
%%% proto.message.EncodedU64Array %%%
```

</details>

#### Inline vs External Storage

Row ID sequences are stored either inline in the fragment metadata or in external files.
Sequences smaller than ~200KB are stored inline to avoid additional I/O, while larger sequences are written to external files referenced by path and offset.
This threshold balances manifest size against the overhead of separate file reads.

<details>
<summary>DataFragment row_id_sequence field</summary>

```protobuf
message DataFragment {
  oneof row_id_sequence {
    bytes inline_row_ids = 5;
    ExternalFile external_row_ids = 6;
  }
}
```

</details>

### Row ID Index

#### Construction

The row ID index is built at table load time by aggregating row ID sequences from all fragments:

```
For each fragment F with ID f:
  For each (position p, row_id r) in F.row_id_sequence:
    index[r] = (f, p)
```

This creates a mapping from row ID to current row address.

#### Index Invalidation with Updates

When rows are updated, the row ID index must account for stale mappings:

**Example Scenario:**

1. Initial state: Fragment 1 contains rows with IDs `[1, 2, 3]` at offsets `[0, 1, 2]`
2. Update operation modifies row 2:
   - New fragment 2 created with row ID `4` (new ID assigned)
   - Deletion vector marks row ID `2` as deleted in fragment 1
3. Row ID index:
   - `1 → (1, 0)` ✓ Valid
   - `2 → (1, 1)` ✗ Invalid (deleted)
   - `3 → (1, 2)` ✓ Valid
   - `4 → (2, 0)` ✓ Valid (new row)

#### Fragment Bitmaps for Index Masking

Secondary indices use fragment bitmaps to track which row IDs remain valid:

**Without Row ID Updates:**

```
String Index on column "str":
  Fragment Bitmap: {1, 2}  (covers fragments 1 and 2)
  All indexed row IDs are valid
```

**With Row ID Updates:**

```
Vector Index on column "vec":
  Fragment Bitmap: {1}  (only fragment 1)
  Row ID 2 was updated, so index entry for ID 2 is stale
  Index query filters out ID 2 using deletion vectors
```

This bitmap-based approach allows indices to remain immutable while accounting for row modifications.

## Row Version Tracking

### Created At Version

Each row tracks the version at which it was created.
The sequence uses run-length encoding for efficient storage, where each run specifies a span of consecutive rows and the version they were created in.

Example: Fragment with 1000 rows created in version 5:
```
RowDatasetVersionSequence {
  runs: [
    RowDatasetVersionRun { span: Range{start: 0, end: 1000}, version: 5 }
  ]
}
```

<details>
<summary>DataFragment created_at_version_sequence field</summary>

```protobuf
message DataFragment {
  oneof created_at_version_sequence {
    bytes inline_created_at_versions = 9;
    ExternalFile external_created_at_versions = 10;
  }
}
```

</details>

<details>
<summary>RowDatasetVersionSequence protobuf messages</summary>

```protobuf
%%% proto.message.RowDatasetVersionSequence %%%
```

</details>

### Last Updated At Version

Each row tracks the version at which it was last modified.
When a row is created, `last_updated_at_version` equals `created_at_version`.
When a row is updated, a new row is created with both `created_at_version` and `last_updated_at_version` set to the current version, and the old row is marked deleted.

Example: Row created in version 3, updated in version 7:
```
Old row (marked deleted):
  created_at_version: 3
  last_updated_at_version: 3

New row:
  created_at_version: 7
  last_updated_at_version: 7
```

<details>
<summary>DataFragment last_updated_at_version_sequence field</summary>

```protobuf
message DataFragment {
  oneof last_updated_at_version_sequence {
    bytes inline_last_updated_at_versions = 7;
    ExternalFile external_last_updated_at_versions = 8;
  }
}
```

</details>

## Change Data Feed

Lance supports querying rows that changed between versions through version tracking columns.
These queries can be expressed as standard SQL predicates on the `_row_created_at_version` and `_row_last_updated_at_version` columns.

### Inserted Rows

Rows created between two versions can be retrieved by filtering on `_row_created_at_version`:

```sql
SELECT * FROM dataset
WHERE _row_created_at_version > {begin_version}
  AND _row_created_at_version <= {end_version}
```

This query returns all rows inserted in the specified version range, including the version metadata columns `_row_created_at_version`, `_row_last_updated_at_version`, and `_rowid`.

### Updated Rows

Rows modified (but not newly created) between two versions can be retrieved by combining filters on both version columns:

```sql
SELECT * FROM dataset
WHERE _row_created_at_version <= {begin_version}
  AND _row_last_updated_at_version > {begin_version}
  AND _row_last_updated_at_version <= {end_version}
```

This query excludes newly inserted rows by requiring `_row_created_at_version <= {begin_version}`, ensuring only pre-existing rows that were subsequently updated are returned.

