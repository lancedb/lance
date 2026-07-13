# Row ID and Lineage Specification

## Overview

Lance distinguishes physical row locations from logical row identity. Physical
addresses support direct I/O. Logical identifiers survive physical rewrites.
Row version tracking records when rows were created and last modified.

## Row Identifier Forms

A row has two externally visible identifiers:

- **Row address** - the current physical location of the row in the dataset.
- **Row ID** - the row's logical identity. Storage version 2.3 always provides a
  stable row ID. In earlier versions it is stable only when the legacy stable
  row-ID feature is enabled; otherwise it equals the row address.


### Row Address

Row address is the physical location of a row in the table, represented as a 64-bit identifier composed of two 32-bit values:

```
row_address = (fragment_id << 32) | local_row_offset
```

This addressing scheme enables efficient random access: given a row address, the fragment and offset are extracted with bit operations.
Row addresses change when data is reorganized through compaction or updates.

`_rowaddr` and `_rowoffset` remain physical and unstable in storage version 2.3.
They must not be persisted as logical identity.

### Row ID

Row ID is a logical identifier for a row.

#### Stable Row ID

The `_rowid` system column exposes logical identity. In storage version 2.3 its
value is a stable logical row address. Earlier versions use either the legacy
stable-row-ID representation or the current physical row address.

#### Historical/unstable usage

Historically, the term "row id" was often used to refer to the physical row address (`_rowaddr`), which is not stable across compaction or updates.

!!! warning
      With the introduction of stable row IDs, there may still be places in code and documentation that mix the terms "row ID" and "row address" or "row ID" and "stable row ID".
      Please raise a PR if you find any place incorrect or confusing.

## Storage Version 2.3 Stable Logical Row Addresses

Storage version 2.3 has one row-identity mode. It is enabled for every newly
created 2.3 dataset, does not use a feature flag, and cannot be disabled. A 2.2
or earlier dataset cannot be migrated to 2.3 by overwrite, restore, or manifest
rewriting.

A stable logical row address is encoded as:

```
logical_row_address = (logical_fragment_id << 32) | immutable_slot
```

The dataset namespace UUID and the raw 64-bit address together form global
identity. A clone creates a new namespace while preserving raw addresses.

### Native Assignment

An append allocates one logical fragment ID per new physical fragment. Slots are
the physical offsets assigned at birth. The fragment stores one
`NativeLogicalDomain`; it does not store a per-row mapping. Physical and logical
fragment high-water marks never decrease, including across overwrite and
restore.

### Placement Root

Physical rewrites preserve logical addresses by replacing native ownership with
manifest-resident placement metadata. The placement root is canonical,
fingerprinted, and has a destination-side index. Its inline codecs are:

- `Direct` for one affine logical-to-physical extent;
- `PackedRun` for many logical fragments packed in source order;
- `Selected` for one exact selected source;
- `ExtentList` for a bounded number of source-contiguous extents;
- `SparseSelection` for selected rows from multiple logical fragments.

These codecs resolve an address to one physical owner in memory. A reader never
probes multiple fragments. `ExplicitMap` is reserved for explicit `Repack` and
`Recluster`; it uses a manifest page directory and an external locator, so its
cold lookup cost is part of those operations' public contract rather than the
default read path.

Every snapshot transitively owns the data, deletion, index, and placement
objects reachable from its manifest. Before an independent 2.3 shallow clone
publishes its target manifest, it writes a source-side lease under `_refs` for
the cloned source version. Source cleanup treats that lease like a protected
reference for as long as the target dataset prefix exists. This protects local
data and `ExplicitMap` locator objects across target successor manifests. A
descendant clone also writes a branch-wide lease at every inherited dataset
root, so deleting an intermediate clone cannot break the reachability chain.
Same-store cleanup releases a completed lease after the final target prefix is
removed. Cross-store liveness and an ambiguous failed commit are retained
conservatively; cleanup must never guess that the target is absent.

### Mutation and Liveness

Update and matched merge-insert rows retain `_rowid`, write a new physical row,
and tombstone the old physical row. Inserts allocate new logical domains.
Deletes first use deletion vectors. When maintenance removes the last physical
copy, the corresponding logical addresses become retired. Restore restores the
target snapshot's live relation but retains allocator high-water marks from the
actual head.

Content generations are tracked independently from identity and liveness. An
indexed field update advances the generation for the exact logical selection.
Generation regions can be retired only after every covering current index has
validated through them.

### Secondary Indices

Storage version 2.3 index postings contain stable logical row addresses and
carry exact logical coverage plus per-field generation watermarks. Default
compaction and fragment reuse do not rewrite postings or add a read-time remap
object. Query planning translates matched logical addresses through the inline
placement root before data take. Stale postings for changed indexed content and
retired rows are filtered in the logical domain before ranking results are
returned.

## Legacy Stable Row IDs (Storage Version 2.2 and Earlier)

Earlier storage versions optionally assign monotonically increasing row IDs and
persist per-fragment row-ID sequences. The remainder of this section specifies
that legacy representation; it does not describe storage version 2.3.

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

### Enabling Stable Row IDs

Stable row IDs are a dataset-level feature recorded in the table manifest.

- Stable row IDs **must be enabled when the dataset is first created**.
- Currently, they **cannot be turned on later** for an existing dataset. Attempts to write with `enable_stable_row_ids = true` against a dataset that was created without stable row IDs will not change the dataset's configuration.
- When stable row IDs are disabled, the `_rowid` column (if requested) is not stable and should not be used as a persistent identifier.

For storage versions through 2.2, row-level version tracking
(`_row_created_at_version`, `_row_last_updated_at_version`) and the row ID index
described below are only available when stable row IDs are enabled. Storage
version 2.3 always uses stable logical row addresses, so these pseudo-columns
are always available and there is no separate enablement mode.

### Row ID Behavior on Updates

When stable row IDs are enabled, updates preserve the logical row ID and remap it to a new physical address instead of assigning a new ID.

**Update Workflow:**

1. Original row with `_rowid = R` exists at address `(F1, O1)`.
2. An update operation writes a new physical row with the updated values at address `(F2, O2)`.
3. The new physical row is assigned the same `_rowid = R`, so the logical identifier is preserved.
4. The original physical row at `(F1, O1)` is marked deleted using the deletion vector for fragment `F1`.
5. The row ID index for the new dataset version maps `_rowid = R` to `(F2, O2)`, and uses deletion vectors and fragment bitmaps to avoid returning the tombstoned row at `(F1, O1)`.

This design keeps `_rowid` stable for the lifetime of a logical row while allowing physical storage and secondary indices to be maintained independently.

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

When rows are updated and stable row IDs are enabled, the row ID index for a given dataset version only contains mappings for live physical rows. Tombstoned rows are excluded using deletion vectors, and logical row IDs whose contents have changed simply map to new row addresses.

**Example Scenario:**

1. Initial state (version V): Fragment 1 contains rows with IDs `[1, 2, 3]` at offsets `[0, 1, 2]`.
2. An update operation modifies the row with `_rowid = 2`:
    - A new fragment 2 is created with a row for `_rowid = 2` at offset `0`.
    - In fragment 1, the original physical row at offset `1` is marked deleted in the deletion vector.
3. Row ID index in version V+1:
    - `1 → (1, 0)` ✓ Valid
    - `2 → (2, 0)` ✓ Valid (updated row in fragment 2)
    - `3 → (1, 2)` ✓ Valid

The address `(1, 1)` is no longer reachable via the row ID index because it is filtered out by the deletion vector when the index is constructed.

#### Fragment Bitmaps for Index Masking

Secondary indices use fragment bitmaps to track which row IDs remain valid:

**Without Row Updates:**

```
String Index on column "str":
  Fragment Bitmap: {1, 2}  (covers fragments 1 and 2)
  All indexed row addresses are valid
```

**With Row Updates:**

```
Vector Index on column "vec":
  Fragment Bitmap: {1}  (only fragment 1)
  The row with _rowid = 2 was updated, so the index entry that points to its old physical address is stale
  Index queries filter out the stale address using deletion vectors while returning the row at its new address
```

This bitmap-based approach allows indices to remain immutable while accounting for row modifications.

## Row Version Tracking

Row version tracking is available for datasets that use stable row IDs. Version sequences are aligned with the stable `_rowid` ordering within each fragment.

### Created At Version

Each row tracks the version at which it was created.
For rows that are later updated, this creation version remains the version in which the row first appeared; updates do not change it.
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

When stable row IDs are enabled and a row is updated, Lance writes a new physical row for the same logical `_rowid` while tombstoning the old physical row. The `created_at_version` for that logical row is preserved from the original row, and `last_updated_at_version` is set to the current dataset version at the time of the update.

In storage version 2.3, a logical domain's `creation_version` is the canonical
base for both values when a fragment does not carry an explicit per-row version
sequence. Directly appended domains use their commit version. Packed, selected,
and explicit placements retain the source domain's base across compaction and
placement maintenance. An update materializes only the exception needed to keep
`created_at_version` unchanged and advance `last_updated_at_version` for the
touched logical rows; absence of metadata must never be interpreted as version
1.

Example: Row created in version 3, updated in version 7:
```
Old physical row (tombstoned):
  _rowid: R
  created_at_version: 3
  last_updated_at_version: 3

New physical row (current):
  _rowid: R
  created_at_version: 3
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
