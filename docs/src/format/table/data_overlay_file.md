# Data Overlay Files

!!! warning "Experimental"

    This feature is currently experimental and not yet supported in any library.

<!-- TODO: When overlay file support is implemented, update this note to state
     the released version that first supports the feature (and drop the
     "experimental" framing once it is stable). -->

!!! note "Overlay files require feature flag 64 (data overlay files)"

    A reader or writer that does not understand overlay files must refuse a
    dataset that uses them. Silently ignoring an overlay would return stale base
    values, which is a correctness bug rather than a degraded experience.

Overlay files supply new values for a subset of `(row offset, field)` cells
within a fragment **without rewriting the fragment's base data files**. They make
updates cheap when only a small fraction of rows and/or columns change: instead
of rewriting whole columns or moving rows to a new fragment, a writer appends a
small file carrying just the changed cells.

This is Lance's third mechanism for changing data in place, alongside
[deletion files](index.md#deletion-files) (which remove rows) and
[data evolution](index.md#data-evolution) (which adds or rewrites whole columns).
An overlay changes individual cells.

## Concepts

### Coverage and resolution

Each overlay declares which cells it provides through a **coverage** bitmap (or,
for sparse overlays, one bitmap per field). The bitmaps index **physical row
offsets** — positions in the base data files, counting deleted rows — so they are
stable across deletions, exactly like deletion vectors.

To resolve a cell `(offset, field)` on read, walk the fragment's overlays from
**newest to oldest**. The first overlay that covers `(offset, field)` wins; its
value is used. If no overlay covers the cell, the value falls through to the base
data file (or is `NULL` if no base data file holds that field).

Precedence among overlays is determined by:

1. `committed_version` — higher wins (see [Versioning](#versioning-and-ordering)).
2. Position in `DataFragment.overlays` as a tiebreaker — a later entry is newer.

A covered offset whose value is `NULL` overrides the cell **to** `NULL`. This is
distinct from an offset that is simply absent from the bitmap, which falls
through to the base. Coverage, not value-nullness, decides whether an overlay
applies.

### Interaction with deletions

Deletions take precedence over overlays. If a row offset is marked deleted in the
fragment's deletion file, any overlay value for that offset is dead and is
ignored, regardless of commit order. This keeps the invariant simple: a deletion
is the final word on a row, so a concurrent overlay against a row that was
deleted needs no special conflict handling — its values are merely inert.

### Physical layout

An overlay's data file stores **one value column per field**, in the order of
`data_file.fields`. It does **not** store a row-offset key column. The position of
a covered offset's value within its column is the **rank** of that offset in the
field's coverage bitmap — the number of set bits below it. For a Roaring bitmap
this is an O(1) operation, so random access to any cell is a rank computation
followed by a single value fetch, with no offset column to read and no binary
search.

Because different fields may cover different offset sets, the value columns of a
single sparse overlay may have **different lengths**. The Lance file format
permits columns of differing item counts within one file, so a sparse overlay is
representable as a single file. (See [Writer support](#writer-support) for the
current implementation status.)

### Dense vs. sparse overlays

A single overlay is one of two shapes:

- **Dense (rectangular).** One `shared_offset_bitmap` applies to every field. Every
  covered offset has a value for every field. This is the common case for a plain
  `UPDATE`, where one `SET` list is applied to one set of rows.
- **Sparse.** A `FieldCoverage` carries one bitmap per field, used when different
  fields cover different offset sets — for example a `MERGE` with multiple
  `WHEN MATCHED` branches, where different rows update different columns. A dense
  overlay would have to widen to the bounding rectangle and fill the untouched
  cells with their current values (post-images), which for wide columns such as
  embeddings means re-storing data that did not change. A sparse overlay stores
  exactly the changed cells.

A writer may always express a non-rectangular update as **multiple dense overlays
in one transaction** (one per coverage group) instead of a single sparse overlay.

## Protobuf

<details>
<summary>DataOverlayFile protobuf message</summary>

```protobuf
%%% proto.message.DataOverlayFile %%%
```

</details>

<details>
<summary>FieldCoverage protobuf message</summary>

```protobuf
%%% proto.message.FieldCoverage %%%
```

</details>

## Versioning and ordering

Overlays reuse the dataset version as their ordering clock rather than
introducing a separate generation counter.

`committed_version` is the dataset version at which an overlay **became
effective** — the version of the commit that introduced it, **not** the version
it was read from. It is stamped at commit time and re-stamped if the commit is
retried, in the same way as the created-at / last-updated-at version sequences.

This single value drives every ordering decision:

- **Overlay vs. overlay** (read precedence): higher `committed_version` wins.
- **Overlay vs. index** (query correctness): an index records the
  `dataset_version` it was built from. An index whose `dataset_version >=
  committed_version` already incorporates the overlay. An overlay whose
  `committed_version > index.dataset_version` is newer than the index and its
  cells must be excluded from index results and re-evaluated.
- **Scheduler signal**: the gap between an overlay's `committed_version` and an
  index's `dataset_version`, or between an overlay and the base, is a staleness
  measure the compaction scheduler can use.

!!! note "Why effective version, not read version"

    Suppose an overlay reads version 5 and commits at version 6, while an index
    is built reading version 5 (before the overlay) and commits at version 7 with
    `dataset_version = 5`. If the overlay stored its *read* version (5), the test
    `5 > 5` is false, the row would not be excluded, and the index — which never
    saw the overlay — would return a stale result. Storing the *effective*
    version (6) makes `6 > 5` true, the cell is excluded and re-evaluated, and the
    result is correct.

## Index integration

Building an index over a fragment that has overlays does **not** require dropping
the fragment from the index's coverage. The fragment stays indexed, and the query
path reconciles overlays at query time using an **exclusion set**.

The exclusion set for an index on field `F` is the union of the coverage bitmaps,
restricted to field `F`, of every overlay whose `committed_version >
index.dataset_version`. The exclusion is **field-aware**: an overlay that touches
only unrelated columns does not exclude anything from the index on `F`.

The query then proceeds as:

1. Run the index search as usual, producing candidate rows.
2. Remove any candidate in the exclusion set. (Its indexed value may be stale.)
3. **Re-evaluate** the excluded rows against their current values — the same flat
   path already used for the unindexed tail of fragments. For a scalar predicate
   this re-applies the filter; for a vector query it re-scores the row's current
   vector. Rows that still match are added back to the result.

Step 3 is what makes exclusion correct rather than merely safe: removing a row
from index candidates without re-evaluating it would silently drop a row that
should match under its new value.

### Correctness invariant

> For every indexed field `F` and every row offset `o` in a fragment the index
> covers, the index's entry for `(o, F)` is trusted unless `o` is excluded.
> `o` is excluded iff some overlay with `committed_version > index.dataset_version`
> covers `(o, F)`.

The write and compaction paths together preserve this:

- **Writes** change a cell only by adding an overlay, and that overlay's
  `committed_version` exceeds the version of any pre-existing index — so the
  change is always covered by an exclusion.
- **Compaction** may remove an overlay only if the index no longer relies on it
  (see below).

## Compaction

Overlays accumulate read cost — every overlay is a bitmap to test and a possible
file to open. Compaction bounds that cost in two modes:

- **Overlay → overlay.** Merge several overlays into fewer, computing the
  post-image per `(offset, field)` by walking the merged overlays newest-first.
  The merged overlay takes the **maximum** `committed_version` of its inputs, so
  the exclusion semantics are preserved. Indexes are unaffected. This is cheap and
  does not touch the base.
- **Overlay → base.** Fold overlays into a fresh base data file, computing the
  post-image for every covered cell, then clear the overlays. The base is
  complete, so every post-image is well defined. Overlay offsets are physical, so
  they cannot survive a rewrite that reorders rows; folding therefore materializes
  values rather than carrying overlays forward.

!!! warning "Folding an indexed field must update its index"

    An overlay→base fold removes the overlay, which removes the exclusion signal
    that kept an index correct. Folding an overlay that covers an indexed field
    `F` is therefore equivalent to a column rewrite of `F` and must, in the same
    commit, either rebuild the index to a `dataset_version` at least the folded
    overlay's `committed_version`, or remove the fragment from the index's
    coverage so the rows fall to the flat path. Otherwise the index would serve
    stale values with no overlay to exclude them. This is the same rule that
    already governs rewriting a column that an index is built on.

When a fragment with overlays is compacted by a row-rewriting operation
(`RewriteRows`, which produces new fragments with new row addresses), the
overlays are folded into the new base as part of the rewrite, and existing
[fragment-reuse remapping](row_id_lineage.md) handles the row-address changes as
it does today.

## Row lineage

An overlay write updates the `last_updated_at_version` of every covered row, so
change-data-feed and time-travel queries observe the update. Because overlays are
addressed by physical offset, they do **not** require stable row IDs to be
enabled; lineage updates apply only when those features are on.

## Worked example

A table `users` with stable row IDs enabled and these fields:

| field id | name      | type                    |
|----------|-----------|-------------------------|
| 1        | id        | `int32` (primary key)   |
| 2        | name      | `utf8`                  |
| 3        | age       | `int32`                 |
| 4        | embedding | `fixed_size_list<f32,4>`|

Created at version 1 as a single fragment `0` with one base data file
`data/file0.lance` holding all four columns. `physical_rows = 4`:

| offset | id | name  | age | embedding        |
|--------|----|-------|-----|------------------|
| 0      | 1  | Alice | 30  | …                |
| 1      | 2  | Bob   | 25  | …                |
| 2      | 3  | Carol | 40  | …                |
| 3      | 4  | Dave  | 22  | …                |

A BTree scalar index on `age` is built at version 1, covering fragment `0`
(`dataset_version = 1`).

### Step 1 — write an overlay

```sql
UPDATE users SET age = 26 WHERE id = 2;   -- Bob, offset 1
```

This touches one field (`age`) for one row, so the writer emits a dense overlay
and commits it as version 2. Fragment `0` gains:

```text
DataOverlayFile {
  data_file: { path: "data/overlay-<uuid>.lance", fields: [3], column_indices: [0] }
  coverage:  shared_offset_bitmap = {1}
  committed_version: 2
}
```

The overlay file stores a single `age` column with one value, `[26]`, at
rank `{1}.rank(1) = 0`. `last_updated_at_version[1]` is set to 2.

### Step 2 — read

`SELECT id, age FROM users` reads base ages `[30, 25, 40, 22]`. For `age`
(field 3), the overlay covers offset 1, so `age[1]` is replaced with the overlay
value at position `{1}.rank(1) = 0` → `26`. Result ages: `[30, 26, 40, 22]`.

### Step 3 — index query

```sql
SELECT * FROM users WHERE age = 26;
```

The `age` index was built at `dataset_version = 1`; the overlay's
`committed_version` is 2. Since `2 > 1`, the overlay's coverage for `age`, `{1}`,
is the exclusion set for this query.

- The index (built at v1) holds Bob's *old* `age = 25`, so a lookup for `26`
  returns nothing from the index.
- Offset 1 is in the exclusion set, so it is re-evaluated on the flat path. Its
  current `age` (26, via the overlay) matches `age = 26`, so Bob is returned.

The mirror case `WHERE age = 25` shows exclusion preventing a stale hit: the index
returns offset 1 (stale `25`), but offset 1 is excluded, re-evaluated to `26`, and
correctly dropped.

### Step 4 — a second, non-rectangular write

```sql
MERGE INTO users USING staged ON users.id = staged.id
WHEN MATCHED AND staged.kind = 'rename'  THEN UPDATE SET name = staged.name        -- Carol(2), Dave(3)
WHEN MATCHED AND staged.kind = 'revec'   THEN UPDATE SET embedding = staged.embedding -- Bob(1)
```

`name` is updated for offsets `{2, 3}` and `embedding` for offset `{1}` — different
fields over different rows. This is a sparse overlay, committed as version 3:

```text
DataOverlayFile {
  data_file: { path: "data/overlay-<uuid2>.lance", fields: [2, 4], column_indices: [0, 1] }
  coverage:  field_coverage { offset_bitmaps: [ {2,3},   {1} ] }
  //                                     name (field 2)  ^    ^ embedding (field 4)
  committed_version: 3
}
```

The file's `name` column has **two** values (`["Caroline", "David"]`, at
ranks 0 and 1 of `{2,3}`) and its `embedding` column has **one** value (at rank 0
of `{1}`) — columns of different lengths in one file.

### Step 5 — read after the second write

`SELECT name, age, embedding FROM users` resolves each field independently,
newest overlay first:

- `name`: the v3 overlay covers `{2,3}` → `["Alice", "Bob", "Caroline", "David"]`.
- `age`: the v3 overlay does not cover `age`; the v2 overlay still applies at
  offset 1 → `[30, 26, 40, 22]`.
- `embedding`: the v3 overlay covers `{1}` → Bob's vector is the new one, others
  from base.

Overlays from different versions coexist and apply per field.

### Step 6 — compaction (overlay → base)

The scheduler folds both overlays into fragment `0` at version 4, computing
post-images for `age`, `name`, and `embedding`, and writing a new base data file
`data/file1.lance` with those columns. In the old file, fields 2, 3, and 4 are
tombstoned (`-2`); field 1 (`id`) remains. The fragment's `overlays` list is
cleared. Row addresses are preserved (a column rewrite, not a row rewrite), so
stable row IDs and the deletion vector are untouched.

Because the fold removed the overlay that was excluding offset 1 from the `age`
index, the same commit must reconcile that index: either rebuild it at
`dataset_version >= 2`, or drop fragment `0` from its coverage so `age` queries
fall to the flat path. After a rebuild at version 4, no overlay remains and the
`age` index directly returns `26` for Bob with no exclusion needed.

## Guidance

!!! note "This section is a stub."

    The following are implementation considerations, not part of the on-disk
    specification.

### When to overlay vs. rewrite a column vs. move rows

<!-- TODO: Replace the rough heuristic below with concrete thresholds once we
     have benchmarked the crossover points between overlays, column rewrites,
     and row moves. -->

*(To be expanded.)* The choice between appending an overlay, rewriting a full
column (data evolution), and moving updated rows to a new fragment depends on the
fraction of rows changed, the fraction of columns changed, column width, the
presence of indexes on the changed columns, and the accumulated overlay read
cost. Roughly: few rows changed favors overlays; most rows in a few columns
favors a column rewrite; most columns changed favors moving rows to a new
fragment.

### Writer support

<!-- TODO: Fill in as writer implementation progresses, including the status of
     single-file sparse overlays (independent-length columns). -->

*(To be expanded.)* Dense (rectangular) overlays write with the existing
equal-length file writer today. Sparse overlays stored as a **single** file
require the writer to emit columns of independent lengths, which the current v2
writer does not yet do (it advances all columns from one global row counter).
Until that support lands, a writer can express a sparse update as multiple dense
overlays in one transaction.

### Scheduling compaction

<!-- TODO: Fill in with a concrete cost/benefit policy once compaction is
     implemented and benchmarked. -->

*(To be expanded.)* The overlay→overlay and overlay→base modes have very
different costs; a cost/benefit scheduler decides when each is worthwhile, using
the version gap as a staleness signal.

### Open questions

*(To be resolved.)*

- **Per-fragment vs. per-table overlays.** Overlays are attached per fragment.
  Should there be a table-level overlay concept, and how would it interact with
  fragment-level row addressing?
- **Relationship to LSM.** Overlays plus compaction resemble an LSM tree (newest
  layer wins, periodic merge). How far should that analogy be taken, and what do
  we deliberately do differently given Lance's random-access requirements?
- **Coverage bitmap spill.** Coverage bitmaps live inline in the manifest. Very
  large coverage (an overlay touching many rows) may warrant external spill, as
  the row-ID and last-updated-at sequences already do above a size threshold.

## Related specifications

- [Table format overview](index.md)
- [Transactions](transaction.md)
- [Row ID & Lineage](row_id_lineage.md)
- [Index Formats](../index/index.md)
- [Format Versioning](versioning.md)
