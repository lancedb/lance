# Zone Map Index

Zone maps summarize small groups of physical rows in a column. Readers use the
summaries to skip zones that cannot match a predicate. Candidate rows still
require predicate rechecking.

## Index Details

```protobuf
%%% proto.message.ZoneMapIndexDetails %%%
```

The index stores its zone records in `zonemap.lance`. A complete optional null
bitmap can answer `IS NULL` exactly; without it, null counts identify candidates.

## Statistics versions

- **Version 0:** the existing format. It has no signed NaN count and writes a
  `+NaN` maximum marker whenever either NaN sign is present.
- **Version 1:** ordinary-value bounds and separate signed NaN counts, defined
  below. `IndexMetadata.index_version` is 1, and the file schema metadata
  `statistics_version` is the decimal string `"1"`.

A write seed records the same `statistics_version` in its Arrow IPC schema
metadata. Absence means version 0. Readers must validate the version before
using either an index or a seed; an unsupported version must not contribute
pruning decisions. Adding a version that old readers ignore is not a
compatibility fence: the manifest feature bits below provide that fence.

A version 1 file must contain the declared metadata and the nullable UInt32
sign-count field. Missing fields, wrong types, or invalid counts are malformed
version 1 statistics, not a reason to reinterpret the file as version 0.

## Version 1 statistics

`value_type` is the Arrow type used to store the indexed column's bounds.
`min` and `max` use its comparison ordering.

```python
pa.schema([
    pa.field("min", value_type, nullable=True),
    pa.field("max", value_type, nullable=True),
    pa.field("null_count", pa.uint32(), nullable=False),
    pa.field("nan_count", pa.uint32(), nullable=False),
    pa.field("negative_nan_count", pa.uint32(), nullable=True),
    pa.field("fragment_id", pa.uint64(), nullable=False),
    pa.field("zone_start", pa.uint64(), nullable=False),
    pa.field("zone_length", pa.uint64(), nullable=False),
])
```

### Bounds and counts

- **`min` and `max`:** lower and upper bounds on non-null, non-NaN values.
  Neither field may contain a NaN. Bounds may be wider than the exact extrema,
  but must contain every ordinary value in the zone.
- **`null_count`:** the exact number of null physical rows.
- **`nan_count`:** the exact number of NaNs of either sign.
- **`negative_nan_count`:** the number of those NaNs with the sign bit set.
  Positive NaNs number `nan_count - negative_nan_count`.
- **`fragment_id`, `zone_start`, `zone_length`:** the fragment and physical
  row range described by the record. Counts are not counts of rows remaining
  after deletion filtering.

Newly collected floating-point records must write a known sign count. Non-float
records write zero for both NaN counts. A null sign count is allowed only when
carrying statistics whose sign split is unknown, as described below. It never
means zero.

Floating-point comparisons follow Arrow Rust's IEEE total ordering, including
`-0.0 < +0.0`. Infinities are ordinary values. Negative NaNs precede all ordinary
values; positive NaNs follow them. Same-sign NaNs are also ordered by payload.

The counts must satisfy:

- `null_count + nan_count <= zone_length`;
- a known `negative_nan_count <= nan_count`.

Null bounds mean no ordinary values when `zone_length = null_count + nan_count`.
Otherwise a null bound means that side is unknown, including unsupported value
bounds and carried legacy summaries. Unknown bounds cannot justify exclusion.

### Using signed NaNs in queries

A reader considers three non-null groups separately: ordinary values, negative
NaNs, and positive NaNs. A zone stays a candidate if any group may match.

Conceptually, extend the ordinary range downward when negative NaNs are possible
and upward when positive NaNs are possible. This is query-time interpretation:
the reader does not put NaNs back into stored `min` or `max`, and does not compare
an integer count with a column value.

For range predicates:

- Ordinary values use their bounds. A negative NaN query boundary lies below
  them; a positive NaN boundary lies above them. Both ends must be checked.
- Negative NaNs may match if the lower end is unbounded or is a negative NaN,
  provided negative NaNs are possible.
- Positive NaNs may match if the upper end is unbounded or is a positive NaN,
  provided positive NaNs are possible.

A boundary within a possible same-sign NaN group must remain conservative:
counts do not record payloads, so one canonical NaN cannot serve as that group's
exact bound. Equality and `IN` with NaN literals also return candidates, not exact
NaN row addresses. Ordinary literals use ordinary bounds. Null predicates use
the null bitmap when available, or require rechecking count-based candidates.

A global-range API that returns only ordinary min/max must report an unknown
range when NaNs are present. It must not advertise a finite pair as a bound on
the entire column, excluding its NaN values.

## Reading legacy statistics

A version 1 reader must also understand version 0 indices and seeds:

- Missing `negative_nan_count` means an unknown sign split. If `nan_count > 0`,
  either sign remains possible; if the total is zero, neither is possible.
- A legacy `+NaN` maximum marker means the ordinary maximum is unknown. It does
  not prove that a positive NaN exists.
- A NaN stored as either bound must not be used as an ordinary-value bound.

Old files need no in-place rewrite. When copying legacy zones into a version 1
index, preserve their counts, replace any NaN bound with null, and retain null
for an unknown sign count. This records missing information without inventing
an ordinary maximum or requiring a data scan. Rebuilding a zone can replace
unknown information with newly measured bounds and counts.

## Reader and writer compatibility fence

Version 1 statistics require `FLAG_ZONEMAP_SIGNED_NAN_STATS = 1 << 16` in
**both** `Manifest.reader_feature_flags` and `Manifest.writer_feature_flags`.

The commit that first references a version 1 index or a data file containing a
version 1 seed must set both bits atomically. Merely upgrading a client or
opening a legacy table does not enable the feature. Clients that do not support
the bit must refuse the dataset before reading or maintaining these statistics.

This is a table-level compatibility fence, not an instruction for an old reader
to fall back to scanning. New readers retain the legacy decoding path. New
writers must preserve the contract and both bits during append, column rewrite,
index maintenance, cleanup, clone, and restore. The bits are sticky in derived
snapshots, including after dropping an index: retained data files may still
contain version 1 seeds. Existing historical snapshots are unchanged.

An index version or seed metadata version alone cannot protect existing readers
that ignore it. The feature bits are required even when the data-file encoding
itself is unchanged.

This proposal reserves the bit; implementations must not advertise support or
emit version 1 statistics until the reader, writer, commit, and maintenance
paths implement the complete contract.

## Write seeds

A seed is an Arrow IPC file in a data-file global buffer. Data-file schema
metadata `lance.seed.<column_name>` locates it as
`<buffer_index>:<rows_per_zone>`; version 1 adds the statistics schema metadata
defined above, not another positional component to that locator.

Its records omit fragment IDs and zone starts:

```python
pa.schema([
    pa.field("min", value_type, nullable=True),
    pa.field("max", value_type, nullable=True),
    pa.field("null_count", pa.uint32(), nullable=False),
    pa.field("nan_count", pa.uint32(), nullable=False),
    pa.field("negative_nan_count", pa.uint32(), nullable=True),
    pa.field("zone_length", pa.uint64(), nullable=False),
])
```

The reader obtains the fragment ID from the containing fragment and reconstructs
zone start as the zero-based record number times `rows_per_zone` (8,192 by
default). The last zone may be shorter. An incompatible zone size or unsupported
seed version must not be harvested; the index builder may scan the column.

Optional IPC schema metadata `seed_null_bitmap` contains a hex-encoded Roaring
bitmap of null row offsets. Missing metadata means exact offsets are unknown,
not that no nulls exist. This proposal does not change that bitmap encoding.

## Index metadata and global buffers

`rows_per_zone` is stored as a decimal string. Optional schema metadata
`null_bitmap` stores a decimal global-buffer index pointing to a serialized
`RowAddrTreeMap` of exact null row addresses. Missing metadata means no complete
null bitmap is available. These fields and the bitmap encoding are unchanged.
