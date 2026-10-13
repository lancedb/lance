# Zone-level Column Statistics

A **write seed** is a saved summary of one column, collected while writing a
data file. It divides the column into **zones**, normally 8,192 rows each, and
stores one statistics record per zone in the same data file.

A reader can use these summaries to skip zones that cannot match a filter.
Combining zone summaries gives fragment or index-segment summaries for pruning
at a larger scale. An index builder can also reuse seeds instead of reading the
column again.

This proposal defines **version 1 of the write-seed format** and reserves its
dataset feature bit. Writing and reading version 1 seeds requires a follow-up
implementation. The existing ZoneMap index format and query behavior are unchanged.

## Arrow Rust ordering

Bounds follow **Arrow Rust's ascending value ordering** for the declared column
type. Writers and consumers must agree on these rules:

- Signed and unsigned integers compare by numeric value within their type.
- Booleans use `false < true`.
- Decimals compare their signed, unscaled integers within the declared scale.
- Dates, times, timestamps, and durations compare their underlying integer
  values within the declared type and unit. Timestamp timezone metadata does
  not change this comparison.
- Strings compare lexicographically by UTF-8 bytes; binary values compare
  lexicographically by bytes. A shorter prefix precedes a longer value. There
  is no locale collation, case folding, or whitespace normalization.
- Floats use IEEE 754 `totalOrder`: negative NaNs precede negative infinity,
  positive NaNs follow positive infinity, and `-0.0 < +0.0`. NaN representations
  are ordered within each sign group, including signaling/quiet distinctions
  and payloads. Version 1 bounds exclude all NaNs; counts describe them instead.

Nulls are counted separately. Descending sort options and null placement do not
change the ascending bounds. An ordering rule does not require bounds support
for every Arrow type; an unavailable bound must remain unknown.

Writers and readers must keep these rules when upgrading Arrow. If a query uses
different comparison rules, its reader may use these statistics to skip data
only when it can prove that no matching rows will be lost.

## Version 1 statistics fields

- `min` and `max` are lower and upper bounds on the zone's non-null, non-NaN
  values. For values `5, 6, 7, 8`, bounds `[2, 10]` are valid; `[6, 7]` are not.
  Positive and negative infinity are included in these bounds.
- `null_count` is the exact number of null rows.
- `nan_count` is the exact number of NaNs of either sign.
- `negative_nan_count` is the exact number of NaNs with the sign bit set.
  Positive NaNs number `nan_count - negative_nan_count`.
- `zone_length` is the actual number of physical rows summarized by the record,
  as in version 0. The locator's `rows_per_zone` gives the configured zone size;
  the final record can be shorter. For example, 10,000 rows at 8,192 rows per
  zone produce records with lengths 8,192 and 1,808.

All counts describe physical rows: the values written to the file, before any
later row deletions. Both NaN counts are required in version 1. Non-float columns
write zero for both. Counts must fit their declared types; writers must not
truncate them or replace an overflowing count with the type's maximum.

### When min/max are unavailable

For types without supported value bounds, such as whole `List`, `FixedSizeList`,
or `Struct` values, both bounds are null. These seeds can still store an exact
`null_count` and an optional null-row bitmap.

Statistics describe the column value at each row, not values nested inside it.
For example, a `List<Float64>` zone containing `null`, `[1.0, null, NaN]`, and
`[]` has null `min` and `max`, `null_count = 1`, and both NaN counts equal to
zero. If present, its null bitmap marks only the first row. The second row is
a non-null list; its child null and NaN are not counted. Likewise, a non-null
struct containing a null field is not itself null. These seeds do not provide
statistics for individual child fields.

A null bound does not, by itself, mean that all rows are null. Use the counts:

- If `zone_length = null_count + nan_count`, no non-null, non-NaN values remain.
- Otherwise, a null bound means that bound is unknown. A reader cannot use it
  to exclude values.

Each record must satisfy `null_count + nan_count <= zone_length` and
`negative_nan_count <= nan_count`. Readers must check these constraints without
overflowing the arithmetic.

## Storage layout

The data file's schema metadata tells the reader where to find each column's
seed. The key is `lance.seed.<column_name>` and the value is
`<buffer_index>:<rows_per_zone>`. For example, `lance.seed.price = "7:8192"`
means: read the `price` statistics from global buffer 7 in this file, with a
configured zone size of 8,192 rows. This locator is unchanged from version 0.

The buffer contains a complete Arrow IPC file, with its own schema and metadata.
For a column named `price`, the layout of a version 1 seed is:

```text
Data file
|-- Data-file schema metadata
|   `-- lance.seed.price = "7:8192"
|       Buffer index 7; configured zone size 8,192 rows
`-- Global buffer 7: Arrow IPC seed file
    |-- IPC schema metadata
    |   |-- version = "1": seed statistics version
    |   `-- seed_null_bitmap = <hex>: optional null row offsets
    `-- Statistics records: one record per zone
        min, max, null_count, nan_count, negative_nan_count, zone_length
```

The zone size is stored in the outer locator, not as a separate
`rows_per_zone` key in the seed's IPC schema metadata. The bitmap, when present,
covers the seed's physical row sequence; `null_count` is stored per zone.

The column's Arrow type is `value_type` in the schemas below. Seeds contain no
fragment ID or zone-start field. Record order determines the zone's local row
offset: the zero-based record number times `rows_per_zone`. The last zone may
be shorter. Consumers must place these ranges using the containing file's row
coverage, not assume that every data file covers an entire fragment.

### Version 1 seed schema

Each row in this schema is one zone's statistics record.

```python
pa.schema([
    pa.field("min", value_type, nullable=True),
    pa.field("max", value_type, nullable=True),
    pa.field("null_count", pa.uint32(), nullable=False),
    pa.field("nan_count", pa.uint32(), nullable=False),
    pa.field("negative_nan_count", pa.uint32(), nullable=False),
    pa.field("zone_length", pa.uint64(), nullable=False),
])
```

### Seed schema metadata

- `version`: the decimal string `"1"` for version 1. Absence means legacy
  version 0. This versions the seed statistics, not the Arrow IPC format or
  the containing data file.
- `seed_null_bitmap`: an optional bitmap identifying exactly which rows are
  null, stored as a hex-encoded Roaring bitmap. There is one bitmap per column's
  seed buffer, covering all its zones. Offsets are relative to the seed's
  physical row sequence. Its encoding is unchanged.

The bitmap lets an index builder obtain null positions without rereading the
column. It must agree with each zone's `null_count`. If absent, the null counts
are still available, but the exact positions are not. Version 1 keeps the
bitmap optional; min/max and counts do not depend on it.

## Consuming and aggregating statistics

Before using a seed, readers must check its version, column type, zone size,
and which physical rows it describes. If the version is unsupported, they can
read the column instead. A malformed seed of a known version is an error;
readers must not try to interpret it as version 0.

For pruning, use min/max for non-NaN values and the NaN counts separately.
Negative NaNs sort below non-NaN values; positive NaNs sort above them. The
counts do not identify individual NaN representations or their row positions.
If a predicate distinguishes NaNs with the same sign, retain any zone that
could match and check its rows.

For a complete, non-overlapping collection of zones:

- The lower bound is the minimum of their lower bounds; the upper bound is the
  maximum of their upper bounds, ignoring zones with no ordinary values.
- Null and NaN counts are summed.
- An unknown component bound remains unknown in the aggregate unless that
  component has no ordinary values. Unknown sign counts also remain unknown.

The same rules apply when aggregating fragment summaries. Segment pruning must
account for every row that the segment may return. Using bounds for whole
contributing fragments can be conservative, but exact counts require complete,
non-overlapping coverage. Missing statistics must never cause rows to be
excluded.

A consumer building an existing ZoneMap index must not copy version 1 seed
records verbatim into its unchanged format. It must explicitly adapt to that
format's contract or decline the seed and scan the column. This proposal does
not fix existing ZoneMap index reader behavior.

## Version 0: existing seeds

Existing seeds have no `version` metadata entry. Readers must interpret those
buffers as version 0, using this schema:

```python
pa.schema([
    pa.field("min", value_type, nullable=True),
    pa.field("max", value_type, nullable=True),
    pa.field("null_count", pa.uint32(), nullable=False),
    pa.field("nan_count", pa.uint32(), nullable=False),
    pa.field("zone_length", pa.uint64(), nullable=False),
])
```

Version 1 changes three things: it adds `version = "1"` to the seed's metadata,
adds `negative_nan_count`, and keeps all NaNs out of min/max.

Version 0 records only the total NaN count. Its `max` may contain a `+NaN`
marker even when all NaNs are negative. Readers must not treat that marker as
proof of a positive NaN, or as a bound on non-NaN values. When `nan_count > 0`,
either sign remains possible.

Existing buffers remain version 0. They must not be relabeled as version 1 or
assigned invented sign counts. Recollecting the column values can produce a
version 1 seed. An in-memory consumer may represent the unknown sign split
explicitly while preserving the exact legacy total.

## Reader and writer compatibility

The seed version tells a new reader how to decode the buffer. It cannot protect
an old reader that ignores the version entry. Dataset feature flags provide
that protection.

The first commit referencing a data file with version 1 seeds must set
`FLAG_ZONE_STATISTICS_SEEDS = 1 << 16` in both `Manifest.reader_feature_flags`
and `Manifest.writer_feature_flags`, atomically with that reference.

Clients that do not support the bit must refuse the dataset. Enabling version 1
is an explicit upgrade; opening an existing table does not enable it.

Both bits must be preserved in derived snapshots, including clone and restore.
Dropping an index does not clear them: data files can still contain seeds.
Existing historical snapshots are unchanged, and legacy seeds may coexist with
version 1 seeds.

The bit is reserved and unsupported until the seed writer, consumers, commit
paths, and maintenance implement this contract. Reserving it does not enable
version 1 emission or change any existing index format.
