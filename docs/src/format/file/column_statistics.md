# Zone-level Column Statistics

A write seed stores column statistics collected while writing a data file.
Each record summarizes a zone of physical rows, normally 8,192 rows. The records
are embedded in the data file as an Arrow IPC buffer, separate from column data.

Consumers can use these statistics to prune zones or aggregate them for fragment
and index-segment pruning. They can also build statistics without scanning the
column again. This specification changes only the write-seed format, not the
ZoneMap index schema, index version, or query implementation.

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

These rules are a format contract. An Arrow dependency upgrade must not silently
change them. A consumer using different predicate comparisons must not exclude
rows unless it can prove that the statistics remain conservative for those
comparisons.

## Version 1 statistics fields

- `min` and `max` bound all non-null, non-NaN values in the zone. Bounds may be
  wider than the exact extrema but must never exclude such a value. Infinities
  are ordinary values. For types without supported value bounds, such as whole
  `List`, `FixedSizeList`, or `Struct` values, both bounds are null. This means
  the bounds are unavailable, not that every value is null. Such seeds can still
  contain an exact `null_count` and an optional null-row bitmap.
- `null_count` is the exact number of null physical rows.
- `nan_count` is the exact number of NaNs of either sign.
- `negative_nan_count` is the exact number of NaNs with the sign bit set.
  Positive NaNs number `nan_count - negative_nan_count`.
- `zone_length` is the actual number of physical rows summarized by the record,
  as in version 0. The locator's `rows_per_zone` gives the configured zone size;
  the final record can be shorter. For example, 10,000 rows at 8,192 rows per
  zone produce records with lengths 8,192 and 1,808.

Counts describe the values written to the file, before later deletion filtering.
Version 1 seeds are newly collected records: both NaN counts are non-nullable.
Non-float columns write zero for both counts. Counts must fit their declared
types; writers must not truncate or saturate them.

Statistics describe the column value at each row, not values nested inside it.
For example, a `List<Float64>` zone containing `null`, `[1.0, null, NaN]`, and
`[]` has null `min` and `max`, `null_count = 1`, and both NaN counts equal to
zero. If present, its null bitmap marks only the first row. The second row is
a non-null list; its child null and NaN are not counted. Likewise, a non-null
struct containing a null field is not itself null. These seeds do not provide
statistics for individual child fields.

Each record must satisfy `null_count + nan_count <= zone_length` and
`negative_nan_count <= nan_count`. Arithmetic used to validate these constraints
must not overflow.

Null bounds mean there are no ordinary values when
`zone_length = null_count + nan_count`. Otherwise a null bound means that side
is unknown, for example because value bounds are unsupported for the type.
An unknown bound cannot justify pruning.

## Storage layout

Data-file schema metadata `lance.seed.<column_name>` locates the seed as
`<buffer_index>:<rows_per_zone>`. The buffer index addresses a global buffer in
the containing data file. The locator is unchanged; its default zone size is
8,192 rows.

The buffer contains a complete Arrow IPC file, with its own schema and metadata.
The column's Arrow type is `value_type` in the schemas below. Seeds contain no
fragment ID or zone-start field. Record order determines the zone's local row
offset: the zero-based record number times `rows_per_zone`. The last zone may
be shorter. Consumers must place these ranges using the containing file's row
coverage, not assume that every data file covers an entire fragment.

### Seed buffer schema

#### Version 1

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

#### Version 0 (legacy)

```python
pa.schema([
    pa.field("min", value_type, nullable=True),
    pa.field("max", value_type, nullable=True),
    pa.field("null_count", pa.uint32(), nullable=False),
    pa.field("nan_count", pa.uint32(), nullable=False),
    pa.field("zone_length", pa.uint64(), nullable=False),
])
```

Legacy seeds have no sign count. Their `max` may contain a `+NaN` marker when
either NaN sign is present; it is not evidence of a positive NaN.

### Seed schema metadata

- `version`: the decimal string `"1"` for version 1. Absence means legacy
  version 0. This versions the seed statistics, not the Arrow IPC format or
  the containing data file.
- `seed_null_bitmap`: optional hex-encoded Roaring bitmap of null row offsets
  in the seed's physical row sequence. Its encoding is unchanged. Absence means
  exact offsets are unavailable, not that the data has no nulls.

The optional bitmap must agree with `null_count`. Version 1 does not require
adding or removing a bitmap; min/max and counts do not depend on its presence.

## Consuming and aggregating statistics

Consumers must validate the seed version, column type, zone size, and actual row
coverage before using it. Unsupported versions are unusable seeds; a consumer
can scan the column instead. A malformed known version is an error, not a reason
to reinterpret the buffer as version 0.

For zone pruning, ordinary values use the bounds. Negative NaNs may match below
ordinary values; positive NaNs may match above them. Counts do not record NaN
payloads, so predicates within a possible same-sign NaN group require
conservative candidates and row rechecking. Counts are not row addresses.

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

## Reading legacy seeds

An absent version selects the legacy decoder. The total NaN count is known, but
the sign split is unknown. If `nan_count > 0`, either sign remains possible.
A legacy NaN bound must not be interpreted as an ordinary-value bound.

Existing buffers remain version 0. They must not be relabeled as version 1 or
assigned invented sign counts. Recollecting the column values can produce a
version 1 seed. An in-memory consumer may represent the unknown sign split
explicitly while preserving the exact legacy total.

## Reader and writer compatibility

The first commit referencing a data file with version 1 seeds must set
`FLAG_ZONE_STATISTICS_SEEDS = 1 << 16` in both `Manifest.reader_feature_flags`
and `Manifest.writer_feature_flags`, atomically with that reference.

Clients that do not support the bit must refuse the dataset. The seed version
alone cannot fence older consumers that ignore version metadata. This is an
explicit capability upgrade; opening an existing table does not enable it.

Both bits must be preserved in derived snapshots, including clone and restore.
Dropping an index does not clear them: data files can still contain seeds.
Existing historical snapshots are unchanged, and legacy seeds may coexist with
version 1 seeds.

The bit is reserved and unsupported until the seed writer, consumers, commit
paths, and maintenance implement this contract. Reserving it does not enable
version 1 emission or change any existing index format.
