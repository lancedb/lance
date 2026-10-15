# Column Comparison Rules

Table-layer features that specify **Arrow Rust ordering** use the following
ascending value comparisons for the declared column type. Their own format
contracts define which values participate in a summary or index; for example,
version 1 [write seeds](../index/scalar/zonemap_write_seeds.md) exclude nulls and
NaNs from min/max and count them separately.

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
  and payloads.

Sort direction and null placement are separate from the value comparisons above.
This contract does not require bounds support for every Arrow type; the feature's
specification defines how unavailable bounds are represented.

Implementations must preserve these comparisons when upgrading Arrow. A change
to the dependency does not by itself change the persisted comparison contract.
These rules do not require the data-file reader or writer to interpret filters
or statistics.
