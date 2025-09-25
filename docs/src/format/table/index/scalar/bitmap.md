# Bitmap Index

Bitmap indices use bit arrays to represent the presence or absence of values,
providing extremely fast query performance for low-cardinality columns.

## Index Details

```protobuf
%%% proto.message.BitmapIndexDetails %%%
```

## Storage Layout

The bitmap index consists of a single file `bitmap_page_lookup.lance` that stores the mapping from values to their bitmaps.

### File Schema

| Column    | Type       | Nullable | Description                                                         |
|-----------|------------|----------|---------------------------------------------------------------------|
| `keys`    | {DataType} | true     | The unique value from the indexed column                            |
| `bitmaps` | Binary     | true     | Serialized RowIdTreeMap containing row IDs where this value appears |

## Accelerated Queries

| Query Type | Description               | Operation                                  |
|------------|---------------------------|--------------------------------------------|
| **Equals** | `column = value`          | Returns the bitmap for the specific value  |
| **Range**  | `column BETWEEN a AND b`  | Unions all bitmaps for values in the range |
| **IsIn**   | `column IN (v1, v2, ...)` | Unions bitmaps for all specified values    |
| **IsNull** | `column IS NULL`          | Returns the pre-computed null bitmap       |
