# Label List Index

Label list indices are optimized for columns containing multiple labels or tags per row.
They provide efficient set-based queries on multi-value columns using an underlying bitmap index.

## Index Details

```protobuf
%%% proto.message.LabelListIndexDetails %%%
```

## Storage Layout

The label list index uses a bitmap index internally and stores its data in:

1. `bitmap_page_lookup.lance` - Bitmap index mapping unique labels to row IDs

### File Schema

| Column    | Type       | Nullable | Description                                                         |
|-----------|------------|----------|---------------------------------------------------------------------|
| `keys`    | {DataType} | true     | The unique label value from the indexed column                      |
| `bitmaps` | Binary     | true     | Serialized RowIdTreeMap containing row IDs where this label appears |

## Accelerated Queries

The label list index provides exact results for the following query types:

| Query Type           | Description                            | Operation                                   | Result Type |
|----------------------|----------------------------------------|---------------------------------------------|-------------|
| **array_has_all**    | Array contains all specified values    | Intersects bitmaps for all specified labels | Exact       |
| **array_has_any**    | Array contains any of specified values | Unions bitmaps for all specified labels     | Exact       |