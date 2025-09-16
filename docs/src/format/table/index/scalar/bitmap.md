# Bitmap Index

Bitmap indices use bit arrays to represent the presence or absence of values, providing extremely fast query performance for low-cardinality columns.

## Index Details

```protobuf
%%% proto.message.BitmapIndexDetails %%%
```

## Storage Layout

The bitmap index consists of a single file `bitmap_page_lookup.lance` that stores the mapping from values to their bitmaps:

### File Structure

```
bitmap_page_lookup.lance
├── Schema
│   ├── keys: {DataType} (nullable)     # The unique values
│   └── bitmaps: Binary                 # Serialized RowIdTreeMap
└── Data (RecordBatches)
    ├── Batch 1 (up to 2048 rows)
    ├── Batch 2
    └── ...
```

### Implementation Details

- **Value Dictionary**: Each unique value (including null) maps to a row offset in the file
- **Bitmap Storage**: Bitmaps are serialized as `RowIdTreeMap` and stored as binary data
- **Chunking**: Data is written in batches of up to 2048 rows to avoid memory issues
- **Binary Array Limit**: Each batch's bitmap data cannot exceed ~2GB (i32::MAX - 1MB headroom)
- **Null Handling**: Null values are stored as a special entry in the index

### Loading Strategy

The index uses lazy loading:
1. Only the value-to-offset mapping is loaded initially
2. Individual bitmaps are loaded on-demand during queries
3. Loaded bitmaps are cached using `LanceCache`
