# BTree Index

The BTree index provides efficient range queries and sorted access by maintaining min/max statistics for data pages.

## Index Details

```protobuf
%%% proto.message.BTreeIndexDetails %%%
```

## Storage Layout

The BTree index consists of two files:

### Files

```
page_lookup.lance       # Page statistics and lookup
page_data.lance        # Actual data pages with values and row IDs
```

### Page Lookup Structure

```
page_lookup.lance
├── Schema
│   ├── min: {DataType}        # Minimum value in page
│   ├── max: {DataType}        # Maximum value in page
│   ├── page_id: UInt32        # Page identifier
│   └── row_id_range: Struct   # Row ID range in page
│       ├── start: UInt64
│       └── end: UInt64
└── Metadata
    └── batch_size: u64        # Default: 4096
```

### Page Data Structure

```
page_data.lance
├── Schema (varies by sub-index type)
│   ├── values: {DataType}     # Sorted values
│   └── ids: UInt64            # Corresponding row IDs
└── Data (Pages)
    ├── Page 0
    ├── Page 1
    └── ...
```

## Implementation Details

- **Page Organization**: Data is sorted and divided into fixed-size pages
- **Statistics**: Each page stores min/max values for pruning
- **Sub-indices**: Supports pluggable sub-index types (currently FlatIndex)
- **Null Handling**: Tracks pages that may contain null values
- **Lazy Loading**: Pages are loaded on-demand during queries