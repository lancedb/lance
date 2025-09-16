# Zone Map Index

Zone maps store statistical metadata about data zones to enable efficient scan pruning.

## Index Details

```protobuf
%%% proto.message.ZoneMapIndexDetails %%%
```

## Storage Layout

The zone map index is integrated with the BTree index implementation, using the same file structure but optimized for zone-level statistics rather than individual values.

### Files

```
page_lookup.lance       # Zone statistics
page_data.lance        # Zone data (if sub-index is used)
```

### Zone Statistics Structure

```
page_lookup.lance
├── Schema
│   ├── min: {DataType}        # Minimum value in zone
│   ├── max: {DataType}        # Maximum value in zone
│   ├── page_id: UInt32        # Zone identifier
│   └── row_id_range: Struct   # Row ID range in zone
│       ├── start: UInt64
│       └── end: UInt64
└── Metadata
    └── batch_size: u64        # Zone size (rows per zone)
```

## Implementation Details

- **Zone Size**: Configurable via `zone_size` parameter (default: 4096 rows)
- **Statistics**: Min/max values per zone for pruning
- **Null Tracking**: Records zones containing null values
- **Pruning**: Zones are skipped if their min/max range doesn't overlap with query predicates