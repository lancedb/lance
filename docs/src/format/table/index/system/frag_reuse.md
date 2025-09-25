# Fragment Reuse Index

The Fragment Reuse Index is an internal index used to optimize fragment operations during compaction and dataset updates.

## Index Details

```protobuf
%%% proto.message.FragmentReuseIndexDetails %%%
```

## Purpose

The Fragment Reuse Index tracks how row IDs are remapped when fragments are reorganized during:
- Compaction operations
- Fragment merging
- Dataset optimization

## Storage

This index is typically stored in memory and serialized as part of the index metadata. It maintains mappings between old and new row addresses to ensure other indices remain valid after fragment reorganization.

## Implementation Details

- **Row ID Remapping**: Maps old row addresses to new ones after compaction
- **Index Coordination**: Ensures all indices are updated consistently
- **Lazy Updates**: Remapping is applied on-demand during index queries