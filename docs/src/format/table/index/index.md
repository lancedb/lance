# Indices in Lance

Lance supports two main categories of indices to accelerate data access:

1. **[Scalar Indices](scalar/index.md)** - Traditional database-style indices for fast lookups and range queries
2. **[Vector Indices](vector/index.md)** - Specialized indices for approximate nearest neighbor (ANN) search
3. **[System Indices](system/index.md)** - Indices for internal system operations

## Index Section in Manifest

Lance main protobuf manifest stores the file position of the index section:

```protobuf
 optional uint64 index_section = 6;
```

and index section is stored at the specified start file position until the end of the manifest file:

```protobuf
%%% proto.message.IndexSection %%%
```


## Index Metadata

Index section stores a list of index metadata:

```protobuf
%%% proto.message.IndexMetadata %%%
```

### Delta Indices

Each index has a unique ID. Multiple indexes of different IDs can share the same name.
When this happends, these indexes are called Delta Indices because they together forms a complete index.

## Index Storage

Each index is stored at `_indices/{UUID}` directory under the dataset directory.
