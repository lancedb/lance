# Indices in Lance

Lance supports three main categories of indices to accelerate data access:

1. **[Scalar Indices](scalar/index.md)** - Traditional indices for accelerating various database query patterns
2. **[Vector Indices](vector/index.md)** - Specialized indices for vector search
3. **[System Indices](system/index.md)** - Auxiliary indices for accelerating internal system operations

## Index Section in Manifest

Lance main protobuf manifest stores the file position of the index section,
so that the index section is not loaded when the dataset is opened,
and only loaded when needed:

```protobuf
optional uint64 index_section = 6;
```

## Index Metadata

Index section stores a list of index metadata:

```protobuf
%%% proto.message.IndexSection %%%

%%% proto.message.IndexMetadata %%%
```

### Index ID, Name and Delta Indices

Each index has a unique UUID. Multiple indexes of different IDs can share the same name.
When this happens, these indexes are called **Delta Indices** because they together forms a complete index.
Delta indices are typically used when the index is updated incrementally to avoid full rebuild.

## Index Coverage and Fragment Bitmap

An index records the fragments it covers using a bitmap of the `uint32` fragment IDs, 
so that during query planning phase, Lance can generate a split plan to leverage index for covered fragments,
and perform scan for uncovered fragments and merge the results.

## Index Remap and Row Address

In general, indexes describe how to find a row address based on some value of a column.
For example, a B-tree index can be used to find the row address of a specific value in a sorted array.

When compaction happens, because the row address has changed and some delete markers are removed, the index needs to be updated accordingly.
This update is fast because it's a pure mapping operation to delete some values or change the old row address to new row address.
We call this process **Index Remap**.
For more details, see [Fragment Reuse Index](fragment_reuse_index.md)

### Stable Row ID for Index

Using stable row ID to replace row address for index is a work in progress.
The main benefit is that remap is not needed, and an update only needs to invaludate the index if related column data has changed.
The tradeoff is that it requires an additional IOPS to translate row ID to row address.
We are still working on evaluating the performance impact of this change.


## Index Storage

The content of each index is stored at `_indices/{UUID}` directory under the dataset directory.
The actual content stored depends on the index type.
