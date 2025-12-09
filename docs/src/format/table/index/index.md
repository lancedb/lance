# Indices in Lance

Lance supports three main categories of indices to accelerate data access:

1. **Scalar Indices** - Traditional indices for accelerating various database query patterns
2. **Vector Indices** - Specialized indices for vector search
3. **System Indices** - Auxiliary indices for accelerating internal system operations

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

Each index has a unique UUID. Multiple indices of different IDs can share the same name.
When this happens, these indices are called **Delta Indices** because they together form a complete index.
Delta indices are typically used when the index is updated incrementally to avoid full rebuild.
The Lance SDK provides functions for users to choose when to create delta indices,
and when to merge them back into a single index.

### Index Coverage and Fragment Bitmap

An index records the fragments it covers using a bitmap of the `uint32` fragment IDs, 
so that during the query planning phase, Lance can generate a split plan to leverage the index for covered fragments,
and perform scan for uncovered fragments and merge the results.

### Index Remap and Row Address

In general, indices describe how to find a row address based on some value of a column.
For example, a B-tree index can be used to find the row address of a specific value in a sorted array.

When compaction happens, because the row address has changed and some delete markers are removed, the index needs to be updated accordingly.
This update is fast because it's a pure mapping operation to delete some values or change the old row address to the new row address.
We call this process **Index Remap**.
For more details, see [Fragment Reuse Index](system/frag_reuse.md)

### Stable Row ID for Index

Using a stable row ID to replace the row address for an index is a work in progress.
The main benefit is that remap is not needed, and an update only needs to invalidate the index if related column data has changed.
The tradeoff is that it requires an additional index search to translate a stable row ID to the physical row address.
We are still working on evaluating the performance impact of this change before making it more widely used.

## Index Storage

The content of each index is stored at `_indices/{UUID}` directory under the dataset directory.
We call this location the **index directory**.
The actual content stored in the index directory depends on the index type.
