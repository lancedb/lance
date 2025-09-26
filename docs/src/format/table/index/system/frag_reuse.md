# Fragment Reuse Index

The Fragment Reuse Index is an internal index used to optimize fragment operations 
during compaction and dataset updates.

When data modifications happen against a Lance table,
it could trigger compaction and index optimization at the same time to improve data layout and index coverage.
By default, compaction will remap all indices at the same time to prevent read regression.
This means both compaction and index optimization could modify the same index and cause one process to fail.
Typically, the compaction would fail because it has to modify all indices and takes longer,
resulting in table layout degrading over time.

Fragment Reuse Index allows a compaction to defer the index remap process.
Suppose a compaction removes fragments A and B and produces C.
At query runtime, it reuses the old fragments A and B by 
updating the row addresses related to A and B in the index to the latest ones in C.
Because indices are typically cached in memory after initial load,
the in-memory index is up to date after the fragment reuse application process.

## Index Details

```protobuf
%%% proto.message.FragmentReuseIndexDetails %%%
```

## Expected Use Pattern

Fragment Reuse Index should be created if the user defers index remap in compaction.
The index accumulates a new **reuse version** every time a compaction is executed.

As long as all the scalar and vector indices are created after the specific reuse version,
the indices are all caught up and the specific reuse version can be trimmed.

It is expected that the user schedules an additional process to trim the index periodically
to keep the list of reuse versions in control.