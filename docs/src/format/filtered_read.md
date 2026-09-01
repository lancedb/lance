# Filtered Read Plans

Filtered read plans are serialized query-execution contracts. They are not
stored in Lance datasets. `FilteredReadExecProto` identifies the table, carries
the read options, and may carry a planner-produced selection of rows and
filters for a remote executor.

## Read options

```protobuf
%%% proto.message.FilteredReadOptionsProto %%%
```

`materialization_readahead_bytes` controls Blob v2 payload materialization when
the projection requests binary Blob values. When present, it must be nonzero
and bounds the aggregate estimated bytes reserved by decoded descriptors and
materialized payload batches that are awaiting ordered emission in one physical
scanner execution. A single output batch whose estimate exceeds the bound is
admitted only when no other materialization is reserved, which guarantees
forward progress.

When `materialization_readahead_bytes` is absent, Blob v2 materialization has no
independent memory bound. The field has no effect when the projection does not
materialize Blob v2 values as binary.
