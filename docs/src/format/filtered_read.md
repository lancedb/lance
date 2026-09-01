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
and bounds the aggregate bytes reserved by decoded descriptors and materialized
payload batches that are awaiting ordered emission in one physical scanner
execution.

Admission tokens are assigned in output order and cannot be bypassed by a later
batch that becomes ready first. A single complete output batch whose charge
exceeds the bound is admitted only when no other materialization is reserved.
Reservations are retained through every ordering buffer and released when the
batch is emitted, fails, is cancelled, or is dropped. These rules prevent an
out-of-order batch from holding memory needed by an earlier batch and guarantee
forward progress.

The charge includes descriptor-array memory, output offsets, and every payload
byte. Before admission, an external descriptor whose stored size is zero must
resolve the complete object length from object metadata and charge that resolved
length. Byte arithmetic saturates at `uint64` maximum, so overflow is treated as
an oversized batch rather than wrapping or under-accounting.

When `materialization_readahead_bytes` is absent, Blob v2 materialization has no
independent memory bound. The field has no effect when the projection does not
materialize Blob v2 values as binary.
