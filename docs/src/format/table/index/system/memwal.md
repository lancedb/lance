# MemWAL Index

The MemTable and Write-Ahead Log (MemWAL) Index is used for fast upserts into the Lance table.

The index is used as the centralized synchronization system for a log-structured merge tree (LSM-tree),
leaving the actual implementation of the MemTable and WAL up to the specific implementer of the spec.

Each region represents a single writer that writes to both a MemTable and a WAL,
and a region can have increasing generations of MemWALs.
Every time data is written into a WAL, the index is updated with the latest watermark.
If a specific writer of a region dies, a new writer is able to read the information in the specific region and replay the WAL.

## Index Details

```protobuf
%%% proto.message.MemWalIndexDetails %%%
```

## Expected Use Pattern

It is expected that:

1. there is exactly one writer for each region, guaranteed by optimistic update of the owner_id
2. each writer updates the MemWAL index after a successful write to WAL and MemTable
3. a new writer always finds unsealed MemWALs and performs replay before accepting new writes
4. background processes are responsible for merging flushed MemWALs to the main Lance table, and making index up to date.
5. a MemWAL-aware reader is able to merge results of MemTables in the MemWALs with results in the base Lance table. 