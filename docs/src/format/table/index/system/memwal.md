# MemWAL Index

The Memory Write-Ahead Log (MemWAL) Index is used for transaction management and crash recovery.

## Index Details

```protobuf
%%% proto.message.MemWalIndexDetails %%%
```

## Purpose

The MemWAL Index provides:
- Transaction durability before commit
- Crash recovery capabilities
- Write buffering for performance

## Storage

The MemWAL is primarily an in-memory structure with periodic persistence to ensure durability. It maintains a log of pending operations that haven't been committed to the main dataset.

## Implementation Details

- **Write Buffering**: Accumulates writes in memory before flushing
- **Transaction Log**: Maintains ordered list of operations
- **Recovery**: Replays uncommitted transactions after crash
- **Checkpoint**: Periodic persistence of WAL state