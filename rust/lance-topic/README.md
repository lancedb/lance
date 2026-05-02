# lance-topic

`lance-topic` is an experimental Kafka-like topic API backed by Lance tables
and MemWAL shard WAL files. It is a storage-only design with no topic server.

## Architecture

All components are backed by Lance tables with MemWAL shards on shared
storage. There are no servers — producers, consumers, and consumer groups
coordinate through storage-level fencing and atomic writes.

```
                       ┌──────────────────────────────────────────────────┐
   Producers           │ Topic (Lance Table + MemWAL)                     │
                       │                                                  │
 ┌──────────┐  send    │ Partition 0     Partition 1     ...  Partition N │
 │ server-1 │ ──────── │ ┌───────────┐  ┌───────────┐       ┌───────────┐ │
 └──────────┘          │ │ WAL shard │  │ WAL shard │       │ WAL shard │ │
                       │ │(server-1) │  │(server-1) │       │(server-1) │ │
 ┌──────────┐  send    │ ├───────────┤  ├───────────┤       ├───────────┤ │
 │ server-2 │ ──────── │ │ WAL shard │  │ WAL shard │       │ WAL shard │ │
 └──────────┘          │ │(server-2) │  │(server-2) │       │(server-2) │ │
                       │ └───────────┘  └───────────┘       └───────────┘ │
                       └───────┬──────────────┬───────────────────────────┘
                               │              │
                   poll        │              │         poll
              ┌────────────────┘              └───────────────────┐
              │                                                   │
 ┌────────────┴────────────────────────────────────────────────┐  │
 │ Consumer Group "billing" (Lance Table + MemWAL)             │  │
 │                                                             │  │
 │ Consumer 0 → assigned partitions [0, 2]                     │  │
 │ Consumer 1 → assigned partitions [1, 3]                     │  │
 │                                                             │  │
 │ Table schema:                                               │  │
 │   consumer_position | topic_partition_id | producer_id | ...│  │
 │                                                             │  │
 │ _mem_wal/ shards (one per consumer position + partition):   │  │
 │ ┌───────────────┐ ┌───────────────┐                         │  │
 │ │ WAL shard     │ │ WAL shard     │                         │  │
 │ │ (pos=0, p0)   │ │ (pos=0, p2)   │ ← Consumer 0 writes     │  │
 │ └───────────────┘ └───────────────┘   committed offsets     │  │
 │ ┌───────────────┐ ┌───────────────┐                         │  │
 │ │ WAL shard     │ │ WAL shard     │                         │  │
 │ │ (pos=1, p1)   │ │ (pos=1, p3)   │ ← Consumer 1 writes     │  │
 │ └───────────────┘ └───────────────┘   committed offsets     │  │
 │                                                             │  │
 │ Each offset entry records:                                  │  │
 │   (topic_partition_id, producer_id) → next_entry_position   │  │
 └─────────────────────────────────────────────────────────────┘  │
                                                                  │
 ┌────────────────────────────────────────────────────────────────┴┐
 │ Consumer Group "analytics" (Lance Table + MemWAL)               │
 │                                                                 │
 │ Consumer 0 → assigned partitions [0, 1, 2, 3]                   │
 │ _mem_wal/ shards: (pos=0, p0), (pos=0, p1), ...                 │
 └─────────────────────────────────────────────────────────────────┘
```

- A **Topic** is one Lance table. Its `_mem_wal/` shards store message data,
  keyed by `(logical_partition, producer_id)`.
- **Producers** write to the topic's WAL shards. Each producer owns one shard
  per partition. Multiple producers share the same topic without coordination.
- A **Consumer Group** is a separate Lance table with its own `_mem_wal/`
  shards that store committed offsets. Each group independently tracks
  progress against the topic.
- **Consumers** within a group are assigned partitions by position. Each
  consumer polls all producer shards for its assigned partitions.

### Topic

A topic is a Lance table with a fixed schema:

- `id: Utf8` — unenforced primary key and logical partition key
- `producer_id: Utf8` — producer shard identifier
- `payload: lance.json` — JSON message payload

### Partitions and Shards

The topic has N logical partitions configured at creation time. The logical
partition for a message is determined by hashing `id` (Murmur3 bucket). Each
`(logical_partition, producer_id)` pair maps to one physical MemWAL shard with
a deterministic UUID. This means:

- Multiple producers can write to the same topic without coordination
- Each producer owns its own set of physical shards (one per logical partition)
- A logical partition may contain entries from many producers

```
Topic (4 partitions, 2 producers)
├── Logical Partition 0
│   ├── Physical Shard (partition=0, producer="server-1")  →  _mem_wal/{uuid}/wal/
│   └── Physical Shard (partition=0, producer="server-2")  →  _mem_wal/{uuid}/wal/
├── Logical Partition 1
│   ├── Physical Shard (partition=1, producer="server-1")
│   └── Physical Shard (partition=1, producer="server-2")
├── Logical Partition 2
│   └── ...
└── Logical Partition 3
    └── ...
```

Physical shards are not predeclared. Opening a producer creates and claims
shards on demand. Shard discovery happens by listing `_mem_wal/` directories
and reading each shard's latest manifest.

### WAL Appender

Note: WAL appender is a concept defined in the Lance MemWAL specification. See
the MemWAL spec for details on shard manifests, epoch fencing, and the
WAL file layout.

Each physical shard is written by a `WalAppender`. A WAL appender owns a
single shard and writes Arrow IPC entries as sequentially numbered files under
`_mem_wal/{shard_uuid}/wal/`. Each shard has a manifest that tracks the
current `writer_epoch` — a monotonically increasing fencing token.

When a producer opens, it creates one WAL appender per logical partition.
Each appender claims a fresh writer epoch by atomically updating its shard
manifest. A successful `send` writes a WAL entry file using atomic
put-if-not-exists. The appender does not read the manifest on the hot write
path — a successful WAL file create is accepted as the winning append.

If a WAL file create fails (conflict with another writer), the appender
rechecks the shard manifest. If a newer epoch exists, the appender is fenced.
Once any appender is fenced, the entire producer is terminally fenced — all
further sends fail immediately without attempting to reclaim. This means a
producer with the same `producer_id` can only be replaced, never shared.

Consumer group offset writers use the same WAL appender mechanism. Each
consumer claims appender epochs for its assigned consumer group shards,
providing the same hard fencing guarantee: if a new consumer with the same
position opens, the old consumer's offset writers are fenced.

### Delivery Semantics

- At-least-once delivery. Producer retries may duplicate messages if the
  WAL entry was committed but the caller did not receive acknowledgement.
- `send_batch` is best-effort across partitions — some partitions may
  succeed before another fails. It is not transactional.
- Offsets are entry-level WAL positions, not row-level.
- If any shard fails during consumer `poll`, the whole poll fails and the
  consumer keeps its previous in-memory offsets.

### Consumer Groups

A consumer group is a separate Lance table with its own MemWAL shards. Both
the topic table and its consumer group tables are managed through a Lance
namespace client. A namespace resolves a multi-segment table ID to a physical
storage location. The table ID is a logical path — not a filesystem path.

```
Namespace (e.g. DirectoryNamespace at /data/lance)
│
├── Table ID: ["website", "events"]
│   ← Topic table (schema: id, producer_id, payload)
│   └── _mem_wal/
│       ├── {shard-uuid-1}/  (partition=0, producer="server-1")
│       ├── {shard-uuid-2}/  (partition=0, producer="server-2")
│       └── ...
│
├── Table ID: ["website", "events", "consumer_group", "billing-service"]
│   ← Consumer group table (schema: consumer_position, topic_partition_id,
│      producer_id, next_entry_position)
│   └── _mem_wal/
│       ├── {shard-uuid}/  (position=0, partition=0)
│       └── ...
│
└── Table ID: ["website", "events", "consumer_group", "analytics"]
    ← Another consumer group
    └── ...
```

The topic table ID is provided at creation (e.g. `["website", "events"]`).
Consumer group table IDs are derived by appending
`["consumer_group", "<group_id>"]` to the topic's table ID. The namespace
client resolves each table ID independently to its own Lance dataset with its
own manifest and WAL directory.

Each consumer group tracks committed offsets per `(topic_partition_id,
producer_id)`. Consumers are identified by a 0-indexed position within a
declared total count. Partitions are assigned deterministically using
rendezvous hashing over `(partition_id, position)` — each consumer can
independently compute its assignment without coordination.

```
Consumer Group "billing-service" (3 consumers, 4 topic partitions)
├── Consumer 0  →  assigned partitions [0, 2]
├── Consumer 1  →  assigned partitions [1]
└── Consumer 2  →  assigned partitions [3]
```

The consumer group table schema is:

- `consumer_position: UInt32` — consumer position (unenforced primary key)
- `topic_partition_id: UInt32` — topic partition being tracked
- `producer_id: Utf8` — producer shard being tracked
- `next_entry_position: UInt64` — next WAL entry to read

Its MemWAL shard spec is `identity(consumer_position),
identity(topic_partition_id)`. Each consumer claims writer epochs for its
assigned partition shards. Consumers use hard fencing: if a new consumer with
the same position starts, the old one is terminally fenced.

If the total consumer count exceeds the topic's partition count, excess
consumers are idle (assigned zero partitions).

### WAL Tailer

Note: WAL Tailer is a concept defined in the Lance MemWAL specification. See the
MemWAL spec for details on WAL entry format and position semantics.

A `WalTailer` reads entries from a single physical shard. It is the
underlying read primitive used by both consumers (to read topic data) and
consumer group stores (to replay committed offsets).

Each WAL entry is a self-contained Arrow IPC file. The tailer reads entries
by position — sequential integers starting from 1. When determining the
next available position, the tailer uses the shard manifest's
`wal_entry_position_last_seen` field as a starting hint and probes forward
to find the true tip, avoiding a full directory listing when the hint is
recent. If the hint is stale or unavailable, it falls back to listing.

### MemWAL Usage

The topic system works without any background maintenance. Producers append
to WAL shards and consumers read from them indefinitely.

However, the underlying Lance MemWAL supports flushing WAL entries into
Flushed MemTables and merging Flushed MemTables into the base Lance table, 
updating the MemWAL index with shard snapshots along the way. When this maintenance 
is performed, all historical topic messages and consumer group offset commits
become regular Lance table data. This means users get a full history of
all events and consumer activity as queryable Lance datasets, where they
can create vector indices, scalar indices, and full-text indices to run
search and analytics leveraging all Lance capabilities — unifying real-time
streaming and offline training and analytics into the same storage backend.

## Examples

### Create or Open a Topic

```rust
let topic = Topic::builder()
    .directory("/tmp/lance-topics", ["website", "events"])
    .partition_count(4)
    .create()
    .await?;

let reopened = Topic::builder()
    .directory("/tmp/lance-topics", ["website", "events"])
    .open()
    .await?;
```

`directory(root, table_id)` builds a directory namespace client internally.
For catalog-backed tables, use `namespace(namespace_client, table_id)`.

### Produce Messages

```rust
let producer = topic.producer("producer-server-1").await?;

producer
    .send("order-123", json!({ "status": "created", "total": 42.50 }))
    .await?;

producer
    .send_batch(
        ["order-124", "order-125"],
        [
            json!({ "status": "created", "total": 18.00 }),
            json!({ "status": "created", "total": 73.25 }),
        ],
    )
    .await?;
```

`send` and `send_batch` acknowledge after the WAL entry is written.
`send_batch` commits one WAL entry per touched logical partition.

### Consumer Groups

Create or open a consumer group, then create consumers with a position and
total count:

```rust
let group = topic.consumer_group("billing-service").create().await?;
let mut consumer = group.consumer(0, 8).await?;
```

### Consume Messages

Poll messages from a consumer and acknowledge the messages after processing:

```rust
let batches = consumer.poll().await?;
for batch in &batches {
    for message in batch.messages()? {
        println!("{}: {}", message.id, message.payload);
    }
}

consumer.commit(&batches).await?;
```
