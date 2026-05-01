# lance-queue

`lance-queue` is an experimental queue API backed by a Lance table and its
MemWAL storage layout. The queue table owns a fixed schema:

- `id: Utf8` as the Lance unenforced primary key and partitioning key
- `payload: lance.json` for JSON payloads

Messages are appended to MemWAL-compatible WAL files under the backing Lance
table. Producers use stable primary-key hash partitioning. Consumers read WAL
entries and commit group offsets back to Lance table metadata.

## Create or Open a Queue

Queue creation hides the Lance schema, primary-key metadata, table creation,
and MemWAL setup.

```rust
use lance_queue::Queue;

# async fn example() -> lance_core::Result<()> {
let queue = Queue::builder()
    .uri("file:///tmp/events.lance")
    .partition_count(8)
    .create()
    .await?;

let reopened = Queue::builder()
    .uri("file:///tmp/events.lance")
    .open()
    .await?;
# Ok(())
# }
```

For catalog-backed tables, use `namespace(namespace_client, table_id)` instead
of `uri(...)`.

## Produce Messages

Producers send JSON payloads keyed by `id`. The `id` is always the partitioning
key.

```rust
use lance_queue::Queue;
use serde_json::json;

# async fn example(queue: Queue) -> lance_core::Result<()> {
let producer = queue.producer();

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
# Ok(())
# }
```

## Consume Messages

A consumer group is a named logical subscription. Consumers in the same group
share progress through table metadata keys. By default, a consumer reads all
queue partitions in its group.

```rust
use lance_queue::{ConsumerConfig, Queue};

# async fn example(queue: Queue) -> lance_core::Result<()> {
let mut consumer = queue
    .consumer(ConsumerConfig::new("billing-service"))
    .await?;

let batches = consumer.poll().await?;
for batch in &batches {
    for message in batch.messages()? {
        println!("{}: {}", message.id, message.payload);
    }
}

consumer.commit(&batches).await?;
# Ok(())
# }
```

To split work across multiple consumers in one group, configure the total
consumer partition count and this consumer's partition id. Queue partitions are
assigned to consumer partitions with stable rendezvous hashing.

```rust
use lance_queue::{ConsumerConfig, Queue};

# async fn example(queue: Queue) -> lance_core::Result<()> {
let mut consumer = queue
    .consumer(
        ConsumerConfig::new("billing-service")
            .with_consumer_partition(3, 0),
    )
    .await?;

let assigned = consumer.assigned_partitions();
let batches = consumer.poll().await?;
consumer.commit(&batches).await?;
# Ok(())
# }
```

Manual queue-partition assignment is available for diagnostics and controlled
replay:

```rust
use lance_queue::{ConsumerConfig, Queue};

# async fn example(queue: Queue) -> lance_core::Result<()> {
let mut consumer = queue
    .consumer(ConsumerConfig::new("billing-service").with_partitions([0, 2]))
    .await?;
# Ok(())
# }
```

## Delivery Semantics

- Delivery is at least once.
- Offsets are entry-level WAL positions, not row-level offsets.
- Offset commits are monotonic. A stale commit lower than the already committed
  position is ignored and logged.
- If any assigned partition fails during `poll`, the whole poll fails and the
  consumer keeps its previous in-memory offsets.

## Storage Notes

Queue data is rooted at a real Lance table. The table's `__lance_mem_wal` index
stores the partition/shard spec and inline shard snapshots. Consumer group
offsets are stored in table metadata under keys like:

```text
lance_queue.group.<group>.commits.<partition>.next_entry_position
```

Queue table creation disables automatic cleanup so WAL maintenance can be
handled explicitly by future queue maintenance processes.
