# lance-queue

`lance-queue` is an experimental Kafka-like queue API backed by a real Lance
table and MemWAL shard WAL files. It is currently a storage-only design: there
is no broker service in this crate.

Queue creation owns the table schema:

- `id: Utf8` as the Lance unenforced primary key and logical partitioning key
- `producer_id: UInt32` as the producer shard slot
- `payload: lance.json` for JSON payloads

The logical queue partition is computed from `id` only. The physical WAL shard
is `(logical partition, producer_id)`. For example, 4 logical partitions and 10
producer slots create 40 MemWAL shards. A consumer assigned to a logical
partition reads every producer shard for that partition.

## Create or Open a Queue

Queue creation hides the Lance schema, primary-key metadata, table creation,
automatic-cleanup disabling, and MemWAL setup.

```rust
use lance_queue::Queue;

# async fn example() -> lance_core::Result<()> {
let queue = Queue::builder()
    .uri("file:///tmp/events.lance")
    .partition_count(4)
    .producer_count(10)
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

Each running producer uses one configured `producer_id` slot. Producers send
JSON payloads keyed by `id`; `id` is always the queue partition key.

```rust
use lance_queue::Queue;
use serde_json::json;

# async fn example(queue: Queue) -> lance_core::Result<()> {
let producer = queue.producer(7)?;

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

`send` and `send_batch` acknowledge only after the corresponding WAL entry or
entries are written. `send_batch` does not defer durability beyond the call; its
batch size is the number of messages in one producer call and therefore the WAL
entry granularity per touched logical partition.

## Consume Messages

A consumer group is a named logical subscription. Consumers in the same group
share progress through Lance table metadata. By default, a consumer reads all
logical queue partitions in its group and expands each partition to all
producer shards.

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
assigned to consumer partitions with stable rendezvous hashing over logical
partition ids. The assignment ignores `producer_id`, and each assigned logical
partition includes every producer shard.

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

Manual logical-partition assignment is available for diagnostics and controlled
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
- Producer retries may duplicate messages if the WAL entry was committed but
  the caller did not receive the acknowledgement.
- Offsets are entry-level WAL positions, not row-level offsets.
- Consumer group progress is tracked per `(logical partition, producer_id)`.
- Offset commits are monotonic. A stale commit lower than the already committed
  position is ignored and logged.
- If any assigned physical producer shard fails during `poll`, the whole poll
  fails and the consumer keeps its previous in-memory offsets.

## Storage Notes

Queue data is rooted at a real Lance table. The table's `__lance_mem_wal` index
stores the logical-partition and producer-shard spec plus inline shard
snapshots. Consumer group offsets are stored in table metadata under keys like:

```text
lance_queue.group.<group>.commits.<partition>.<producer>.next_entry_position
```

Queue table creation disables automatic cleanup so WAL maintenance can be
handled explicitly by future queue maintenance processes.

The crate enables the AWS object-store provider by default. Disable default
features for local-only builds, or enable `azure` / `gcp` for those object-store
providers.
