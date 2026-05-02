# lance-topic

`lance-topic` is an experimental Kafka-like topic API backed by a real Lance
table and MemWAL shard WAL files. It is currently a storage-only design: there
is no topic server in this crate.

Topic creation owns the table schema:

- `id: Utf8` as the Lance unenforced primary key and logical partitioning key
- `producer_id: UInt32` as the producer shard slot
- `payload: lance.json` for JSON payloads

The logical topic partition is computed from `id` only. The physical WAL shard
is `(logical partition, producer_id)`. Opening a producer creates and claims
one partition writer for every logical partition, so a fleet can add producer
ids without updating topic metadata. A consumer assigned to a logical partition
refreshes shard discovery and reads every discovered producer shard for that
partition.

## Create or Open a Topic

Topic creation hides the Lance schema, primary-key metadata, table creation,
automatic-cleanup disabling, and MemWAL setup.

```rust
use lance_topic::Topic;

# async fn example() -> lance_core::Result<()> {
let topic = Topic::builder()
    .directory("/tmp/lance-topics", ["events"])
    .partition_count(4)
    .create()
    .await?;

let reopened = Topic::builder()
    .directory("/tmp/lance-topics", ["events"])
    .open()
    .await?;
# Ok(())
# }
```

`directory(root, table_id)` is the default path and builds a directory
namespace client internally. For catalog-backed tables, use
`namespace(namespace_client, table_id)`.

## Produce Messages

Each running producer uses a stable `producer_id`. Producers send JSON payloads
keyed by `id`; `id` is always the topic partition key.

```rust
use lance_topic::Topic;
use serde_json::json;

# async fn example(topic: Topic) -> lance_core::Result<()> {
let producer = topic.producer(7).await?;

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

`Topic::producer(producer_id)` claims a fresh writer epoch for every logical
partition. If another producer with the same `producer_id` starts later, the
older producer detects fencing when a later WAL create conflict or put failure
causes it to recheck the shard manifest. A successful WAL create is accepted as
the winning append. Once any partition writer is fenced, the whole producer
instance is terminally fenced and later sends fail without trying to reclaim.

## Consume Messages

A consumer group is a named logical subscription. Consumers in the same group
share progress through a nested Lance consumer-group table. A consumer has a
stable string `consumer_id`. Active consumers are discovered from the group
table and topic partitions are assigned with rendezvous hashing over
`(group_id, logical partition id, consumer_id)`. The assignment ignores
`producer_id`, and each assigned logical partition includes every producer
shard.
`Topic::consumer_group(group_id).create()` creates the nested consumer-group
table and then opens it. Use `.open()` for an existing group.
`TopicConsumerGroup::consumer_config(consumer_id)` builds a consumer config for
that group.

```rust
use lance_topic::{ConsumerConfig, Topic};

# async fn example(topic: Topic) -> lance_core::Result<()> {
let group = topic.consumer_group("billing-service").create().await?;
let mut consumer = group
    .consumer(group.consumer_config("billing-worker-1").build())
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

For multiple consumers in one group, each running process uses a distinct
string consumer id:

```rust
use lance_topic::{ConsumerConfig, Topic};

# async fn example(topic: Topic) -> lance_core::Result<()> {
let group = topic.consumer_group("billing-service").open().await?;
let mut consumer = group
    .consumer(
        group
            .consumer_config("billing-worker-2")
            .build(),
    )
    .await?;

let assigned = consumer.assigned_partitions();
let batches = consumer.poll().await?;
consumer.commit(&batches).await?;
# Ok(())
# }
```

Manual logical-partition reads are available through `PartitionReader` for
diagnostics and controlled replay. This reader is not a consumer group member
and does not commit consumer group offsets.

```rust
use lance_topic::{Topic, StartPosition};

# async fn example(topic: Topic) -> lance_core::Result<()> {
let mut reader = topic.partition_reader(0, StartPosition::Earliest).await?;
let batches = reader.poll().await?;
# Ok(())
# }
```

## Delivery Semantics

- Delivery is at least once.
- Producer retries may duplicate messages if the WAL entry was committed but
  the caller did not receive the acknowledgement.
- `send_batch` is best-effort across touched logical partitions and is not
  transactional. Some partitions may have appended before another partition
  returns an error.
- If any producer partition writer is fenced, the producer instance is
  terminally fenced.
- Offsets are entry-level WAL positions, not row-level offsets.
- Consumer group progress is tracked per `(logical partition, producer_id)`.
- Offset commits are monotonic when replayed. If multiple commits exist for the
  same `(logical partition, producer_id)`, the greatest next-entry position is
  used.
- If any assigned physical producer shard fails during `poll`, the whole poll
  fails and the consumer keeps its previous in-memory offsets.
- If any consumer group WAL writer owned by a consumer is fenced, the consumer
  instance is terminally fenced and should be recreated.

## Storage Notes

Topic data is rooted at a real Lance table. The table's `__lance_mem_wal` index
stores the shard spec: hash/bucket over `id` with `num_buckets = N`, and
identity over `producer_id`. Physical producer shards are not predeclared.
Topic open and consumer poll discover physical shards by listing `_mem_wal/`
shard directories and reading each shard's latest manifest instead of relying
on the MemWAL index snapshot as the discovery path.
Topic-created shard UUIDs are deterministic for `(logical partition,
producer_id)`, so producers with the same `producer_id` contend on the same
partition writer manifests and are fenced by epoch.

Consumer groups are nested Lance tables at
`<topic table id>/consumer_group/<group_id>`. Their WAL shard spec is
`identity(consumer_id), identity(topic_partition_id)`. Opening a consumer
claims a writer epoch independently for every logical topic partition under
that `consumer_id`, writes heartbeat events for membership, and writes offset
commit events per `(logical partition, producer_id)`.

Consumer group offsets are stored in a separate Lance table whose namespace id
is the topic table id with `consumer_group/<group>` appended. For a topic table
id `["ns1", "topic1"]`, group `billing-service` stores offsets in
`["ns1", "topic1", "consumer_group", "billing-service"]`.

The consumer group table schema is:

- `consumer_id: Utf8`
- `event_type: Utf8`
- `topic_partition_id: UInt32`
- `producer_id: UInt32?`
- `next_entry_position: UInt64?`
- `lease_expires_at_ms: UInt64?`

Its MemWAL shard spec is `identity(consumer_id), identity(topic_partition_id)`.
Each consumer claims writer epochs for its consumer/partition shards and
appends heartbeat and offset-commit events as WAL entries. Reads replay every
consumer-group shard and take the maximum committed `next_entry_position` for
each `(topic_partition_id, producer_id)`, preserving monotonic offset behavior
when assignments change.

Topic table creation disables automatic cleanup so WAL maintenance can be
handled explicitly by future topic maintenance processes.

The crate enables the AWS object-store provider by default. Disable default
features for local-only builds, or enable `azure` / `gcp` for those object-store
providers.
