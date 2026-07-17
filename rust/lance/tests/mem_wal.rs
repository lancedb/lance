// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::record_batch;
use lance::dataset::mem_wal::{ShardWriter, ShardWriterConfig};
use lance_core::FenceReason;
use lance_io::object_store::ObjectStore;
use uuid::Uuid;

#[tokio::test]
async fn durable_put_does_not_alias_across_memtable_generations() {
    let temp_dir = tempfile::tempdir().unwrap();
    let base_uri = format!("file://{}", temp_dir.path().display());
    let (store, base_path) = ObjectStore::from_uri(&base_uri).await.unwrap();
    let shard_id = Uuid::new_v4();
    let first_batch = record_batch!(("id", Int32, [1])).unwrap();
    let schema = first_batch.schema();
    let config = ShardWriterConfig {
        shard_id,
        durable_write: true,
        max_wal_buffer_size: 64 * 1024 * 1024,
        max_wal_flush_interval: None,
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    };

    let writer_a = ShardWriter::open(
        store.clone(),
        base_path.clone(),
        base_uri.clone(),
        config.clone(),
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();

    writer_a.put(vec![first_batch]).await.unwrap();
    let first_generation = writer_a.memtable_stats().await.unwrap().generation;
    writer_a.force_seal_active().await.unwrap();
    writer_a.wait_for_flush_drain().await.unwrap();
    assert_eq!(
        writer_a.memtable_stats().await.unwrap().generation,
        first_generation + 1
    );

    let writer_b = ShardWriter::open(store, base_path, base_uri, config, schema, vec![])
        .await
        .unwrap();
    assert!(writer_b.epoch() > writer_a.epoch());

    let second_batch = record_batch!(("id", Int32, [2])).unwrap();
    let error = writer_a
        .put(vec![second_batch])
        .await
        .expect_err("generation-2 durable put must wait for its own WAL flush");
    assert_eq!(error.fence_reason(), Some(FenceReason::PeerClaimedEpoch));
    assert!(error.to_string().contains("Writer fenced"));

    writer_b.close().await.unwrap();
}
