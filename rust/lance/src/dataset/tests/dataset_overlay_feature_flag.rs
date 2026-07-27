// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end tests for the two facts a dataset records about data overlay files.
//!
//! [`FLAG_DATA_OVERLAY_FILES`] says overlays are *present*: the manifest advertises it
//! exactly while some fragment carries an overlay. The commit that attaches the first
//! overlay sets it, and the commit that removes the last one -- by deleting the overlaid
//! fragments or by compacting the overlays into base data -- clears it again, so the
//! dataset becomes readable by pre-overlay readers once more.
//!
//! `lance.overlays.enabled` says overlays are *permitted*, and is what a writer consults
//! before choosing to store an update as an overlay. It is a two-way door: it can be
//! turned off again, but only once no overlay remains, since a disabled dataset must be
//! resolvable without overlay support.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatchIterator, record_batch};
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_io::utils::CachedFileSize;
use lance_table::feature_flags::FLAG_DATA_OVERLAY_FILES;
use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
use lance_table::format::{DataFile, LANCE_OVERLAYS_ENABLED};
use roaring::RoaringBitmap;
use tempfile::{TempDir, tempdir};
use uuid::Uuid;

use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::transaction::{DataOverlayGroup, Operation, UpdateMap, UpdateMapEntry};
use crate::dataset::{DATA_DIR, WriteDestination, WriteParams};
use crate::{Dataset, Result};

/// Two-fragment Int32 dataset: `id` (field 0) = 0..12 and `val` (field 1) = id * 10,
/// six rows per fragment (fragments 0 and 1), with overlay writes enabled. Backed by a
/// temp dir rather than `memory://` so tests can reopen the dataset and check the
/// persisted manifest.
async fn create_base_dataset() -> (TempDir, Dataset) {
    create_base_dataset_with(Some(true)).await
}

async fn create_base_dataset_with(enable_overlays: Option<bool>) -> (TempDir, Dataset) {
    let test_dir = tempdir().unwrap();
    let uri = test_dir.path().to_str().unwrap().to_string();
    let batch = record_batch!(
        ("id", Int32, (0..12).collect::<Vec<_>>()),
        ("val", Int32, (0..12).map(|v| v * 10).collect::<Vec<_>>())
    )
    .unwrap();
    let schema = batch.schema();
    let write_params = WriteParams {
        max_rows_per_file: 6,
        max_rows_per_group: 6,
        enable_overlays,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let dataset = Dataset::write(reader, &uri, Some(write_params))
        .await
        .unwrap();
    (test_dir, dataset)
}

/// Whether the manifest advertises data overlay files. Readers and writers are gated
/// together -- a writer that cannot resolve overlays cannot safely rewrite the data
/// either -- so disagreement between the two bits is a bug.
fn advertises_overlays(dataset: &Dataset) -> bool {
    let reader = dataset.manifest.reader_feature_flags & FLAG_DATA_OVERLAY_FILES != 0;
    let writer = dataset.manifest.writer_feature_flags & FLAG_DATA_OVERLAY_FILES != 0;
    assert_eq!(
        reader, writer,
        "overlay reader/writer feature flags disagree: {reader} vs {writer}"
    );
    reader
}

/// Reopen the dataset from storage so assertions see the persisted manifest rather
/// than the in-memory one produced by the commit.
async fn reopen(dataset: &Dataset) -> Dataset {
    Dataset::open(dataset.uri()).await.unwrap()
}

/// Commit a dense overlay setting `val` at `offset` of `fragment_id` to `value`.
async fn commit_overlay(dataset: Dataset, fragment_id: u64, offset: u32, value: i32) -> Dataset {
    try_commit_overlay(dataset, fragment_id, offset, value, None)
        .await
        .unwrap()
}

/// Like [`commit_overlay`] but surfaces the commit error, and lets a test commit
/// against a stale `read_version` to exercise conflict rebase.
async fn try_commit_overlay(
    dataset: Dataset,
    fragment_id: u64,
    offset: u32,
    value: i32,
    read_version: Option<u64>,
) -> Result<Dataset> {
    let read_version = read_version.unwrap_or_else(|| dataset.version().version);
    let overlay_schema = dataset.schema().project_by_ids(&[1], true);
    let filename = format!("{}.lance", Uuid::new_v4());
    let path = dataset.base.clone().join(DATA_DIR).join(filename.as_str());
    let obj_writer = dataset.object_store.create(&path).await.unwrap();
    let mut writer =
        FileWriter::try_new(obj_writer, overlay_schema, FileWriterOptions::default()).unwrap();
    let (major, minor) = writer.version().to_numbers();
    let column: ArrayRef = Arc::new(Int32Array::from(vec![value]));
    writer.write_column(0, column).await.unwrap();
    let summary = writer.finish().await.unwrap();

    let mut data_file = DataFile::new_unstarted(filename, major, minor);
    data_file.fields = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(f, _)| *f as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.column_indices = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(_, c)| *c as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);

    Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataOverlay {
            groups: vec![DataOverlayGroup {
                fragment_id,
                overlays: vec![DataOverlayFile {
                    data_file,
                    coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([offset])),
                    committed_version: 0,
                }],
            }],
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
}

/// Commit a `lance.overlays.enabled` change against an explicit read version, so a test
/// can issue a disable that was planned before a concurrent overlay landed.
async fn commit_overlays_enabled_at(
    dataset: Dataset,
    enabled: bool,
    read_version: u64,
) -> Result<Dataset> {
    Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::UpdateConfig {
            config_updates: Some(UpdateMap {
                update_entries: vec![UpdateMapEntry {
                    key: LANCE_OVERLAYS_ENABLED.to_string(),
                    value: Some(enabled.to_string()),
                }],
                replace: false,
            }),
            table_metadata_updates: None,
            schema_metadata_updates: None,
            field_metadata_updates: HashMap::new(),
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
}

/// Scan `id` and `val` and return an `id -> val` map (order-independent).
async fn id_val_map(dataset: &Dataset) -> BTreeMap<i32, i32> {
    let batch = dataset
        .scan()
        .project(&["id", "val"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = batch["id"].as_any().downcast_ref::<Int32Array>().unwrap();
    let vals = batch["val"].as_any().downcast_ref::<Int32Array>().unwrap();
    (0..batch.num_rows())
        .map(|i| (ids.value(i), vals.value(i)))
        .collect()
}

#[tokio::test]
async fn test_overlay_commit_sets_feature_flag() {
    let (_test_dir, dataset) = create_base_dataset().await;
    assert!(!advertises_overlays(&dataset));

    let dataset = commit_overlay(dataset, 0, 0, 1000).await;
    assert!(advertises_overlays(&dataset));
    // The flag is durable, and a fresh open still accepts the dataset.
    assert!(advertises_overlays(&reopen(&dataset).await));
}

#[tokio::test]
async fn test_deleting_overlaid_fragment_clears_feature_flag() {
    let (_test_dir, dataset) = create_base_dataset().await;
    let mut dataset = commit_overlay(dataset, 0, 0, 1000).await;
    assert!(advertises_overlays(&dataset));

    // Deleting some of the fragment's rows leaves the overlay in place.
    dataset.delete("id = 1").await.unwrap();
    assert!(advertises_overlays(&dataset));

    // Deleting the rest drops fragment 0, and with it the last overlay.
    dataset.delete("id < 6").await.unwrap();
    assert!(dataset.get_fragment(0).is_none());
    assert!(!advertises_overlays(&dataset));
    assert!(!advertises_overlays(&reopen(&dataset).await));
}

#[tokio::test]
async fn test_flag_remains_while_another_fragment_is_overlaid() {
    let (_test_dir, dataset) = create_base_dataset().await;
    let dataset = commit_overlay(dataset, 0, 0, 1000).await;
    let mut dataset = commit_overlay(dataset, 1, 0, 2000).await;

    // Fragment 0 and its overlay go away, but fragment 1 is still overlaid.
    dataset.delete("id < 6").await.unwrap();
    assert!(dataset.get_fragment(0).is_none());
    assert!(advertises_overlays(&dataset));
}

#[tokio::test]
async fn test_compacting_overlays_clears_feature_flag() {
    let (_test_dir, dataset) = create_base_dataset().await;
    let dataset = commit_overlay(dataset, 0, 0, 1000).await;
    let mut dataset = commit_overlay(dataset, 0, 1, 1001).await;
    assert!(advertises_overlays(&dataset));

    // `target_rows_per_fragment` matches the fragment size so the overlay count is
    // the only compaction trigger; fragment 0 (2 overlays > 1) is rewritten.
    let options = CompactionOptions {
        max_overlays_per_fragment: Some(1),
        target_rows_per_fragment: 6,
        ..Default::default()
    };
    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert_eq!(metrics.fragments_removed, 1);

    assert!(
        dataset
            .get_fragments()
            .iter()
            .all(|f| f.metadata().overlays.is_empty())
    );
    assert!(!advertises_overlays(&dataset));
    assert!(!advertises_overlays(&reopen(&dataset).await));

    // The overlaid values survived as base data, so clearing the flag did not
    // silently discard them.
    let values = id_val_map(&dataset).await;
    assert_eq!(values[&0], 1000);
    assert_eq!(values[&1], 1001);
    assert_eq!(values[&2], 20);
}

#[tokio::test]
async fn test_overlays_disabled_unless_enabled() {
    // A dataset that says nothing follows the library default, which is currently
    // off, so an overlay commit is refused rather than silently allowed.
    let (_test_dir, dataset) = create_base_dataset_with(None).await;
    assert!(!dataset.overlays_enabled());
    let error = try_commit_overlay(dataset, 0, 0, 1000, None)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("while they are disabled"),
        "unexpected error: {error}"
    );

    // The same dataset created with the setting on accepts the commit.
    let (_test_dir, dataset) = create_base_dataset_with(Some(true)).await;
    assert!(dataset.overlays_enabled());
    let dataset = commit_overlay(dataset, 0, 0, 1000).await;
    assert!(advertises_overlays(&dataset));
}

#[tokio::test]
async fn test_enablement_survives_the_flag_clearing() {
    // The two facts are independent: compacting the overlays away clears the
    // feature flag but must leave the dataset free to write overlays again.
    let (_test_dir, dataset) = create_base_dataset().await;
    let mut dataset = commit_overlay(dataset, 0, 0, 1000).await;
    dataset.remove_overlays().await.unwrap();

    assert!(!advertises_overlays(&dataset));
    assert!(dataset.overlays_enabled());
    assert!(reopen(&dataset).await.overlays_enabled());
    commit_overlay(dataset, 1, 0, 2000).await;
}

#[tokio::test]
async fn test_cannot_disable_while_overlays_remain() {
    let (_test_dir, dataset) = create_base_dataset().await;
    let mut dataset = commit_overlay(dataset, 0, 0, 1000).await;

    let error = dataset.set_overlays_enabled(false).await.unwrap_err();
    let message = error.to_string();
    assert!(message.contains("still carry overlays"), "{message}");
    // The offending fragment is named so the caller knows what to compact.
    assert!(message.contains('0'), "{message}");
    assert!(dataset.overlays_enabled());
}

#[tokio::test]
async fn test_remove_overlays_then_disable() {
    let (_test_dir, dataset) = create_base_dataset().await;
    let dataset = commit_overlay(dataset, 0, 0, 1000).await;
    let mut dataset = commit_overlay(dataset, 0, 1, 1001).await;

    let metrics = dataset.remove_overlays().await.unwrap();
    // Only the overlaid fragment is rewritten. Fragment 1 is well under the
    // default row target, so a plain compaction would have swept it up too.
    assert_eq!(metrics.fragments_removed, 1);
    assert_eq!(metrics.fragments_added, 1);
    assert!(dataset.get_fragment(1).is_some());
    assert!(!advertises_overlays(&dataset));

    dataset.set_overlays_enabled(false).await.unwrap();
    assert!(!dataset.overlays_enabled());
    assert!(!reopen(&dataset).await.overlays_enabled());

    // Disabling did not cost any data.
    let values = id_val_map(&dataset).await;
    assert_eq!(values[&0], 1000);
    assert_eq!(values[&1], 1001);
    assert_eq!(values[&2], 20);

    // And it is a real gate, not just a label. Fragment 1 was never overlaid, so
    // compaction left it in place.
    let error = try_commit_overlay(dataset, 1, 0, 2000, None)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("while they are disabled"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn test_disable_racing_a_concurrent_overlay_is_rejected() {
    // The disable is planned against a version with no overlays, so it is valid
    // when issued. A concurrent writer then adds one. The check lives in
    // build_manifest, which re-runs when the transaction is rebased onto the
    // newer version, so the race is caught rather than silently committed.
    let (_test_dir, dataset) = create_base_dataset().await;
    let stale_version = dataset.version().version;
    let dataset = commit_overlay(dataset, 0, 0, 1000).await;
    assert!(dataset.version().version > stale_version);
    let uri = dataset.uri().to_string();

    let error = commit_overlays_enabled_at(dataset, false, stale_version)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("still carry overlays"),
        "unexpected error: {error}"
    );

    // The dataset is untouched: still enabled, still carrying its overlay.
    let dataset = Dataset::open(&uri).await.unwrap();
    assert!(dataset.overlays_enabled());
    assert!(advertises_overlays(&dataset));
}

#[tokio::test]
async fn test_unparsable_setting_is_rejected_at_write() {
    let (_test_dir, mut dataset) = create_base_dataset().await;
    let error = dataset
        .update_config([(LANCE_OVERLAYS_ENABLED.to_string(), "yes".to_string())])
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("must be \"true\" or \"false\""),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn test_dataset_with_overlays_but_no_setting_stays_writable() {
    // Datasets written while overlays were experimental carry overlays without
    // the setting. Removing the key reproduces that shape: the dataset must stay
    // overlay-writable rather than becoming un-updatable.
    let (_test_dir, dataset) = create_base_dataset().await;
    let mut dataset = commit_overlay(dataset, 0, 0, 1000).await;
    dataset
        .update_config([(LANCE_OVERLAYS_ENABLED.to_string(), None)])
        .await
        .unwrap();
    assert!(!dataset.config().contains_key(LANCE_OVERLAYS_ENABLED));

    assert!(dataset.overlays_enabled());
    let dataset = commit_overlay(dataset, 0, 1, 1001).await;
    assert_eq!(
        dataset.get_fragment(0).unwrap().metadata().overlays.len(),
        2
    );
}
