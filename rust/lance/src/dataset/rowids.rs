// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::Dataset;
use crate::session::caches::{RowIdIndexKey, RowIdSequenceKey};
use crate::{Error, Result};
use futures::{Stream, StreamExt, TryFutureExt, TryStreamExt};
use lance_core::utils::address::{LogicalRowAddress, RowAddress};
use lance_core::utils::deletion::DeletionVector;
use lance_table::rowids::segment::U64Segment;
use lance_table::{
    format::{Fragment, RowDatasetVersionRun, RowDatasetVersionSequence, RowIdMeta},
    rowids::{FragmentRowIdIndex, RowIdIndex, RowIdSequence, read_row_ids},
};
use std::{
    collections::{HashMap, hash_map::Entry},
    sync::Arc,
};

/// Load a row id sequence from the given dataset and fragment.
pub async fn load_row_id_sequence(
    dataset: &Dataset,
    fragment: &Fragment,
) -> Result<Arc<RowIdSequence>> {
    // Virtual path to prevent collisions in the cache.
    match &fragment.row_id_meta {
        None => Err(Error::internal("Missing row id meta")),
        Some(RowIdMeta::Inline(data)) => {
            let data = data.clone();
            let key = RowIdSequenceKey {
                fragment_id: fragment.id,
            };
            dataset
                .metadata_cache
                .get_or_insert_with_key(key, || async move { read_row_ids(&data) })
                .await
        }
        Some(RowIdMeta::External(file_slice)) => {
            let file_slice = file_slice.clone();
            let dataset_clone = dataset.clone();
            let key = RowIdSequenceKey {
                fragment_id: fragment.id,
            };
            dataset
                .metadata_cache
                .get_or_insert_with_key(key, || async move {
                    let path = dataset_clone.base.clone().join(file_slice.path.as_str());
                    let range = file_slice.offset as usize
                        ..(file_slice.offset as usize + file_slice.size as usize);
                    let data = dataset_clone
                        .object_store
                        .open(&path)
                        .await?
                        .get_range(range)
                        .await?;
                    read_row_ids(&data)
                })
                .await
        }
    }
}

/// Load row id sequences from the given dataset and fragments.
///
/// Returned as a vector of (fragment_id, sequence) pairs. These are not
/// guaranteed to be in the same order as the input fragments.
pub fn load_row_id_sequences<'a>(
    dataset: &'a Dataset,
    fragments: &'a [Fragment],
) -> impl Stream<Item = Result<(u32, Arc<RowIdSequence>)>> + 'a {
    futures::stream::iter(fragments)
        .map(|fragment| {
            load_row_id_sequence(dataset, fragment).map_ok(move |seq| (fragment.id as u32, seq))
        })
        .buffer_unordered(dataset.object_store.io_parallelism())
}

fn version_sequence(values: &[u64]) -> RowDatasetVersionSequence {
    let mut runs = Vec::new();
    let mut start = 0_u64;
    while start < values.len() as u64 {
        let version = values[start as usize];
        let mut end = start + 1;
        while end < values.len() as u64 && values[end as usize] == version {
            end += 1;
        }
        runs.push(RowDatasetVersionRun {
            span: U64Segment::Range(start..end),
            version,
        });
        start = end;
    }
    RowDatasetVersionSequence { runs }
}

/// Resolve row-version values in the same order as a list of live logical row IDs.
///
/// Storage version 2.3 defines a logical domain's `creation_version` as the
/// base value whenever per-row version metadata is absent.  Existing metadata,
/// when present, overrides that base for the addressed physical row.
pub async fn resolve_logical_row_version_sequences(
    dataset: &Dataset,
    row_ids: &[u64],
) -> Result<(RowDatasetVersionSequence, RowDatasetVersionSequence)> {
    if !dataset.manifest.uses_stable_logical_row_addresses() {
        return Err(Error::invalid_input(
            "logical row-version resolution requires storage version 2.3",
        ));
    }
    let physical = dataset.resolve_logical_row_ids_async(row_ids).await?;
    let fragments = dataset
        .manifest
        .fragments
        .iter()
        .map(|fragment| (fragment.id as u32, fragment))
        .collect::<HashMap<_, _>>();
    let router = dataset.row_address_router()?;
    let mut domain_versions = HashMap::<u32, u64>::new();
    let mut metadata = HashMap::<
        u32,
        (
            Option<RowDatasetVersionSequence>,
            Option<RowDatasetVersionSequence>,
        ),
    >::new();
    let mut created = Vec::with_capacity(row_ids.len());
    let mut updated = Vec::with_capacity(row_ids.len());

    for (row_id, physical) in row_ids.iter().copied().zip(physical) {
        let logical = LogicalRowAddress::try_from(row_id)?;
        let domain_version =
            if let Some(version) = domain_versions.get(&logical.logical_fragment_id()) {
                *version
            } else {
                let version = router
                    .logical_domain(logical.logical_fragment_id())?
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "logical domain {} is missing while preserving row versions",
                            logical.logical_fragment_id()
                        ))
                    })?
                    .creation_version;
                domain_versions.insert(logical.logical_fragment_id(), version);
                version
            };
        let physical = physical.ok_or_else(|| {
            Error::invalid_input(format!(
                "row-version source contains non-live logical row ID {row_id}"
            ))
        })?;
        let fragment = fragments.get(&physical.fragment_id()).ok_or_else(|| {
            Error::internal("row-version source fragment is absent from the manifest")
        })?;
        if let Entry::Vacant(entry) = metadata.entry(physical.fragment_id()) {
            let created = fragment
                .created_at_version_meta
                .as_ref()
                .map(|value| value.load_sequence())
                .transpose()?;
            let updated = fragment
                .last_updated_at_version_meta
                .as_ref()
                .map(|value| value.load_sequence())
                .transpose()?;
            entry.insert((created, updated));
        }
        let sequences = metadata
            .get(&physical.fragment_id())
            .expect("row-version metadata cache was populated");
        let offset = physical.row_offset() as usize;
        let created_version = match &sequences.0 {
            Some(sequence) => sequence.version_at(offset).ok_or_else(|| {
                Error::internal(format!(
                    "created-at metadata for fragment {} does not cover row offset {offset}",
                    fragment.id
                ))
            })?,
            None => domain_version,
        };
        let updated_version = match &sequences.1 {
            Some(sequence) => sequence.version_at(offset).ok_or_else(|| {
                Error::internal(format!(
                    "last-updated metadata for fragment {} does not cover row offset {offset}",
                    fragment.id
                ))
            })?,
            None => created_version,
        };
        created.push(created_version);
        updated.push(updated_version);
    }

    Ok((version_sequence(&created), version_sequence(&updated)))
}

/// Update selected physical row offsets while preserving every other row's
/// last-updated value under the storage-version-2.3 domain contract.
pub async fn refresh_v2_3_fragment_last_updated_at(
    dataset: &Dataset,
    fragment: &mut Fragment,
    updated_offsets: &[usize],
    current_version: u64,
) -> Result<()> {
    if !dataset.manifest.uses_stable_logical_row_addresses() {
        return Err(Error::invalid_input(
            "storage-version-2.3 row-version refresh requires logical row addresses",
        ));
    }
    let row_count = fragment.physical_rows.unwrap_or_default();
    if row_count == 0 {
        return Ok(());
    }
    let base = if let Some(metadata) = fragment.last_updated_at_version_meta.as_ref() {
        metadata.load_sequence()?
    } else if let Some(metadata) = fragment.created_at_version_meta.as_ref() {
        metadata.load_sequence()?
    } else if let Some(native) = fragment.native_logical_domain.as_ref() {
        RowDatasetVersionSequence::from_uniform_row_count(row_count as u64, native.creation_version)
    } else {
        let fragment_id = u32::try_from(fragment.id)
            .map_err(|_| Error::invalid_input("physical fragment id exceeds u32"))?;
        let deletion_vector = dataset
            .get_fragment(fragment.id as usize)
            .ok_or_else(|| Error::internal("row-version source fragment is missing"))?
            .get_deletion_vector()
            .await?
            .unwrap_or_else(|| Arc::new(DeletionVector::NoDeletions));
        let router = dataset.row_address_router()?;
        let mut domains = HashMap::<u32, u64>::new();
        let mut versions = Vec::with_capacity(row_count);
        const RESOLVE_BATCH_SIZE: usize = 65_536;
        for start in (0..row_count).step_by(RESOLVE_BATCH_SIZE) {
            let end = (start + RESOLVE_BATCH_SIZE).min(row_count);
            let physical = (start..end)
                .map(|offset| RowAddress::new_from_parts(fragment_id, offset as u32))
                .collect::<Vec<_>>();
            let logical = dataset.resolve_physical_row_ids_async(&physical).await?;
            for (offset, logical) in (start..end).zip(logical) {
                let Some(logical) = logical else {
                    if deletion_vector.contains(offset as u32) {
                        // Deleted slots are removed before any version value is observable.
                        versions.push(dataset.manifest.version);
                        continue;
                    }
                    return Err(Error::internal(format!(
                        "live physical row {}:{} has no logical owner while preserving row versions",
                        fragment.id, offset
                    )));
                };
                let version = if let Some(version) = domains.get(&logical.logical_fragment_id()) {
                    *version
                } else {
                    let version = router
                        .logical_domain(logical.logical_fragment_id())?
                        .ok_or_else(|| {
                            Error::internal(format!(
                                "logical domain {} is missing while preserving row versions",
                                logical.logical_fragment_id()
                            ))
                        })?
                        .creation_version;
                    domains.insert(logical.logical_fragment_id(), version);
                    version
                };
                versions.push(version);
            }
        }
        version_sequence(&versions)
    };

    lance_table::rowids::version::refresh_row_latest_update_meta_for_partial_frag_rewrite_cols_with_base(
        fragment,
        updated_offsets,
        current_version,
        &base,
    )
}

pub async fn get_row_id_index(
    dataset: &Dataset,
) -> Result<Option<Arc<lance_table::rowids::RowIdIndex>>> {
    if dataset.manifest.uses_legacy_stable_row_ids() {
        let key = RowIdIndexKey {
            version: dataset.manifest.version,
        };
        let index = dataset
            .metadata_cache
            .get_or_insert_with_key(key, || load_row_id_index(dataset))
            .await?;
        Ok(Some(index))
    } else {
        Ok(None)
    }
}

async fn load_row_id_index(dataset: &Dataset) -> Result<lance_table::rowids::RowIdIndex> {
    let sequences = load_row_id_sequences(dataset, &dataset.manifest.fragments)
        .try_collect::<Vec<_>>()
        .await?;

    let fragments = dataset.get_fragments();
    let fragment_map: std::collections::HashMap<u32, &crate::dataset::fragment::FileFragment> =
        fragments.iter().map(|f| (f.id() as u32, f)).collect();

    let fragment_indices: Vec<_> =
        futures::stream::iter(sequences.into_iter().map(|(fragment_id, sequence)| {
            let fragment = fragment_map
                .get(&fragment_id)
                .expect("Fragment should exist");
            let has_deletion_file = fragment.metadata().deletion_file.is_some();
            let fragment_clone = (*fragment).clone();
            async move {
                let deletion_vector = if has_deletion_file {
                    match fragment_clone.get_deletion_vector().await {
                        Ok(Some(dv)) => dv,
                        Ok(None) | Err(_) => Arc::new(DeletionVector::default()),
                    }
                } else {
                    Arc::new(DeletionVector::default())
                };

                Ok::<FragmentRowIdIndex, Error>(FragmentRowIdIndex {
                    fragment_id,
                    row_id_sequence: sequence,
                    deletion_vector,
                })
            }
        }))
        .buffer_unordered(dataset.object_store.io_parallelism())
        .try_collect()
        .await?;

    let index = RowIdIndex::new(&fragment_indices)?;

    Ok(index)
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use crate::dataset::{UpdateBuilder, WriteMode, WriteParams, builder::DatasetBuilder};

    use super::*;

    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::index::DatasetIndexExt;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, Int32Type, UInt64Type};
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, UInt64Array};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::Future;
    use lance_core::datatypes::Schema;
    use lance_core::{ROW_ADDR, ROW_ID, utils::address::RowAddress};
    use lance_datagen::Dimension;
    use lance_index::{IndexType, scalar::ScalarIndexParams};
    use std::collections::HashMap;
    use std::collections::HashSet;

    fn sequence_batch(values: Range<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from_iter_values(values))]).unwrap()
    }

    #[tokio::test]
    async fn test_empty_dataset_rowids() {
        let schema = sequence_batch(0..0).schema();
        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        assert!(dataset.manifest.uses_stable_row_ids());

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).is_none());

        assert_eq!(dataset.manifest().next_row_id, 0);
    }

    #[tokio::test]
    async fn test_must_set_on_creation() {
        let tmp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &tmp_dir;

        let batch = sequence_batch(0..10);
        let reader =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: false,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();
        assert!(!dataset.manifest().uses_stable_row_ids());

        // Trying to append without stable row ids should pass (a warning is emitted) but should not
        // affect the stable_row_ids setting.
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            mode: WriteMode::Append,
            ..Default::default()
        };
        let reader =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), batch.schema());
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();
        assert!(!dataset.manifest().uses_stable_row_ids());
    }

    #[tokio::test]
    async fn test_new_row_ids() {
        let num_rows = 25u64;
        let batch = sequence_batch(0..num_rows as i32);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 10,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();

        let found_addresses = (0..num_rows)
            .map(|i| index.get(i).unwrap())
            .collect::<Vec<_>>();
        let expected_addresses = (0..num_rows)
            .map(|i| {
                let fragment_id = i / 10;
                RowAddress::new_from_parts(fragment_id as u32, (i % 10) as u32)
            })
            .collect::<Vec<_>>();
        assert_eq!(found_addresses, expected_addresses);

        assert_eq!(dataset.manifest().next_row_id, num_rows);
    }

    #[tokio::test]
    async fn test_row_ids_overwrite() {
        // Validate we don't re-use after overwriting
        let num_rows = 10u64;
        let batch = sequence_batch(0..num_rows as i32);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &temp_dir;
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.manifest().next_row_id, num_rows);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();

        // Overwriting should NOT reset the row id counter.
        assert_eq!(dataset.manifest().next_row_id, 2 * num_rows);

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).is_none());
        assert!(index.get(num_rows).is_some());
    }

    #[tokio::test]
    async fn test_row_ids_append() {
        // Validate we handle row ids well when appending concurrently.
        fn write_batch(uri: &str, start: i32) -> impl Future<Output = Result<()>> + '_ {
            let batch = sequence_batch(start..(start + 10));
            let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
            let write_params = WriteParams {
                enable_stable_row_ids: true,
                mode: WriteMode::Append,
                ..Default::default()
            };
            async move {
                let _ = Dataset::write(reader, uri, Some(write_params)).await?;
                Ok(())
            }
        }

        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &temp_dir;
        let mut start = 0;
        // Just do one first to create the dataset.
        write_batch(tmp_path, start).await.unwrap();
        start += 10;
        // Now do the rest concurrently.
        let futures = (0..5)
            .map(|offset| write_batch(tmp_path, start + offset * 10))
            .collect::<Vec<_>>();
        futures::future::try_join_all(futures).await.unwrap();

        let dataset = DatasetBuilder::from_uri(tmp_path).load().await.unwrap();

        assert_eq!(dataset.manifest().next_row_id, 60);

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).is_some());
        assert!(index.get(60).is_none());
    }

    #[tokio::test]
    async fn test_scan_row_ids() {
        // Write dataset with multiple files -> _rowid != _rowaddr
        // Scan with and without each.;
        let batch = sequence_batch(0..6);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 2,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

        for with_row_id in [true, false] {
            for with_row_address in &[true, false] {
                for projection in &[vec![], vec!["id"]] {
                    if !with_row_id && !with_row_address && projection.is_empty() {
                        continue;
                    }

                    let mut scan = dataset.scan();
                    if with_row_id {
                        scan.with_row_id();
                    }
                    if *with_row_address {
                        scan.with_row_address();
                    }
                    let scan = scan.project(projection).unwrap();
                    let result = scan.try_into_batch().await.unwrap();

                    if with_row_id {
                        let row_ids = result[ROW_ID]
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap();
                        let expected = vec![0, 1, 2, 3, 4, 5].into();
                        assert_eq!(row_ids, &expected);
                    }

                    if *with_row_address {
                        let row_addrs = result[ROW_ADDR]
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap();
                        let expected =
                            vec![0, 1, 1 << 32, (1 << 32) + 1, 2 << 32, (2 << 32) + 1].into();
                        assert_eq!(row_addrs, &expected);
                    }

                    if !projection.is_empty() {
                        let ids = result["id"].as_any().downcast_ref::<Int32Array>().unwrap();
                        let expected = vec![0, 1, 2, 3, 4, 5].into();
                        assert_eq!(ids, &expected);
                    }
                }
            }
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn test_delete_with_row_ids(#[values(true, false)] with_scalar_index: bool) {
        let batch = sequence_batch(0..6);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 2,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

        if with_scalar_index {
            dataset
                .create_index(
                    &["id"],
                    IndexType::Scalar,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        }

        dataset.delete("id = 3 or id = 4").await.unwrap();

        let mut scan = dataset.scan();
        scan.with_row_id().with_row_address();
        let result = scan.try_into_batch().await.unwrap();

        let row_ids = result[ROW_ID]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let expected = vec![0, 1, 2, 5].into();
        assert_eq!(row_ids, &expected);

        let row_addrs = result[ROW_ADDR]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let expected = vec![0, 1, 1 << 32, (2 << 32) + 1].into();
        assert_eq!(row_addrs, &expected);
    }

    #[tokio::test]
    async fn test_row_ids_update() {
        // Updated fragments get fresh row ids.
        let num_rows = 5u64;
        let batch = sequence_batch(0..num_rows as i32);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.manifest().next_row_id, num_rows);

        let update_result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 3")
            .unwrap()
            .set("id", "100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = update_result.new_dataset;
        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).is_some());
        // the updated row ids mapping to new address
        assert_eq!(index.get(3), Some(RowAddress::new_from_parts(1, 0)));
        // there is no new row id
        assert_eq!(index.get(5), None);
    }

    fn build_rowid_to_i_map(row_ids: &UInt64Array, i_array: &Int32Array) -> HashMap<u64, i32> {
        row_ids
            .values()
            .iter()
            .zip(i_array.values().iter())
            .map(|(&row_id, &i)| (row_id, i))
            .collect()
    }

    async fn scan_rowid_map(dataset: &Dataset) -> HashMap<u64, i32> {
        let mut scan = dataset.scan();
        scan.with_row_id();
        scan.scan_in_order(true);
        let result = scan.try_into_batch().await.unwrap();
        let i = result["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let row_ids = result[ROW_ID]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        build_rowid_to_i_map(row_ids, i)
    }

    async fn compact(dataset: &mut Dataset, target_rows: usize) {
        let options = CompactionOptions {
            target_rows_per_fragment: target_rows,
            ..Default::default()
        };
        let _ = compact_files(dataset, options, None).await.unwrap();
    }

    async fn delete(dataset: &mut Dataset, expr: &str) {
        dataset.delete(expr).await.unwrap();
    }

    #[tokio::test]
    async fn test_stable_row_id_after_multiple_deletion_and_compaction() {
        async fn delete(dataset: &mut Dataset, expr: &str) {
            dataset.delete(expr).await.unwrap();
        }

        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col(
                "category",
                lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
            )
            .into_ram_dataset_with_params(
                FragmentCount::from(6),
                FragmentRowCount::from(10),
                Some(WriteParams {
                    max_rows_per_file: 10,
                    enable_stable_row_ids: true,
                    enable_v2_manifest_paths: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        // first delete and compact
        delete(&mut dataset, "i = 2 or i = 3 or i = 5").await;
        let map_before = scan_rowid_map(&dataset).await;
        compact(&mut dataset, 20).await;
        let map_after = scan_rowid_map(&dataset).await;

        // verify row id
        assert_eq!(
            map_before.keys().collect::<HashSet<_>>(),
            map_after.keys().collect::<HashSet<_>>()
        );
        for row_id in map_before.keys() {
            assert_eq!(map_before[row_id], map_after[row_id]);
        }

        // second delete
        delete(&mut dataset, "i = 9").await;
        let mut scan = dataset.scan();
        let result = scan
            .filter("i >= 0")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = result["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let id_set = ids.values().iter().cloned().collect::<HashSet<_>>();
        let expected: Vec<i32> = (0..60)
            .filter(|&i| i != 2 && i != 3 && i != 5 && i != 9)
            .collect();
        assert_eq!(id_set, expected.iter().cloned().collect::<HashSet<_>>());

        // get the row_id where i == 15
        let mut scan = dataset.scan();
        scan.with_row_id();
        scan.scan_in_order(true);
        let result = scan
            .filter("i == 15")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let row_id_vec = result[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();

        // third delete and compact
        delete(&mut dataset, "i = 15 or i = 25").await;
        let map_before = scan_rowid_map(&dataset).await;
        compact(&mut dataset, 30).await;
        let map_after = scan_rowid_map(&dataset).await;

        assert_eq!(
            map_before.keys().collect::<HashSet<_>>(),
            map_after.keys().collect::<HashSet<_>>()
        );
        for row_id in map_before.keys() {
            assert_eq!(map_before[row_id], map_after[row_id]);
        }

        // verify the rowid represent i == 15 has been deleted
        let result = dataset
            .take_rows(&row_id_vec, Schema::try_from(dataset.schema()).unwrap())
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[tokio::test]
    async fn test_stable_row_id_after_deletion_update_and_compaction() {
        // gen dataset
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "i",
                lance_datagen::array::step::<arrow_array::types::Int32Type>(),
            )
            .col(
                "category",
                lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
            )
            .into_ram_dataset_with_params(
                FragmentCount::from(6),
                FragmentRowCount::from(10),
                Some(WriteParams {
                    max_rows_per_file: 10,
                    enable_stable_row_ids: true,
                    enable_v2_manifest_paths: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        // delete some rows
        delete(&mut dataset, "i = 2 or i = 3 or i = 5").await;
        let map_before = scan_rowid_map(&dataset).await;

        // update some rows
        let updated_dataset = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i >= 15")
            .unwrap()
            .set("category", "999")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;

        // compact the dataset
        let mut dataset = Arc::try_unwrap(updated_dataset).expect("no other Arc references");
        compact(&mut dataset, 20).await;
        let map_after = scan_rowid_map(&dataset).await;

        // verify row id
        assert_eq!(
            map_before.keys().collect::<HashSet<_>>(),
            map_after.keys().collect::<HashSet<_>>()
        );
        for row_id in map_before.keys() {
            assert_eq!(map_before[row_id], map_after[row_id]);
        }

        // verify category filed
        let mut scan = dataset.scan();
        scan.with_row_id();
        scan.scan_in_order(true);
        let result = scan.try_into_batch().await.unwrap();
        let i = result["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let category = result["category"]
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for idx in 0..i.len() {
            if i.value(idx) >= 15 {
                assert_eq!(category.value(idx), 999);
            }
        }
    }
}
