// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::Dataset;
use crate::{Error, Result};
use futures::{Stream, StreamExt, TryFutureExt, TryStreamExt};
use snafu::{location, Location};
use std::sync::Arc;

use lance_table::{
    format::{Fragment, RowIdMeta},
    rowids::{read_row_ids, RowIdIndex, RowIdSequence},
};

/// Load a row id sequence from the given dataset and fragment.
pub async fn load_row_id_sequence(
    dataset: &Dataset,
    fragment: &Fragment,
) -> Result<Arc<RowIdSequence>> {
    // Virtual path to prevent collisions in the cache.
    let path = dataset.base.child(fragment.id.to_string()).child("row_ids");
    match &fragment.row_id_meta {
        None => Err(Error::Internal {
            message: "Missing row id meta".into(),
            location: location!(),
        }),
        Some(RowIdMeta::Inline(data)) => {
            dataset
                .session
                .file_metadata_cache
                .get_or_insert(&path, |_path| async { read_row_ids(data) })
                .await
        }
        Some(RowIdMeta::External(file_slice)) => {
            dataset
                .session
                .file_metadata_cache
                .get_or_insert(&path, |_path| async {
                    let path = dataset.base.child(file_slice.path.as_str());
                    let range = file_slice.offset as usize
                        ..(file_slice.offset as usize + file_slice.size as usize);
                    let data = dataset
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
        .buffer_unordered(num_cpus::get())
}

pub async fn get_row_id_index(
    dataset: &Dataset,
) -> Result<Option<Arc<lance_table::rowids::RowIdIndex>>> {
    if dataset.manifest.uses_move_stable_row_ids() {
        // The path here isn't real, it's just used to prevent collisions in the cache.
        let path = dataset
            .base
            .child("row_ids")
            .child(dataset.manifest.version.to_string());
        let index = dataset
            .session
            .file_metadata_cache
            .get_or_insert(&path, |_path| async { load_row_id_index(dataset).await })
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

    let index = RowIdIndex::new(&sequences)?;

    Ok(index)
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use crate::dataset::{builder::DatasetBuilder, UpdateBuilder, WriteMode, WriteParams};

    use super::*;

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, UInt64Array};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::Future;
    use lance_core::{utils::address::RowAddress, ROW_ADDR, ROW_ID};

    fn sequence_batch(values: Range<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(values))],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_empty_dataset_rowids() {
        let schema = sequence_batch(0..0).schema();
        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone());
        let write_params = WriteParams {
            enable_move_stable_row_ids: true,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        assert!(dataset.manifest.uses_move_stable_row_ids());

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).is_none());

        assert_eq!(dataset.manifest().next_row_id, 0);
    }

    #[tokio::test]
    async fn test_must_set_on_creation() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap();

        let batch = sequence_batch(0..10);
        let reader =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), batch.schema());
        let write_params = WriteParams {
            enable_move_stable_row_ids: false,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();
        assert!(!dataset.manifest().uses_move_stable_row_ids());

        // Trying to append without stable row ids should fail.
        let write_params = WriteParams {
            enable_move_stable_row_ids: true,
            mode: WriteMode::Append,
            ..Default::default()
        };
        let reader =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), batch.schema());
        let result = Dataset::write(reader, tmp_path, Some(write_params)).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(),
                Error::NotSupported { source, .. } 
                    if source.to_string().contains("Cannot enable stable row ids on existing dataset")));
    }

    #[tokio::test]
    async fn test_new_row_ids() {
        let num_rows = 25u64;
        let batch = sequence_batch(0..num_rows as i32);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_move_stable_row_ids: true,
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
            enable_move_stable_row_ids: true,
            ..Default::default()
        };
        let temp_dir = tempfile::tempdir().unwrap();
        let tmp_path = temp_dir.path().to_str().unwrap();
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
        fn write_batch<'a>(uri: &'a str, start: &mut i32) -> impl Future<Output = Result<()>> + 'a {
            let batch = sequence_batch(*start..(*start + 10));
            *start += 10;
            let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
            let write_params = WriteParams {
                enable_move_stable_row_ids: true,
                mode: WriteMode::Append,
                ..Default::default()
            };
            async move {
                let _ = Dataset::write(reader, uri, Some(write_params)).await?;
                Ok(())
            }
        }

        let temp_dir = tempfile::tempdir().unwrap();
        let tmp_path = temp_dir.path().to_str().unwrap();
        let mut start = 0;
        // Just do one first to create the dataset.
        write_batch(tmp_path, &mut start).await.unwrap();
        // Now do the rest concurrently.
        let futures = (0..5)
            .map(|_| write_batch(tmp_path, &mut start))
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
            enable_move_stable_row_ids: true,
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
    async fn test_delete_with_row_ids() {
        let batch = sequence_batch(0..6);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_move_stable_row_ids: true,
            max_rows_per_file: 2,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

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
            enable_move_stable_row_ids: true,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.manifest().next_row_id, num_rows);

        let dataset = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 3")
            .unwrap()
            .set("id", "100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).is_some());
        // Old address is still there.
        assert_eq!(index.get(3), Some(RowAddress::new_from_parts(0, 3)));
        // New location is there.
        assert_eq!(index.get(5), Some(RowAddress::new_from_parts(1, 0)));
    }

    // TODO: query / scan / take after deletion, compaction, then deletion
}
