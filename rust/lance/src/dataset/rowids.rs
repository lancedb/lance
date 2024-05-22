// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::Dataset;
use crate::{Error, Result};
use futures::{StreamExt, TryStreamExt};
use snafu::{location, Location};
use std::sync::Arc;

use lance_table::{
    format::RowIdMeta,
    rowids::{read_row_ids, RowIdIndex},
};

// TODO: remove allow unused once we start using this in query and take paths.
#[allow(unused)]
pub async fn get_row_id_index(dataset: &Dataset) -> Result<Arc<lance_table::rowids::RowIdIndex>> {
    // The path here isn't real, it's just used to prevent collisions in the cache.
    let path = dataset
        .base
        .child("row_ids")
        .child(dataset.manifest.version.to_string());
    dataset
        .session
        .file_metadata_cache
        .get_or_insert(&path, |_path| async { load_row_id_index(dataset).await })
        .await
}

async fn load_row_id_index(dataset: &Dataset) -> Result<lance_table::rowids::RowIdIndex> {
    let mut num_external = 0;
    for fragment in dataset.manifest.fragments.iter() {
        match fragment.row_id_meta {
            Some(RowIdMeta::External(_)) => num_external += 1,
            None => {
                return Err(Error::Internal {
                    message: "Missing row id meta".into(),
                    location: location!(),
                })
            }
            _ => {}
        }
    }

    let mut external_files = Vec::with_capacity(num_external);
    let mut inline_files = Vec::with_capacity(dataset.manifest.fragments.len() - num_external);
    for fragment in dataset.manifest.fragments.iter() {
        match &fragment.row_id_meta {
            Some(RowIdMeta::External(file_slice)) => {
                external_files.push((fragment.id as u32, file_slice))
            }
            Some(RowIdMeta::Inline(row_ids)) => inline_files.push((fragment.id as u32, row_ids)),
            _ => {}
        }
    }

    let mut sequences = Vec::with_capacity(dataset.manifest.fragments.len());
    futures::stream::iter(external_files)
        .map(|(id, file_slice)| async move {
            let path = dataset.base.child(file_slice.path.as_str());
            let range =
                file_slice.offset as usize..(file_slice.offset as usize + file_slice.size as usize);
            let data = dataset
                .object_store
                .open(&path)
                .await?
                .get_range(range)
                .await?;
            let sequence = read_row_ids(&data)?;
            Ok::<_, Error>((id, sequence))
        })
        .buffer_unordered(num_cpus::get())
        .try_for_each(|(id, sequence)| {
            sequences.push((id, sequence));
            futures::future::ready(Ok(()))
        })
        .await?;

    for (id, row_ids) in inline_files {
        let sequence = read_row_ids(&row_ids)?;
        sequences.push((id, sequence));
    }

    let index = RowIdIndex::new(&sequences)?;

    Ok(index)
}

#[cfg(test)]
mod test {
    use crate::dataset::WriteParams;

    use super::*;

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::utils::address::RowAddress;

    // TODO: add tests once we support writing.

    #[tokio::test]
    async fn test_empty_dataset_rowids() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone());
        let write_params = WriteParams {
            enable_experimental_stable_row_ids: true,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        let index = get_row_id_index(&dataset).await.unwrap();
        assert!(index.get(0).is_none());

        assert_eq!(dataset.manifest().next_row_id, 0);
    }

    #[tokio::test]
    async fn test_new_row_ids() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let num_rows = 25u64;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..num_rows as i32))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            enable_experimental_stable_row_ids: true,
            max_rows_per_file: 10,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        let index = get_row_id_index(&dataset).await.unwrap();

        let found_addresses = (0..num_rows)
            .into_iter()
            .map(|i| index.get(i as u64).unwrap())
            .collect::<Vec<_>>();
        let expected_addresses = (0..num_rows)
            .into_iter()
            .map(|i| {
                let fragment_id = i / 10;
                RowAddress::new_from_parts(fragment_id as u32, i as u32)
            })
            .collect::<Vec<_>>();
        assert_eq!(found_addresses, expected_addresses);

        assert_eq!(dataset.manifest().next_row_id, num_rows);
    }

    // TODO: test with deletions, updates, compact things do the right thing.

    // TODO: test scan with row id produces correct values.
}
