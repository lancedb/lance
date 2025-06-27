// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{
    dataset::transaction::{Operation, Transaction},
    Dataset,
};
use futures::{stream, StreamExt, TryStreamExt};
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::Result;
use lance_table::format::Fragment;

pub async fn delete(ds: &mut Dataset, predicate: &str) -> Result<()> {
    let mut updated_fragments: Vec<Fragment> = Vec::new();
    let mut deleted_fragment_ids: Vec<u64> = Vec::new();
    stream::iter(ds.get_fragments())
        .map(|f| async move {
            let old_fragment = f.metadata.clone();
            let new_fragment = f.delete(predicate).await?.map(|f| f.metadata);
            Ok((old_fragment, new_fragment))
        })
        .buffer_unordered(get_num_compute_intensive_cpus())
        // Drop the fragments that were deleted.
        .try_for_each(|(old_fragment, new_fragment)| {
            if let Some(new_fragment) = new_fragment {
                if new_fragment != old_fragment {
                    updated_fragments.push(new_fragment);
                }
            } else {
                deleted_fragment_ids.push(old_fragment.id);
            }
            futures::future::ready(Ok::<_, crate::Error>(()))
        })
        .await?;

    let transaction = Transaction::new(
        ds.manifest.version,
        Operation::Delete {
            updated_fragments,
            deleted_fragment_ids,
            predicate: predicate.to_string(),
        },
        // No change is needed to the blobs dataset.  The blobs are implicitly deleted since the
        // rows that reference them are deleted.
        /*blobs_op=*/
        None,
        None,
    );

    ds.apply_commit(transaction, &Default::default(), &Default::default())
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::dataset::InsertBuilder;
    use crate::dataset::{WriteMode, WriteParams};
    use crate::utils::test::TestDatasetGenerator;
    use arrow_array::{RecordBatch, UInt32Array};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_file::version::LanceFileVersion;
    use lance_index::{scalar::ScalarIndexParams, DatasetIndexExt, IndexType};
    use rstest::rstest;
    use std::collections::HashSet;
    use std::ops::Range;
    use std::sync::Arc;

    #[rstest]
    #[tokio::test]
    async fn test_delete(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] with_scalar_index: bool,
    ) {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("i", DataType::UInt32, false),
                ArrowField::new("x", DataType::UInt32, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from_iter_values(range.clone())),
                    Arc::new(UInt32Array::from_iter_values(range.map(|v| v * 2))),
                ],
            )
            .unwrap()
        }
        // Write a dataset
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_string();
        let data = sequence_data(0..100);
        // Split over two files.
        let batches = vec![data.slice(0, 50), data.slice(50, 50)];
        let mut dataset = TestDatasetGenerator::new(batches, data_storage_version)
            .make_hostile(&tmp_path)
            .await;

        if with_scalar_index {
            dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    Some("scalar_index".to_string()),
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        }

        // Delete nothing
        dataset.delete("i < 0").await.unwrap();
        dataset.validate().await.unwrap();

        // We should not have any deletion file still
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(dataset.count_fragments(), 2);
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 0);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(1));
        assert!(fragments[0].metadata.deletion_file.is_none());
        assert!(fragments[1].metadata.deletion_file.is_none());

        // Delete rows
        dataset.delete("i < 10 OR i >= 90").await.unwrap();
        dataset.validate().await.unwrap();

        // Verify result:
        // There should be a deletion file in the metadata
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(dataset.count_fragments(), 2);
        assert!(fragments[0].metadata.deletion_file.is_some());
        assert!(fragments[1].metadata.deletion_file.is_some());
        assert_eq!(
            fragments[0]
                .metadata
                .deletion_file
                .as_ref()
                .unwrap()
                .num_deleted_rows,
            Some(10)
        );
        assert_eq!(
            fragments[1]
                .metadata
                .deletion_file
                .as_ref()
                .unwrap()
                .num_deleted_rows,
            Some(10)
        );

        // The deletion file should contain 20 rows
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 20);
        // First fragment has 0..10 deleted
        let deletion_vector = fragments[0].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(deletion_vector.len(), 10);
        assert_eq!(
            deletion_vector.iter().collect::<HashSet<_>>(),
            (0..10).collect::<HashSet<_>>()
        );
        // Second fragment has 90..100 deleted
        let deletion_vector = fragments[1].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(deletion_vector.len(), 10);
        // The second fragment starts at 50, so 90..100 becomes 40..50 in local row ids.
        assert_eq!(
            deletion_vector.iter().collect::<HashSet<_>>(),
            (40..50).collect::<HashSet<_>>()
        );
        let second_deletion_file = fragments[1].metadata.deletion_file.clone().unwrap();

        // Delete more rows
        dataset.delete("i < 20").await.unwrap();
        dataset.validate().await.unwrap();

        // Verify result
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 30);
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert!(fragments[0].metadata.deletion_file.is_some());
        let deletion_vector = fragments[0].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(deletion_vector.len(), 20);
        assert_eq!(
            deletion_vector.iter().collect::<HashSet<_>>(),
            (0..20).collect::<HashSet<_>>()
        );
        // Second deletion vector was not rewritten
        assert_eq!(
            fragments[1].metadata.deletion_file.as_ref().unwrap(),
            &second_deletion_file
        );

        // Delete full fragment
        dataset.delete("i >= 50").await.unwrap();
        dataset.validate().await.unwrap();

        // Verify second fragment is fully gone
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert_eq!(dataset.count_fragments(), 1);
        assert_eq!(fragments[0].id(), 0);

        // Verify the count_deleted_rows only contains the rows from the first fragment
        // i.e. - deleted_rows from the fragment that has been deleted are not counted
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 20);

        // Append after delete
        let data = sequence_data(0..100);
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&write_params)
            .execute(vec![data])
            .await
            .unwrap();

        dataset.validate().await.unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(dataset.count_fragments(), 2);
        // Fragment id picks up where we left off
        assert_eq!(fragments[0].id(), 0);
        assert_eq!(fragments[1].id(), 2);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(2));
    }
}
