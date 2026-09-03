// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Building and loading the stable partition system index entry.
//!
//! A reordered rewrite job writes its row map file first (see
//! `lance_index::frag_reuse::row_map`), then calls
//! [`build_stable_partition_rewrite`] with the sources, destinations and the
//! counts matrix the row map writer returned. The result rides on
//! `Operation::Rewrite::stable_partition` through the ordinary commit path;
//! `lance-table`'s manifest build splices the entry and leaves the reordered
//! groups' index bitmaps untouched.

use crate::Dataset;
use crate::index::DatasetIndexExt;
use lance_core::Error;
use lance_core::utils::stable_partition::CountsMatrix;
use lance_table::format::pb::ExternalFile;
use lance_table::format::pb::StablePartitionIndexDetails as PbStablePartitionIndexDetails;
use lance_table::format::pb::stable_partition_index_details::{Content, InlineContent};
use lance_table::format::{Fragment, IndexMetadata};
use lance_table::system_index::frag_reuse::FragDigest;
use lance_table::system_index::stable_partition::{
    STABLE_PARTITION_DETAILS_FILE_NAME, STABLE_PARTITION_INDEX_NAME, StablePartitionIndexDetails,
    StablePartitionTransition,
};
use lance_table::transaction::StablePartitionRewrite;
use prost::Message;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

/// Load the stable partition details from an index metadata entry.
pub async fn load_stable_partition_details(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> lance_core::Result<StablePartitionIndexDetails> {
    let details_any = index.index_details.as_ref();
    if details_any.is_none_or(|details| !details.type_url.ends_with("StablePartitionIndexDetails"))
    {
        return Err(Error::index(
            "Index details is not for the stable partition index",
        ));
    }

    let proto = details_any
        .unwrap()
        .to_msg::<PbStablePartitionIndexDetails>()?;
    match &proto.content {
        None => Err(Error::index("Index details content is not found")),
        Some(Content::Inline(content)) => StablePartitionIndexDetails::try_from(content.clone()),
        Some(Content::External(external_file)) => {
            let file_path = dataset
                .indices_dir()
                .join(index.uuid.to_string())
                .join(external_file.path.clone());
            let range = external_file.offset as usize
                ..(external_file.offset as usize + external_file.size as usize);
            let data = dataset
                .object_store
                .open(&file_path)
                .await?
                .get_range(range)
                .await?;
            StablePartitionIndexDetails::try_from(InlineContent::decode(data)?)
        }
    }
}

/// Validate one reordered rewrite against its row map counts and build the
/// [`StablePartitionRewrite`] to attach to `Operation::Rewrite`.
///
/// `sources` are the rewrite's source fragments in scan order; `destinations`
/// are the new fragments in label order (a row map label indexes this slice);
/// `counts` is the matrix returned by the row map writer's `finish`.
///
/// Conservation checks, all against the counts the row map file itself
/// carries:
///
/// * every destination's physical rows equal its label total,
/// * the row map covers exactly the sources' physical rows,
/// * the labeled (live) rows equal the sources' live rows.
pub async fn build_stable_partition_rewrite(
    dataset: &Dataset,
    sources: Vec<FragDigest>,
    destinations: &[Fragment],
    row_map_id: String,
    row_map_size_bytes: u64,
    counts: &CountsMatrix,
) -> lance_core::Result<StablePartitionRewrite> {
    if destinations.len() as u64 != u64::from(counts.num_destinations()) {
        return Err(Error::invalid_input(format!(
            "the rewrite lists {} destinations but its row map labels {}",
            destinations.len(),
            counts.num_destinations()
        )));
    }
    for (label, destination) in destinations.iter().enumerate() {
        let physical_rows = destination.physical_rows.ok_or_else(|| {
            Error::invalid_input(format!(
                "destination fragment {} has no physical row count",
                destination.id
            ))
        })? as u64;
        let labeled = u64::from(counts.total(label as u16));
        if physical_rows != labeled {
            return Err(Error::invalid_input(format!(
                "destination fragment {} holds {physical_rows} rows but the row map labels {labeled}",
                destination.id
            )));
        }
    }
    let source_physical: u64 = sources.iter().map(|frag| frag.physical_rows as u64).sum();
    if counts.total_rows() != source_physical {
        return Err(Error::invalid_input(format!(
            "the row map covers {} rows but the source fragments hold {source_physical}",
            counts.total_rows()
        )));
    }
    let source_live: u64 = sources
        .iter()
        .map(|frag| (frag.physical_rows - frag.num_deleted_rows) as u64)
        .sum();
    if counts.total_live_rows() != source_live {
        return Err(Error::invalid_input(format!(
            "the row map labels {} live rows but the source fragments hold {source_live}",
            counts.total_live_rows()
        )));
    }

    let transition = StablePartitionTransition {
        dataset_version: dataset.manifest.version,
        sources,
        destinations: destinations.iter().map(|frag| frag.id).collect(),
        row_map_id,
        row_map_size_bytes,
    };
    let reordered_sources = transition.source_ids();

    let index_meta = dataset.load_indices().await.map(|indices| {
        indices
            .iter()
            .find(|idx| idx.name == STABLE_PARTITION_INDEX_NAME)
            .cloned()
    })?;
    let new_details = match &index_meta {
        None => StablePartitionIndexDetails {
            transitions: vec![transition],
        },
        Some(index_meta) => {
            let current = load_stable_partition_details(dataset, index_meta).await?;
            let mut transitions = current.transitions;
            transitions.push(transition);
            StablePartitionIndexDetails { transitions }
        }
    };

    let index_id = Uuid::new_v4();
    let inline = InlineContent::from(&new_details);
    let proto = if inline.encoded_len() > 204800 {
        let file_path = dataset
            .indices_dir()
            .join(index_id.to_string())
            .join(STABLE_PARTITION_DETAILS_FILE_NAME);
        let mut writer = dataset.object_store.create(&file_path).await?;
        writer.write_all(inline.encode_to_vec().as_slice()).await?;
        writer.shutdown().await?;
        PbStablePartitionIndexDetails {
            content: Some(Content::External(ExternalFile {
                path: STABLE_PARTITION_DETAILS_FILE_NAME.to_owned(),
                offset: 0,
                size: inline.encoded_len() as u64,
            })),
        }
    } else {
        PbStablePartitionIndexDetails {
            content: Some(Content::Inline(inline)),
        }
    };

    let index = IndexMetadata {
        uuid: index_id,
        name: STABLE_PARTITION_INDEX_NAME.to_string(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        // Provenance: the union of every transition's source ids,
        // deliberately keeping retired fragments (see
        // `IndexMetadata::fragment_bitmap`).
        fragment_bitmap: Some(new_details.source_bitmap()),
        index_details: Some(std::sync::Arc::new(prost_types::Any::from_msg(&proto)?)),
        index_version: index_meta
            .as_ref()
            .map_or(0, |index_meta| index_meta.index_version),
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        // The row map files live in their own `{indices dir}/{row_map_id}/`
        // directories referenced from the details, not under this entry's
        // uuid; lifecycle integration comes with the read path.
        files: None,
    };

    Ok(StablePartitionRewrite {
        index,
        reordered_sources,
        base_entry_version: index_meta.map(|index_meta| index_meta.dataset_version),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::transaction::{Operation, RewriteGroup, TransactionBuilder};
    use crate::index::{DatasetIndexExt, IndexType};
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::Int32Type;
    use lance_index::frag_reuse::row_map::{RowMapWriter, SourceRows};
    use lance_index::scalar::lance_format::LanceIndexStore;
    use lance_index::scalar::{IndexStore, ScalarIndexParams};
    use lance_table::system_index::is_system_index;
    use lance_table::system_index::stable_partition::ROW_MAP_FILE_NAME;
    use std::sync::Arc;

    async fn test_dataset() -> Dataset {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(100))
            .await
            .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("scalar".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        dataset
    }

    /// Write a row map that alternates the concatenated source rows across
    /// two destinations, returning `(row_map_id, size, counts)`.
    async fn write_alternating_row_map(
        dataset: &Dataset,
        sources: Vec<SourceRows>,
    ) -> (String, u64, CountsMatrix) {
        let row_map_id = Uuid::new_v4().to_string();
        let store = LanceIndexStore::new(
            dataset.object_store.clone(),
            dataset.indices_dir().join(row_map_id.clone()),
            Arc::new(lance_core::cache::LanceCache::with_capacity(1024 * 1024)),
        );
        let writer = store
            .new_index_file(ROW_MAP_FILE_NAME, RowMapWriter::schema())
            .await
            .unwrap();
        let total: u64 = sources.iter().map(|s| s.physical_rows).sum();
        let mut writer = RowMapWriter::try_new(writer, sources, 2).unwrap();
        let labels: Vec<u16> = (0..total).map(|row| (row % 2) as u16).collect();
        writer.append_labels(&labels).await.unwrap();
        let (file, counts) = writer.finish().await.unwrap();
        (row_map_id, file.size_bytes, counts)
    }

    /// The commit path of a reordered rewrite: the stable partition entry is
    /// installed, the scalar index bitmap keeps the retired source ids, and
    /// queries degrade to correct full scans.
    #[tokio::test]
    async fn test_reordered_rewrite_commit() {
        let mut dataset = test_dataset().await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        assert_eq!(old_fragments.len(), 2);
        let sources: Vec<FragDigest> = old_fragments.iter().map(FragDigest::from).collect();

        let (row_map_id, row_map_size, counts) = write_alternating_row_map(
            &dataset,
            old_fragments
                .iter()
                .map(|frag| SourceRows {
                    physical_rows: frag.physical_rows.unwrap() as u64,
                    deleted: None,
                })
                .collect(),
        )
        .await;

        // The destinations reuse the source data files under new fragment
        // ids: the commit path validates metadata and conservation, not row
        // placement, so this stands in for a real reordered rewrite.
        let new_fragments: Vec<Fragment> = old_fragments
            .iter()
            .enumerate()
            .map(|(i, frag)| {
                let mut new_fragment = frag.clone();
                new_fragment.id = 2 + i as u64;
                new_fragment
            })
            .collect();

        let stable_partition = build_stable_partition_rewrite(
            &dataset,
            sources.clone(),
            &new_fragments,
            row_map_id.clone(),
            row_map_size,
            &counts,
        )
        .await
        .unwrap();
        assert_eq!(stable_partition.base_entry_version, None);
        assert_eq!(
            stable_partition.reordered_sources,
            roaring::RoaringBitmap::from_iter([0u32, 1])
        );

        let transaction = TransactionBuilder::new(
            dataset.manifest.version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: old_fragments.clone(),
                    new_fragments: new_fragments.clone(),
                }],
                rewritten_indices: vec![],
                frag_reuse_index: None,
                stable_partition: Some(stable_partition),
            },
        )
        .build();
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        // Fragments swapped.
        let live_ids: Vec<u64> = dataset.fragments().iter().map(|frag| frag.id).collect();
        assert_eq!(live_ids, vec![2, 3]);

        let indices = dataset.load_indices().await.unwrap();
        // The scalar index keeps its retired source ids as provenance
        // instead of following the rewrite.
        let scalar = indices.iter().find(|idx| idx.name == "scalar").unwrap();
        assert_eq!(
            scalar.fragment_bitmap.as_ref().unwrap(),
            &roaring::RoaringBitmap::from_iter([0u32, 1])
        );
        // The stable partition entry is installed and round-trips.
        let entry = indices
            .iter()
            .find(|idx| idx.name == STABLE_PARTITION_INDEX_NAME)
            .unwrap();
        assert!(is_system_index(entry));
        assert_eq!(
            entry.fragment_bitmap.as_ref().unwrap(),
            &roaring::RoaringBitmap::from_iter([0u32, 1])
        );
        let details = load_stable_partition_details(&dataset, entry)
            .await
            .unwrap();
        assert_eq!(details.transitions.len(), 1);
        let transition = &details.transitions[0];
        assert_eq!(transition.sources, sources);
        assert_eq!(transition.destinations, vec![2, 3]);
        assert_eq!(transition.row_map_id, row_map_id);
        assert_eq!(transition.total_source_rows(), 200);

        // Reads degrade instead of erroring: the scalar index covers no live
        // fragment, so this filter falls back to a scan and stays correct.
        assert_eq!(dataset.count_rows(None).await.unwrap(), 200);
        assert_eq!(
            dataset
                .count_rows(Some("i >= 100".to_string()))
                .await
                .unwrap(),
            100
        );

        // A second reordered rewrite built against the pre-commit entry state
        // (base = None) must be rejected: splicing it would drop the
        // transition that just landed.
        let stale = StablePartitionRewrite {
            base_entry_version: None,
            ..build_stable_partition_rewrite(
                &dataset,
                dataset.fragments().iter().map(FragDigest::from).collect(),
                &{
                    let mut fragments: Vec<Fragment> =
                        dataset.fragments().iter().cloned().collect();
                    for (i, fragment) in fragments.iter_mut().enumerate() {
                        fragment.id = 4 + i as u64;
                    }
                    fragments
                },
                row_map_id.clone(),
                row_map_size,
                &counts,
            )
            .await
            .unwrap()
        };
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let new_fragments: Vec<Fragment> = old_fragments
            .iter()
            .enumerate()
            .map(|(i, frag)| {
                let mut new_fragment = frag.clone();
                new_fragment.id = 4 + i as u64;
                new_fragment
            })
            .collect();
        let transaction = TransactionBuilder::new(
            dataset.manifest.version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments,
                    new_fragments,
                }],
                rewritten_indices: vec![],
                frag_reuse_index: None,
                stable_partition: Some(stale),
            },
        )
        .build();
        let err = dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("concurrent reordered rewrite"),
            "{err}"
        );
    }

    /// Conservation violations are rejected before any transaction exists.
    #[tokio::test]
    async fn test_conservation_validation() {
        let dataset = test_dataset().await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let sources: Vec<FragDigest> = old_fragments.iter().map(FragDigest::from).collect();
        let (row_map_id, row_map_size, counts) = write_alternating_row_map(
            &dataset,
            old_fragments
                .iter()
                .map(|frag| SourceRows {
                    physical_rows: frag.physical_rows.unwrap() as u64,
                    deleted: None,
                })
                .collect(),
        )
        .await;

        // A destination whose physical rows disagree with its label total.
        let mut wrong_rows: Vec<Fragment> = old_fragments.clone();
        for (i, fragment) in wrong_rows.iter_mut().enumerate() {
            fragment.id = 2 + i as u64;
        }
        wrong_rows[0].physical_rows = Some(99);
        let err = build_stable_partition_rewrite(
            &dataset,
            sources.clone(),
            &wrong_rows,
            row_map_id.clone(),
            row_map_size,
            &counts,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("row map labels"), "{err}");

        // A destination count that disagrees with the row map's m.
        let err = build_stable_partition_rewrite(
            &dataset,
            sources.clone(),
            &old_fragments[..1],
            row_map_id.clone(),
            row_map_size,
            &counts,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("destinations"), "{err}");

        // Sources that disagree with the row map's row count.
        let err = build_stable_partition_rewrite(
            &dataset,
            sources[..1].to_vec(),
            &{
                let mut fragments = old_fragments.clone();
                for (i, fragment) in fragments.iter_mut().enumerate() {
                    fragment.id = 2 + i as u64;
                }
                fragments
            },
            row_map_id,
            row_map_size,
            &counts,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("row map covers"), "{err}");
    }
}
