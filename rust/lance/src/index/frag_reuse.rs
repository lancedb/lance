// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use crate::dataset::optimize::remapping::transpose_row_ids_from_digest;
use crate::index::DatasetIndexExt;
use lance_core::Error;
use lance_index::frag_reuse::{
    FRAG_REUSE_DETAILS_FILE_NAME, FRAG_REUSE_INDEX_NAME, FragReuseGroup, FragReuseIndex,
    FragReuseIndexDetails, FragReuseVersion,
};
use lance_table::format::IndexMetadata;
use lance_table::format::pb::fragment_reuse_index_details::{Content, InlineContent};
use lance_table::format::pb::{ExternalFile, FragmentReuseIndexDetails};
use prost::Message;
use roaring::{RoaringBitmap, RoaringTreemap};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

/// Load fragment reuse index details from index metadata
pub async fn load_frag_reuse_index_details(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> lance_core::Result<Arc<FragReuseIndexDetails>> {
    let details_any = index.index_details.clone();
    if details_any.is_none()
        || !details_any
            .as_ref()
            .unwrap()
            .type_url
            .ends_with("FragmentReuseIndexDetails")
    {
        return Err(Error::index(
            "Index details is not for the fragment reuse index",
        ));
    }

    let proto = details_any.unwrap().to_msg::<FragmentReuseIndexDetails>()?;
    match &proto.content {
        None => Err(Error::index("Index details content is not found")),
        Some(Content::Inline(content)) => {
            Ok(Arc::new(FragReuseIndexDetails::try_from(content.clone())?))
        }
        Some(Content::External(external_file)) => {
            let file_path = dataset
                .indices_dir()
                .join(index.uuid.to_string())
                .join(external_file.path.clone());

            // Use checked arithmetic and bounds validation so that malformed
            // metadata surfaces as a proper `Error` (and therefore a Python
            // exception) instead of panicking or building an invalid range.
            let offset = usize::try_from(external_file.offset).map_err(|_| {
                Error::index(format!(
                    "Fragment reuse external file offset {} does not fit in usize",
                    external_file.offset
                ))
            })?;
            let size = usize::try_from(external_file.size).map_err(|_| {
                Error::index(format!(
                    "Fragment reuse external file size {} does not fit in usize",
                    external_file.size
                ))
            })?;
            // An external details file is only written when the encoded payload exceeds
            // the inline threshold, so a zero size is always corruption. Rejecting it
            // matters because it would otherwise slip past the range check below and
            // decode as an empty (but structurally valid) index with zero reuse versions,
            // which is indistinguishable from a legitimately fully-cleaned-up index.
            if size == 0 {
                return Err(Error::index(
                    "Fragment reuse external file size is 0, which cannot be a valid details file"
                        .to_string(),
                ));
            }
            let end = offset.checked_add(size).ok_or_else(|| {
                Error::index(format!(
                    "Fragment reuse external file range overflows: offset {offset} + size {size}"
                ))
            })?;

            // the file content will be cached in the index cache later
            // so we do not put it to the file cache
            let reader = dataset.object_store.open(&file_path).await?;
            // Propagate the object-store error unchanged: a transient read failure must
            // stay an IO error (and reach Python as IOError) rather than be relabelled
            // as index corruption.
            let file_size = reader.size().await?;
            if end > file_size {
                return Err(Error::index(format!(
                    "Fragment reuse external file range {offset}..{end} is out of bounds for file {file_path} of size {file_size}"
                )));
            }
            let data = reader.get_range(offset..end).await?;

            let pb_sequence = InlineContent::decode(data)?;
            Ok(Arc::new(FragReuseIndexDetails::try_from(pb_sequence)?))
        }
    }
}

/// open fragment reuse index based on its metadata details
pub(crate) async fn open_frag_reuse_index(
    uuid: Uuid,
    details: &FragReuseIndexDetails,
) -> lance_core::Result<FragReuseIndex> {
    let mut row_id_maps: Vec<HashMap<u64, Option<u64>>> =
        Vec::with_capacity(details.versions.len());
    for version in &details.versions {
        let mut row_id_map = HashMap::<u64, Option<u64>>::new();
        for group in version.groups.iter() {
            let cursor = Cursor::new(&group.changed_row_addrs);
            // The row-address digest is opaque bytes inside an otherwise valid protobuf,
            // so a corrupt payload must surface here as an error rather than a panic that
            // aborts the caller (including the Python cleanup binding).
            let changed_row_addrs = RoaringTreemap::deserialize_from(cursor).map_err(|e| {
                Error::index(format!(
                    "Failed to decode changed row addresses in fragment reuse index {uuid} \
                         at dataset version {}: {e}",
                    version.dataset_version
                ))
            })?;
            let group_row_id_map = transpose_row_ids_from_digest(
                changed_row_addrs,
                &group.old_frags,
                &group.new_frags,
            );
            row_id_map.extend(group_row_id_map);
        }
        row_id_maps.push(row_id_map);
    }

    Ok(FragReuseIndex::new(uuid, row_id_maps, details.clone()))
}

pub(crate) async fn build_new_frag_reuse_index(
    dataset: &mut Dataset,
    frag_reuse_groups: Vec<FragReuseGroup>,
    new_fragment_bitmap: RoaringBitmap,
) -> lance_core::Result<IndexMetadata> {
    let new_version = FragReuseVersion {
        dataset_version: dataset.manifest.version,
        groups: frag_reuse_groups,
    };

    let index_meta = dataset.load_indices().await.map(|indices| {
        indices
            .iter()
            .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
            .cloned()
    })?;

    let new_index_details = match &index_meta {
        None => FragReuseIndexDetails {
            versions: Vec::from([new_version]),
        },
        Some(index_meta) => {
            let current_details = load_frag_reuse_index_details(dataset, index_meta).await?;
            let mut versions = current_details.versions.clone();
            versions.push(new_version);
            FragReuseIndexDetails { versions }
        }
    };

    build_frag_reuse_index_metadata(
        dataset,
        index_meta.as_ref(),
        new_index_details,
        new_fragment_bitmap,
    )
    .await
}

pub(crate) async fn build_frag_reuse_index_metadata(
    dataset: &Dataset,
    index_meta: Option<&IndexMetadata>,
    new_index_details: FragReuseIndexDetails,
    new_fragment_bitmap: RoaringBitmap,
) -> lance_core::Result<IndexMetadata> {
    let index_id = uuid::Uuid::new_v4();
    let new_index_details_proto = InlineContent::from(&new_index_details);
    let proto = if new_index_details_proto.encoded_len() > 204800 {
        let file_path = dataset
            .indices_dir()
            .join(index_id.to_string())
            .join(FRAG_REUSE_DETAILS_FILE_NAME);
        let mut writer = dataset.object_store.create(&file_path).await?;
        writer
            .write_all(new_index_details_proto.encode_to_vec().as_slice())
            .await?;
        writer.shutdown().await?;
        let external_file = ExternalFile {
            path: FRAG_REUSE_DETAILS_FILE_NAME.to_owned(),
            offset: 0,
            size: new_index_details_proto.encoded_len() as u64,
        };
        FragmentReuseIndexDetails {
            content: Some(Content::External(external_file)),
        }
    } else {
        FragmentReuseIndexDetails {
            content: Some(Content::Inline(new_index_details_proto)),
        }
    };

    Ok(IndexMetadata {
        uuid: index_id,
        name: FRAG_REUSE_INDEX_NAME.to_string(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(new_fragment_bitmap),
        index_details: Some(Arc::new(prost_types::Any::from_msg(&proto)?)),
        index_version: index_meta.map_or(0, |index_meta| index_meta.index_version),
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        // Fragment reuse index is inline (no files)
        files: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::Int32Type;
    use rstest::rstest;

    /// Size of the valid external details file written by the malformed-input test.
    const PAYLOAD_LEN: usize = 16;

    /// Build index metadata pointing at an external details file with the given
    /// `offset`/`size`, so we can exercise the offset/size arithmetic in
    /// `load_frag_reuse_index_details` with crafted (potentially malformed) values.
    fn external_details_meta(uuid: Uuid, offset: u64, size: u64) -> IndexMetadata {
        let proto = FragmentReuseIndexDetails {
            content: Some(Content::External(ExternalFile {
                path: FRAG_REUSE_DETAILS_FILE_NAME.to_owned(),
                offset,
                size,
            })),
        };
        IndexMetadata {
            uuid,
            name: FRAG_REUSE_INDEX_NAME.to_string(),
            fields: vec![],
            covering_fields: vec![],
            dataset_version: 0,
            fragment_bitmap: Some(RoaringBitmap::new()),
            index_details: Some(Arc::new(prost_types::Any::from_msg(&proto).unwrap())),
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    /// `num_deleted_rows` is read verbatim from persisted metadata, and
    /// `transpose_row_ids_from_digest` used it directly as a `HashMap::with_capacity`
    /// hint. A corrupt value therefore aborted the caller with a capacity-overflow panic
    /// (or thrashed on merely large values) on the same path the Python cleanup binding
    /// reaches, defeating the "corrupt metadata becomes an exception" contract.
    #[tokio::test]
    async fn test_open_frag_reuse_index_tolerates_corrupt_num_deleted_rows() {
        let mut changed_row_addrs = Vec::new();
        RoaringTreemap::new()
            .serialize_into(&mut changed_row_addrs)
            .unwrap();

        let details = FragReuseIndexDetails {
            versions: vec![FragReuseVersion {
                dataset_version: 1,
                groups: vec![FragReuseGroup {
                    changed_row_addrs,
                    old_frags: vec![lance_table::system_index::frag_reuse::FragDigest {
                        id: 0,
                        physical_rows: 4,
                        num_deleted_rows: usize::MAX,
                    }],
                    new_frags: vec![],
                }],
            }],
        };

        let index = open_frag_reuse_index(Uuid::new_v4(), &details)
            .await
            .expect("a corrupt deleted-row count must not abort the caller");
        assert_eq!(
            index.row_id_maps.len(),
            1,
            "the version should still produce one (empty) row id map"
        );
    }

    /// Malformed external metadata must surface as a normal Error instead of
    /// panicking or building an invalid object-store range. Each case crafts a
    /// different bad offset/size against a valid PAYLOAD_LEN-byte details file
    /// and asserts both the error variant and an identifying message fragment.
    #[rstest]
    #[case::out_of_bounds(0, PAYLOAD_LEN as u64 + 1, "out of bounds")]
    #[case::offset_past_eof(PAYLOAD_LEN as u64 + 100, 1, "out of bounds")]
    #[case::overflow(u64::MAX, u64::MAX, "overflow")]
    // A zero size would decode as a structurally valid but empty index, which is
    // indistinguishable from a fully-cleaned-up one, so it must be rejected outright.
    #[case::zero_size(0, 0, "size is 0")]
    #[tokio::test]
    async fn test_load_external_details_malformed_offset_size_errors(
        #[case] offset: u64,
        #[case] size: u64,
        #[case] expected_msg: &str,
    ) {
        let dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(10))
            .await
            .unwrap();

        // Write a small, valid-looking external details file.
        let uuid = Uuid::new_v4();
        let file_path = dataset
            .indices_dir()
            .join(uuid.to_string())
            .join(FRAG_REUSE_DETAILS_FILE_NAME);
        let payload = vec![0u8; PAYLOAD_LEN];
        let mut writer = dataset.object_store.create(&file_path).await.unwrap();
        writer.write_all(&payload).await.unwrap();
        writer.shutdown().await.unwrap();

        let meta = external_details_meta(uuid, offset, size);
        let err = load_frag_reuse_index_details(&dataset, &meta)
            .await
            .expect_err("malformed external metadata must surface as an error, not a panic");
        assert!(
            matches!(err, Error::Index { .. }),
            "expected an Error::Index, got: {err:?}"
        );
        assert!(
            err.to_string().to_lowercase().contains(expected_msg),
            "expected error message to contain {expected_msg:?}, got: {err}"
        );
    }
}
