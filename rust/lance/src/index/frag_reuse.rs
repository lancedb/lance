// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use crate::dataset::optimize::remapping::transpose_row_ids_from_digest;
use crate::index::DatasetIndexExt;
use lance_core::Error;
use lance_core::utils::address::RowAddress;
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

/// Validate one rewrite group's fragment digests before they are transposed.
///
/// The digests come straight from persisted metadata with no validation, and
/// `transpose_row_ids_from_digest` assumes they describe a real row-address domain: it
/// asserts the old fragment list is non-empty, multiplies `id` by
/// `RowAddress::FRAGMENT_SIZE`, and walks `physical_rows` rows per fragment. A corrupt
/// digest therefore aborts or stalls the caller instead of producing an error, so reject
/// out-of-domain values here, at the single fallible open boundary.
///
/// This boundary is on the path of every index open, and also of
/// `cleanup_frag_reuse_index`, which reaches it indirectly: its commit reloads the
/// current index through `apply_commit` -> `commit_transaction` -> `load_indices` ->
/// `open_frag_reuse_index`.
fn validate_frag_reuse_group(
    uuid: &Uuid,
    dataset_version: u64,
    group: &FragReuseGroup,
) -> lance_core::Result<()> {
    let invalid = |msg: String| {
        Error::index(format!(
            "Invalid fragment reuse index {uuid} at dataset version {dataset_version}: {msg}"
        ))
    };

    // `MissingAddrs::new` asserts on an empty old fragment list, and a group that
    // rewrites nothing is meaningless anyway.
    if group.old_frags.is_empty() {
        return Err(invalid("rewrite group has no old fragments".to_string()));
    }

    for (role, digests) in [("old", &group.old_frags), ("new", &group.new_frags)] {
        for digest in digests.iter() {
            // Row addresses pack the fragment id into 32 bits. For old fragments the
            // iterator computes `id * FRAGMENT_SIZE`, which overflows u64 past this
            // bound; for new fragments `id as u32` would silently truncate and produce a
            // mapping onto the wrong fragment.
            if digest.id > u32::MAX as u64 {
                return Err(invalid(format!(
                    "{role} fragment id {} exceeds the maximum fragment id {}",
                    digest.id,
                    u32::MAX
                )));
            }
            // Must be `>=`, not `>`: at exactly FRAGMENT_SIZE the fragment-advance
            // condition `expected_row_addr % FRAGMENT_SIZE == physical_rows` can never
            // hold (the left side is always smaller), so the iterator never terminates,
            // and `physical_rows as u32` truncates to 0. Bounding it below FRAGMENT_SIZE
            // also makes the `expected_row_addr` accumulation provably overflow-free:
            // with `id <= u32::MAX`, the largest address is `(2^32-1) * 2^32 + (2^32-1)`,
            // exactly u64::MAX.
            if digest.physical_rows as u64 >= RowAddress::FRAGMENT_SIZE {
                return Err(invalid(format!(
                    "{role} fragment {} declares {} physical rows, which is not below the maximum {} per fragment",
                    digest.id,
                    digest.physical_rows,
                    RowAddress::FRAGMENT_SIZE
                )));
            }
            if digest.num_deleted_rows > digest.physical_rows {
                return Err(invalid(format!(
                    "{role} fragment {} declares {} deleted rows, more than its {} physical rows",
                    digest.id, digest.num_deleted_rows, digest.physical_rows
                )));
            }
        }
    }

    // A zero-row old fragment never satisfies the iterator's fragment-advance condition,
    // so it would spin for FRAGMENT_SIZE iterations instead of terminating.
    for digest in group.old_frags.iter() {
        if digest.physical_rows == 0 {
            return Err(invalid(format!(
                "old fragment {} declares zero physical rows",
                digest.id
            )));
        }
    }

    Ok(())
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
            validate_frag_reuse_group(&uuid, version.dataset_version, group)?;
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
    use lance_table::system_index::frag_reuse::FragDigest;
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

    fn digest(id: u64, physical_rows: usize, num_deleted_rows: usize) -> FragDigest {
        FragDigest {
            id,
            physical_rows,
            num_deleted_rows,
        }
    }

    fn details_with_group(
        old_frags: Vec<FragDigest>,
        new_frags: Vec<FragDigest>,
    ) -> FragReuseIndexDetails {
        let mut changed_row_addrs = Vec::new();
        RoaringTreemap::new()
            .serialize_into(&mut changed_row_addrs)
            .unwrap();
        FragReuseIndexDetails {
            versions: vec![FragReuseVersion {
                dataset_version: 7,
                groups: vec![FragReuseGroup {
                    changed_row_addrs,
                    old_frags,
                    new_frags,
                }],
            }],
        }
    }

    /// Fragment digests are read verbatim from persisted metadata, but
    /// `transpose_row_ids_from_digest` assumes they describe a real row-address domain.
    /// Each of these values used to abort or stall the caller (an assert on empty old
    /// fragments, a `u64` overflow on `id * FRAGMENT_SIZE`, a ~2^32 iteration spin on a
    /// zero-row fragment, a capacity-overflow panic on a corrupt deleted-row count), so
    /// the open boundary must reject them with a contextual error instead.
    #[rstest]
    #[case::no_old_fragments(vec![], vec![digest(1, 10, 0)], "no old fragments")]
    #[case::old_fragment_id_overflow(
        vec![digest(u32::MAX as u64 + 1, 10, 0)],
        vec![],
        "exceeds the maximum fragment id"
    )]
    #[case::new_fragment_id_overflow(
        vec![digest(0, 10, 0)],
        vec![digest(u32::MAX as u64 + 1, 10, 0)],
        "exceeds the maximum fragment id"
    )]
    #[case::zero_physical_rows(vec![digest(0, 0, 0)], vec![], "zero physical rows")]
    // Exactly FRAGMENT_SIZE is the dangerous boundary: it used to slip past a `>` check
    // and then spin forever in the fragment-advance condition.
    #[case::physical_rows_at_fragment_size(
        vec![digest(0, RowAddress::FRAGMENT_SIZE as usize, 0)],
        vec![],
        "not below the maximum"
    )]
    #[case::physical_rows_above_fragment_size(
        vec![digest(0, RowAddress::FRAGMENT_SIZE as usize + 1, 0)],
        vec![],
        "not below the maximum"
    )]
    #[case::deleted_exceeds_physical(
        vec![digest(0, 4, usize::MAX)],
        vec![],
        "more than its 4 physical rows"
    )]
    #[tokio::test]
    async fn test_open_frag_reuse_index_rejects_out_of_domain_digests(
        #[case] old_frags: Vec<FragDigest>,
        #[case] new_frags: Vec<FragDigest>,
        #[case] expected_msg: &str,
    ) {
        let details = details_with_group(old_frags, new_frags);
        let err = open_frag_reuse_index(Uuid::new_v4(), &details)
            .await
            .expect_err("an out-of-domain fragment digest must not abort or stall the caller");
        assert!(
            matches!(err, Error::Index { .. }),
            "expected an Error::Index, got: {err:?}"
        );
        assert!(
            err.to_string().contains(expected_msg),
            "expected error message to contain {expected_msg:?}, got: {err}"
        );
    }

    /// The counterpart: a digest that stays inside the row-address domain still opens,
    /// and actually produces a mapping rather than an empty one.
    #[tokio::test]
    async fn test_open_frag_reuse_index_accepts_valid_digests() {
        let details = details_with_group(vec![digest(0, 4, 1)], vec![digest(1, 3, 0)]);
        let index = open_frag_reuse_index(Uuid::new_v4(), &details)
            .await
            .expect("a well-formed group must open");
        assert_eq!(index.row_id_maps.len(), 1);
        assert!(
            !index.row_id_maps[0].is_empty(),
            "a group covering four old rows must map something"
        );
    }

    /// `transpose_row_ids_from_digest` is also called directly by compaction, which does
    /// not pass through the open-time validator, so the capacity clamp there remains the
    /// only guard against a corrupt deleted-row count. Keep it covered independently.
    #[test]
    fn test_transpose_row_ids_clamps_corrupt_deleted_row_count() {
        let old_frags = vec![digest(0, 4, usize::MAX)];
        let new_frags = vec![digest(1, 4, 0)];
        let mapping = transpose_row_ids_from_digest(RoaringTreemap::new(), &old_frags, &new_frags);
        assert!(
            !mapping.is_empty(),
            "a corrupt deleted-row count must not abort or stall the transposition"
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
