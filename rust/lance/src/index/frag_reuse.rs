// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::optimize::remapping::transpose_row_ids_from_digest;
use crate::Dataset;
use lance_core::Error;
use lance_index::frag_reuse::{
    FragReuseGroup, FragReuseIndex, FragReuseIndexDetails, FragReuseVersion,
    FRAG_REUSE_DETAILS_FILE_NAME, FRAG_REUSE_INDEX_NAME,
};
use lance_index::DatasetIndexExt;
use lance_table::format::pb::fragment_reuse_index_details::{Content, InlineContent};
use lance_table::format::pb::{ExternalFile, FragmentReuseIndexDetails};
use lance_table::format::Index;
use prost::Message;
use roaring::{RoaringBitmap, RoaringTreemap};
use snafu::location;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

/// Load fragment reuse index details from index metadata
pub async fn load_frag_reuse_index_details(
    dataset: &Dataset,
    index: &Index,
) -> lance_core::Result<Arc<FragReuseIndexDetails>> {
    let details_any = index.index_details.clone();
    if details_any.is_none()
        || !details_any
            .as_ref()
            .unwrap()
            .type_url
            .ends_with("FragmentReuseIndexDetails")
    {
        return Err(Error::Index {
            message: "Index details is not for the fragment reuse index".into(),
            location: location!(),
        });
    }

    let proto = details_any.unwrap().to_msg::<FragmentReuseIndexDetails>()?;
    match &proto.content {
        None => Err(Error::Index {
            message: "Index details content is not found".into(),
            location: location!(),
        }),
        Some(Content::Inline(content)) => {
            Ok(Arc::new(FragReuseIndexDetails::try_from(content.clone())?))
        }
        Some(Content::External(external_file)) => {
            let file_path = dataset
                .indices_dir()
                .child(index.uuid.to_string())
                .child(external_file.path.clone());

            // the file content will be cached in the index cache later
            // so we do not put it to the file cache
            let range = external_file.offset as usize
                ..(external_file.offset as usize + external_file.size as usize);
            let data = dataset
                .object_store
                .open(&file_path)
                .await?
                .get_range(range)
                .await?;

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
            let changed_row_addrs = RoaringTreemap::deserialize_from(cursor).unwrap();
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
) -> lance_core::Result<Index> {
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
        Some(ref index_meta) => {
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
    index_meta: Option<&Index>,
    new_index_details: FragReuseIndexDetails,
    new_fragment_bitmap: RoaringBitmap,
) -> lance_core::Result<Index> {
    let index_id = uuid::Uuid::new_v4();
    let new_index_details_proto = InlineContent::from(&new_index_details);
    let proto = if new_index_details_proto.encoded_len() > 204800 {
        let file_path = dataset
            .indices_dir()
            .child(index_id.to_string())
            .child(FRAG_REUSE_DETAILS_FILE_NAME);
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

    Ok(Index {
        uuid: index_id,
        name: FRAG_REUSE_INDEX_NAME.to_string(),
        fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(new_fragment_bitmap),
        index_details: Some(Arc::new(prost_types::Any::from_msg(&proto)?)),
        index_version: index_meta.map_or(0, |index_meta| index_meta.index_version),
        created_at: Some(chrono::Utc::now()),
        base_id: None,
    })
}
