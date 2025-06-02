// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use crate::{dataset::optimize::remapping::transpose_row_ids, index::scalar::infer_index_type};
use lance_core::Error;
use lance_index::frag_reuse::{
    FragReuseIndex, FragReuseIndexDetails, FragReuseVersion, FRAG_REUSE_DETAILS_FILE_NAME,
    FRAG_REUSE_INDEX_NAME,
};
use lance_index::DatasetIndexExt;
use lance_table::format::pb::fragment_reuse_index_details::{Content, InlineContent};
use lance_table::format::pb::{ExternalFile, FragmentReuseIndexDetails};
use lance_table::format::{Fragment, Index};
use prost::Message;
use roaring::{RoaringBitmap, RoaringTreemap};
use snafu::location;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

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
                .base
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
pub async fn open_frag_reuse_index(
    details: &FragReuseIndexDetails,
    dataset_fragments: &[Fragment],
) -> lance_core::Result<Arc<FragReuseIndex>> {
    let mut row_id_maps: Vec<HashMap<u64, Option<u64>>> =
        Vec::with_capacity(details.versions.len());
    for version in &details.versions {
        let mut new_fragments_in_version = Vec::with_capacity(version.new_frags.len());
        dataset_fragments
            .iter()
            .filter(|f| version.new_frags.contains(&f.id))
            .for_each(|f| new_fragments_in_version.push(f.clone()));

        let cursor = Cursor::new(&version.changed_row_addrs);
        let changed_row_addrs = RoaringTreemap::deserialize_from(cursor).unwrap();

        let row_id_map = transpose_row_ids(
            changed_row_addrs,
            &version.old_frags,
            new_fragments_in_version.as_slice(),
        );
        row_id_maps.push(row_id_map);
    }

    Ok(Arc::new(FragReuseIndex::new(row_id_maps, details.clone())))
}

pub async fn build_new_frag_reuse_index(
    dataset: &mut Dataset,
    old_fragments: Vec<Fragment>,
    new_fragment_ids: Vec<u64>,
    new_fragment_bitmap: RoaringBitmap,
    changed_row_addrs: RoaringTreemap,
) -> lance_core::Result<Index> {
    let index_id = uuid::Uuid::new_v4();
    let mut serialized = Vec::with_capacity(changed_row_addrs.serialized_size());
    changed_row_addrs.serialize_into(&mut serialized)?;

    let new_version = FragReuseVersion {
        dataset_version: dataset.manifest.version,
        old_frags: old_fragments.clone(),
        new_frags: new_fragment_ids,
        changed_row_addrs: serialized,
    };

    let index_meta = dataset.load_indices().await.map(|indices| {
        indices
            .iter()
            .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
            .cloned()
    })?;

    let new_index_details = match index_meta {
        None => FragReuseIndexDetails {
            versions: Vec::from([new_version]),
        },
        Some(index_meta) => {
            let current_details = load_frag_reuse_index_details(dataset, &index_meta).await?;
            let mut versions = current_details.versions.clone();
            versions.push(new_version);
            FragReuseIndexDetails { versions }
        }
    };

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
        let external_file = ExternalFile {
            path: file_path.to_string(),
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
        index_details: Some(prost_types::Any::from_msg(&proto)?),
        index_version: 0,
    })
}
