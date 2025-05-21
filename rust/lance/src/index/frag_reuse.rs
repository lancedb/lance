use crate::dataset::optimize::remapping::transpose_row_ids;
use crate::Dataset;
use lance_core::Error;
use lance_index::frag_reuse::{FragReuseIndex, FragReuseIndexDetails};
use lance_table::format::pb::fragment_reuse_index_details::{Content, InlineContent};
use lance_table::format::pb::FragmentReuseIndexDetails;
use lance_table::format::{Fragment, Index};
use prost::Message;
use roaring::RoaringTreemap;
use snafu::location;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

/// Load fragment reuse index details from index metadata
pub async fn load_index_details(
    dataset: &Dataset,
    index: &Index,
) -> lance_core::Result<Arc<FragReuseIndexDetails>> {
    let details_any = index.index_details.clone();
    if details_any.is_none()
        || !details_any
            .as_ref()
            .unwrap()
            .type_url
            .ends_with("FragReuseIndexDetails")
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

            dataset
                .session
                .file_metadata_cache
                .get_or_insert(&file_path, |_path| async {
                    let range = external_file.offset as usize
                        ..(external_file.offset as usize + external_file.size as usize);
                    let data = dataset
                        .object_store
                        .open(&file_path)
                        .await?
                        .get_range(range)
                        .await?;

                    let pb_sequence = InlineContent::decode(data)?;
                    FragReuseIndexDetails::try_from(pb_sequence)
                })
                .await
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
