// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Metadata for index

use deepsize::DeepSizeOf;
use roaring::RoaringBitmap;
use snafu::location;
use uuid::Uuid;

use super::pb;
use lance_core::{Error, Result};

pub const INIT_INDEX_VERSION: semver::Version = semver::Version::new(0, 27, 0);

/// Index metadata
#[derive(Debug, Clone)]
pub struct Index {
    /// Unique ID across all dataset versions.
    pub uuid: Uuid,

    /// Fields to build the index.
    pub fields: Vec<i32>,

    /// Human readable index name
    pub name: String,

    /// The latest version of the dataset this index covers
    pub dataset_version: u64,

    /// The fragment ids this index covers.
    ///
    /// If this is None, then this is unknown.
    pub fragment_bitmap: Option<RoaringBitmap>,

    /// Metadata specific to the index type
    ///
    /// This is an Option because older versions of Lance may not have this defined.  However, it should always
    /// be present in newer versions.
    pub index_details: Option<prost_types::Any>,

    /// The lance version which introduced this index format,
    /// bump this when the index format changes.
    /// This is used to determine if the index can be read by the current version of lance.
    /// if the lance version is less than the index version, then the index cannot be read.
    pub index_version: semver::Version,
}

impl DeepSizeOf for Index {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.uuid.as_bytes().deep_size_of_children(context)
            + self.fields.deep_size_of_children(context)
            + self.name.deep_size_of_children(context)
            + self.dataset_version.deep_size_of_children(context)
            + self
                .fragment_bitmap
                .as_ref()
                .map(|fragment_bitmap| fragment_bitmap.serialized_size())
                .unwrap_or(0)
    }
}

impl TryFrom<pb::IndexMetadata> for Index {
    type Error = Error;

    fn try_from(proto: pb::IndexMetadata) -> Result<Self> {
        let fragment_bitmap = if proto.fragment_bitmap.is_empty() {
            None
        } else {
            Some(RoaringBitmap::deserialize_from(
                &mut proto.fragment_bitmap.as_slice(),
            )?)
        };

        Ok(Self {
            uuid: proto.uuid.as_ref().map(Uuid::try_from).ok_or_else(|| {
                Error::io(
                    "uuid field does not exist in Index metadata".to_string(),
                    location!(),
                )
            })??,
            name: proto.name,
            fields: proto.fields,
            dataset_version: proto.dataset_version,
            fragment_bitmap,
            index_details: proto.index_details,
            index_version: proto
                .index_version
                .map(|v| semver::Version::parse(&v))
                .transpose()
                .map_err(|e| Error::Index {
                    message: format!("failed to parse the index version: {}", e),
                    location: location!(),
                })?
                .unwrap_or(INIT_INDEX_VERSION),
        })
    }
}

impl From<&Index> for pb::IndexMetadata {
    fn from(idx: &Index) -> Self {
        let mut fragment_bitmap = Vec::new();
        if let Some(bitmap) = &idx.fragment_bitmap {
            if let Err(e) = bitmap.serialize_into(&mut fragment_bitmap) {
                // In theory, this should never error. But if we do, just
                // recover gracefully.
                log::error!("Failed to serialize fragment bitmap: {}", e);
                fragment_bitmap.clear();
            }
        }

        Self {
            uuid: Some((&idx.uuid).into()),
            name: idx.name.clone(),
            fields: idx.fields.clone(),
            dataset_version: idx.dataset_version,
            fragment_bitmap,
            index_details: idx.index_details.clone(),
            index_version: Some(idx.index_version.to_string()),
        }
    }
}
