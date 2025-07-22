// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Metadata for index

use chrono::{DateTime, Utc};
use deepsize::DeepSizeOf;
use roaring::RoaringBitmap;
use snafu::location;
use uuid::Uuid;

use super::pb;
use lance_core::{Error, Result};
/// Index metadata
#[derive(Debug, Clone, PartialEq)]
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

    /// The index version.
    pub index_version: i32,

    /// Timestamp when the index was created
    ///
    /// This field is optional for backward compatibility. For existing indices created before
    /// this field was added, this will be None.
    pub created_at: Option<DateTime<Utc>>,
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
            index_version: proto.index_version.unwrap_or_default(),
            created_at: proto.created_at.map(|ts| {
                DateTime::from_timestamp_millis(ts as i64)
                    .expect("Invalid timestamp in index metadata")
            }),
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
            index_version: Some(idx.index_version),
            created_at: idx.created_at.map(|dt| dt.timestamp_millis() as u64),
        }
    }
}
