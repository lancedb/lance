// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Metadata for index

use std::sync::Arc;

use chrono::{DateTime, Utc};
use deepsize::DeepSizeOf;
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::pb;
use lance_core::{Error, Result};

/// Index metadata
#[derive(Debug, Clone, PartialEq)]
pub struct IndexMetadata {
    /// Unique ID across all dataset versions.
    pub uuid: Uuid,

    /// Fields to build the index.
    pub fields: Vec<i32>,

    /// Human readable index name
    pub name: String,

    /// The version of the dataset this index was last updated on
    ///
    /// This is set when the index is created (based on the version used to train the index)
    /// This is updated when the index is updated or remapped
    pub dataset_version: u64,

    /// The fragment ids this index covers.
    ///
    /// This may contain fragment ids that no longer exist in the dataset.
    ///
    /// If this is None, then this is unknown.
    pub fragment_bitmap: Option<RoaringBitmap>,

    /// Metadata specific to the index type
    ///
    /// This is an Option because older versions of Lance may not have this defined.  However, it should always
    /// be present in newer versions.
    pub index_details: Option<Arc<prost_types::Any>>,

    /// The index version.
    pub index_version: i32,

    /// Timestamp when the index was created
    ///
    /// This field is optional for backward compatibility. For existing indices created before
    /// this field was added, this will be None.
    pub created_at: Option<DateTime<Utc>>,

    /// The base path index of the index files. Used when the index is imported or referred from another dataset.
    /// Lance uses it as key of the base_paths field in Manifest to determine the actual base path of the index files.
    pub base_id: Option<u32>,

    /// The lifecycle state of this index segment.
    pub segment_lifecycle: IndexSegmentLifecycle,
}

impl IndexMetadata {
    pub fn effective_fragment_bitmap(
        &self,
        existing_fragments: &RoaringBitmap,
    ) -> Option<RoaringBitmap> {
        let fragment_bitmap = self.fragment_bitmap.as_ref()?;
        Some(fragment_bitmap & existing_fragments)
    }
}

impl DeepSizeOf for IndexMetadata {
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

impl TryFrom<pb::IndexMetadata> for IndexMetadata {
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
                Error::invalid_input("uuid field does not exist in Index metadata".to_string())
            })??,
            name: proto.name,
            fields: proto.fields,
            dataset_version: proto.dataset_version,
            fragment_bitmap,
            index_details: proto.index_details.map(Arc::new),
            index_version: proto.index_version.unwrap_or_default(),
            created_at: proto.created_at.map(|ts| {
                DateTime::from_timestamp_millis(ts as i64)
                    .expect("Invalid timestamp in index metadata")
            }),
            base_id: proto.base_id,
            segment_lifecycle: IndexSegmentLifecycle::from_proto(proto.segment_lifecycle),
        })
    }
}

impl From<&IndexMetadata> for pb::IndexMetadata {
    fn from(idx: &IndexMetadata) -> Self {
        let mut fragment_bitmap = Vec::new();
        if let Some(bitmap) = &idx.fragment_bitmap
            && let Err(e) = bitmap.serialize_into(&mut fragment_bitmap)
        {
            // In theory, this should never error. But if we do, just
            // recover gracefully.
            log::error!("Failed to serialize fragment bitmap: {}", e);
            fragment_bitmap.clear();
        }

        Self {
            uuid: Some((&idx.uuid).into()),
            name: idx.name.clone(),
            fields: idx.fields.clone(),
            dataset_version: idx.dataset_version,
            fragment_bitmap,
            index_details: idx
                .index_details
                .as_ref()
                .map(|details| details.as_ref().clone()),
            index_version: Some(idx.index_version),
            created_at: idx.created_at.map(|dt| dt.timestamp_millis() as u64),
            base_id: idx.base_id,
            segment_lifecycle: Some(idx.segment_lifecycle.to_proto()),
        }
    }
}

/// Lifecycle state for a committed index segment.
///
/// This captures whether a segment is still expected to absorb future index
/// updates (`Active`) or has been finalized into a stable segment (`Sealed`).
///
/// This type intentionally models lifecycle only. It does not encode query
/// roles such as whether a segment is currently searchable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexSegmentLifecycle {
    /// Segment is still mutable from the maintenance workflow's point of view.
    Active,
    /// Segment is stable and should no longer absorb future updates.
    #[default]
    Sealed,
}

impl IndexSegmentLifecycle {
    fn from_proto(segment_lifecycle: Option<i32>) -> Self {
        match segment_lifecycle.and_then(|value| pb::IndexSegmentLifecycle::try_from(value).ok()) {
            Some(pb::IndexSegmentLifecycle::Active) => Self::Active,
            Some(pb::IndexSegmentLifecycle::Sealed)
            | Some(pb::IndexSegmentLifecycle::Unspecified)
            | None => Self::Sealed,
        }
    }

    fn to_proto(self) -> i32 {
        match self {
            Self::Active => pb::IndexSegmentLifecycle::Active as i32,
            Self::Sealed => pb::IndexSegmentLifecycle::Sealed as i32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_uuid() -> Uuid {
        Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap()
    }

    #[test]
    fn test_index_metadata_defaults_missing_segment_lifecycle_to_sealed() {
        let proto = pb::IndexMetadata {
            uuid: Some((&test_uuid()).into()),
            name: "idx".to_string(),
            fields: vec![1],
            dataset_version: 7,
            fragment_bitmap: vec![],
            index_details: None,
            index_version: Some(3),
            created_at: None,
            base_id: None,
            segment_lifecycle: None,
        };

        let metadata = IndexMetadata::try_from(proto).unwrap();

        assert_eq!(metadata.segment_lifecycle, IndexSegmentLifecycle::Sealed);
    }

    #[test]
    fn test_index_metadata_roundtrip_segment_lifecycle() {
        let metadata = IndexMetadata {
            uuid: test_uuid(),
            fields: vec![1],
            name: "idx".to_string(),
            dataset_version: 7,
            fragment_bitmap: None,
            index_details: None,
            index_version: 3,
            created_at: None,
            base_id: None,
            segment_lifecycle: IndexSegmentLifecycle::Active,
        };

        let proto = pb::IndexMetadata::from(&metadata);

        assert_eq!(
            proto.segment_lifecycle,
            Some(pb::IndexSegmentLifecycle::Active as i32)
        );

        let roundtrip = IndexMetadata::try_from(proto).unwrap();
        assert_eq!(roundtrip.segment_lifecycle, IndexSegmentLifecycle::Active);
    }
}
