// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use roaring::RoaringBitmap;
use uuid::Uuid;

/// A single physical segment of a logical index.
///
/// Each segment is stored independently and will become one manifest entry when committed.
/// The logical index identity (name / target column / dataset version) is provided separately
/// by the commit API.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexSegment {
    /// Unique ID of the physical segment.
    uuid: Uuid,
    /// The fragments covered by this segment.
    fragment_bitmap: RoaringBitmap,
    /// Metadata specific to the index type.
    index_details: Arc<prost_types::Any>,
    /// The on-disk index version for this segment.
    index_version: i32,
}

impl IndexSegment {
    /// Create a fully described segment with the given UUID, fragment coverage, and index
    /// metadata.
    pub fn new<I>(
        uuid: Uuid,
        fragment_bitmap: I,
        index_details: Arc<prost_types::Any>,
        index_version: i32,
    ) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        Self {
            uuid,
            fragment_bitmap: fragment_bitmap.into_iter().collect(),
            index_details,
            index_version,
        }
    }

    /// Return the UUID of this segment.
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }

    /// Return the fragment coverage of this segment.
    pub fn fragment_bitmap(&self) -> &RoaringBitmap {
        &self.fragment_bitmap
    }

    /// Return the serialized index details for this segment.
    pub fn index_details(&self) -> &Arc<prost_types::Any> {
        &self.index_details
    }

    /// Return the on-disk index version for this segment.
    pub fn index_version(&self) -> i32 {
        self.index_version
    }

    /// Consume the segment and return its component parts.
    pub fn into_parts(self) -> (Uuid, RoaringBitmap, Arc<prost_types::Any>, i32) {
        (
            self.uuid,
            self.fragment_bitmap,
            self.index_details,
            self.index_version,
        )
    }
}
