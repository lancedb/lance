// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use crate::IndexType;
use lance_table::format::IndexMetadata;
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

/// A plan for building one physical segment from one or more existing
/// vector index segments.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexSegmentPlan {
    segment: IndexSegment,
    segments: Vec<IndexMetadata>,
    estimated_bytes: u64,
    requested_index_type: Option<IndexType>,
}

impl IndexSegmentPlan {
    /// Create a plan for one built segment.
    pub fn new(
        segment: IndexSegment,
        segments: Vec<IndexMetadata>,
        estimated_bytes: u64,
        requested_index_type: Option<IndexType>,
    ) -> Self {
        Self {
            segment,
            segments,
            estimated_bytes,
            requested_index_type,
        }
    }

    /// Return the segment metadata that should be committed after this plan is built.
    pub fn segment(&self) -> &IndexSegment {
        &self.segment
    }

    /// Return the input segment metadata that should be combined into the segment.
    pub fn segments(&self) -> &[IndexMetadata] {
        &self.segments
    }

    /// Return the estimated number of bytes covered by this plan.
    pub fn estimated_bytes(&self) -> u64 {
        self.estimated_bytes
    }

    /// Return the requested logical index type, if one was supplied to the planner.
    pub fn requested_index_type(&self) -> Option<IndexType> {
        self.requested_index_type
    }
}
