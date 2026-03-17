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
    pub uuid: Uuid,
    /// The fragments covered by this segment.
    pub fragment_bitmap: RoaringBitmap,
    /// Metadata specific to the index type.
    pub index_details: Option<Arc<prost_types::Any>>,
    /// The on-disk index version for this segment.
    pub index_version: i32,
}
