// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use lance_index::{IndexParams, IndexType, optimize::OptimizeOptions};
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::{Error, Result};

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

/// Extends [`crate::Dataset`] with secondary index APIs.
#[async_trait]
pub trait DatasetIndexExt {
    type IndexBuilder<'a>
    where
        Self: 'a;
    type IndexSegmentBuilder<'a>
    where
        Self: 'a;

    /// Create a builder for creating an index on columns.
    ///
    /// This returns a builder that can be configured with additional options
    /// like `name()`, `replace()`, and `train()` before awaiting to execute.
    fn create_index_builder<'a>(
        &'a mut self,
        columns: &'a [&'a str],
        index_type: IndexType,
        params: &'a dyn IndexParams,
    ) -> Self::IndexBuilder<'a>;

    /// Create a builder for building physical index segments from uncommitted
    /// vector index outputs.
    ///
    /// The caller supplies the uncommitted index metadata returned by
    /// `execute_uncommitted()` so the builder can plan segment grouping without
    /// rediscovering fragment coverage.
    ///
    /// This is the canonical entry point for distributed vector segment build.
    /// After building the physical segments, publish them as a
    /// logical index with [`Self::commit_existing_index_segments`].
    fn create_index_segment_builder<'a>(&'a self) -> Self::IndexSegmentBuilder<'a>;

    /// Create indices on columns.
    ///
    /// Upon finish, a new dataset version is generated.
    async fn create_index(
        &mut self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
        replace: bool,
    ) -> Result<IndexMetadata>;

    /// Drop indices by name.
    ///
    /// Upon finish, a new dataset version is generated.
    async fn drop_index(&mut self, name: &str) -> Result<()>;

    /// Prewarm an index by name.
    ///
    /// This will load the index into memory and cache it.
    async fn prewarm_index(&self, name: &str) -> Result<()>;

    /// Read all indices of this Dataset version.
    ///
    /// The indices are lazy loaded and cached in memory within the `Dataset` instance.
    /// The cache is invalidated when the dataset version (Manifest) is changed.
    async fn load_indices(&self) -> Result<Arc<Vec<IndexMetadata>>>;

    /// Loads all the indices of a given UUID.
    ///
    /// Note that it is possible to have multiple indices with the same UUID,
    /// as they are the deltas of the same index.
    async fn load_index(&self, uuid: &str) -> Result<Option<IndexMetadata>> {
        self.load_indices().await.map(|indices| {
            indices
                .iter()
                .find(|idx| idx.uuid.to_string() == uuid)
                .cloned()
        })
    }

    /// Loads a specific index with the given index name.
    ///
    /// Returns `Ok(vec![])` if the index does not exist.
    async fn load_indices_by_name(&self, name: &str) -> Result<Vec<IndexMetadata>> {
        self.load_indices().await.map(|indices| {
            indices
                .iter()
                .filter(|idx| idx.name == name)
                .cloned()
                .collect()
        })
    }

    /// Loads a specific index with the given index name.
    /// This function only works for indices that are unique.
    /// If there are multiple indices sharing the same name, please use [`Self::load_indices_by_name`].
    async fn load_index_by_name(&self, name: &str) -> Result<Option<IndexMetadata>> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            Ok(None)
        } else if indices.len() == 1 {
            Ok(Some(indices[0].clone()))
        } else {
            Err(Error::index(format!(
                "Found multiple indices of the same name: {:?}, please use load_indices_by_name",
                indices.iter().map(|idx| &idx.name).collect::<Vec<_>>()
            )))
        }
    }

    /// Describes indexes in a dataset.
    ///
    /// This method should only access the index metadata and should not load the index into memory.
    async fn describe_indices<'a, 'b>(
        &'a self,
        criteria: Option<lance_index::IndexCriteria<'b>>,
    ) -> Result<Vec<Arc<dyn lance_index::IndexDescription>>>;

    /// Loads a specific scalar index using the provided criteria.
    async fn load_scalar_index<'a, 'b>(
        &'a self,
        criteria: lance_index::IndexCriteria<'b>,
    ) -> Result<Option<IndexMetadata>>;

    /// Optimize indices.
    async fn optimize_indices(&mut self, options: &OptimizeOptions) -> Result<()>;

    /// Find an index with the given name and return its serialized statistics.
    async fn index_statistics(&self, index_name: &str) -> Result<String>;

    /// Commit one or more existing physical index segments as a logical index.
    async fn commit_existing_index_segments(
        &mut self,
        index_name: &str,
        column: &str,
        segments: Vec<IndexSegment>,
    ) -> Result<()>;

    async fn read_index_partition(
        &self,
        index_name: &str,
        partition_id: usize,
        with_vector: bool,
    ) -> Result<SendableRecordBatchStream>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{IndexSegment, IndexSegmentPlan};
    use lance_index::IndexType;
    use uuid::Uuid;

    #[test]
    fn test_index_segment_plan_accessors() {
        let uuid = Uuid::new_v4();
        let segment = IndexSegment::new(uuid, [1_u32, 3], Arc::new(prost_types::Any::default()), 7);
        let plan = IndexSegmentPlan::new(segment.clone(), vec![], 128, Some(IndexType::BTree));

        assert_eq!(segment.uuid(), uuid);
        assert_eq!(
            segment.fragment_bitmap().iter().collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(segment.index_version(), 7);
        assert_eq!(plan.segment().uuid(), uuid);
        assert_eq!(plan.estimated_bytes(), 128);
        assert_eq!(plan.requested_index_type(), Some(IndexType::BTree));
    }
}
