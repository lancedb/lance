// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_schema::Schema;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_index::{metrics::NoOpMetricsCollector, vector::VectorIndex};
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use crate::{Error, Result};

/// A materialized snapshot of one logical vector index and all of its segments.
#[derive(Debug)]
pub struct LogicalVectorIndex {
    name: String,
    column: String,
    segments: Vec<(IndexMetadata, Arc<dyn VectorIndex>)>,
}

/// An IVF-specific inspection and maintenance view over a [`LogicalVectorIndex`].
///
/// Callers must explicitly opt into this view before accessing partition-level
/// APIs so that IVF semantics stay separated from the generic logical vector
/// index abstraction.
#[derive(Clone, Copy, Debug)]
pub struct LogicalIvfView<'a> {
    logical_index: &'a LogicalVectorIndex,
}

impl LogicalVectorIndex {
    pub(crate) fn try_new(
        name: String,
        column: String,
        segments: Vec<(IndexMetadata, Arc<dyn VectorIndex>)>,
    ) -> Result<Self> {
        if segments.is_empty() {
            return Err(Error::invalid_input(format!(
                "LogicalVectorIndex '{}' on column '{}' must contain at least one segment",
                name, column
            )));
        }

        Ok(Self {
            name,
            column,
            segments,
        })
    }

    /// Returns the logical index name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the vector column that this logical index is built on.
    pub fn column(&self) -> &str {
        &self.column
    }

    /// Returns the number of physical segments in this logical index.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Returns the committed metadata for all physical segments.
    pub fn metadatas(&self) -> impl ExactSizeIterator<Item = &IndexMetadata> + '_ {
        self.segments.iter().map(|(metadata, _)| metadata)
    }

    /// Returns the indexed row count for each segment.
    pub fn num_rows_per_segment(&self) -> Vec<(Uuid, u64)> {
        self.segments
            .iter()
            .map(|(metadata, index)| (metadata.uuid, index.num_rows()))
            .collect()
    }

    /// Returns an IVF view over this logical index.
    ///
    /// This is currently fallible only to leave room for future non-IVF vector
    /// index families. All vector indices currently opened through this path are
    /// IVF-backed.
    pub fn as_ivf(&self) -> Result<LogicalIvfView<'_>> {
        Ok(LogicalIvfView {
            logical_index: self,
        })
    }

    pub(crate) fn iter(
        &self,
    ) -> impl ExactSizeIterator<Item = (&IndexMetadata, &Arc<dyn VectorIndex>)> + '_ {
        self.segments
            .iter()
            .map(|(metadata, index)| (metadata, index))
    }
}

impl<'a> LogicalIvfView<'a> {
    pub(crate) fn indices(&self) -> impl ExactSizeIterator<Item = &Arc<dyn VectorIndex>> + '_ {
        self.logical_index.iter().map(|(_, index)| index)
    }

    /// Returns the partition count for each segment in this IVF index.
    pub fn num_partitions_per_segment(&self) -> Vec<(Uuid, usize)> {
        self.logical_index
            .iter()
            .map(|(metadata, index)| (metadata.uuid, index.ivf_model().num_partitions()))
            .collect()
    }

    /// Returns the partition sizes for each segment in this IVF index.
    pub fn partition_sizes(&self) -> Vec<(Uuid, Vec<usize>)> {
        let mut partition_sizes = Vec::with_capacity(self.logical_index.num_segments());
        for (metadata, index) in self.logical_index.iter() {
            let num_partitions = index.ivf_model().num_partitions();
            let mut sizes = Vec::with_capacity(num_partitions);
            for partition_id in 0..num_partitions {
                sizes.push(index.partition_size(partition_id));
            }
            partition_sizes.push((metadata.uuid, sizes));
        }
        partition_sizes
    }

    /// Reads one IVF partition across all segments in this logical index.
    ///
    /// The returned stream preserves segment boundaries only implicitly through
    /// concatenation order; callers should treat it as a merged partition view.
    pub async fn read_partition(
        &self,
        partition_id: usize,
        with_vector: bool,
    ) -> Result<SendableRecordBatchStream> {
        let mut schema: Option<Arc<Schema>> = None;
        let mut partition_streams = Vec::with_capacity(self.logical_index.num_segments());
        for index in self.indices() {
            let stream = index
                .partition_reader(partition_id, with_vector, &NoOpMetricsCollector)
                .await?;
            if schema.is_none() {
                schema = Some(stream.schema());
            }
            partition_streams.push(stream);
        }

        match schema {
            Some(schema) => {
                let merged = stream::select_all(partition_streams);
                let stream = RecordBatchStreamAdapter::new(schema, merged);
                Ok(Box::pin(stream))
            }
            None => Ok(Box::pin(RecordBatchStreamAdapter::new(
                Arc::new(Schema::empty()),
                stream::empty(),
            ))),
        }
    }
}
