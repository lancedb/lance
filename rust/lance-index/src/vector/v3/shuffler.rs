// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shuffler is a component that takes a stream of record batches and shuffles them into
//! the corresponding IVF partitions.

use lance_core::Result;
use lance_io::stream::RecordBatchStream;

#[async_trait::async_trait]
/// A reader that can read the shuffled partitions.
pub trait IvfShuffleReader {
    /// Read a partition by partition_id
    /// will return error if partition_size is 0
    /// check reader.partiton_size(partition_id) before calling this function
    async fn read_partition(
        &self,
        partition_id: usize,
    ) -> Result<Box<dyn RecordBatchStream + Unpin + 'static>>;

    /// Get the size of the partition by partition_id
    fn partiton_size(&self, partition_id: usize) -> Result<usize>;
}

#[async_trait::async_trait]
/// A shuffler that can shuffle the incoming stream of record batches into IVF partitions.
/// Returns a IvfShuffleReader that can be used to read the shuffled partitions.
pub trait IvfShuffler {
    /// Shuffle the incoming stream of record batches into IVF partitions.
    /// Returns a IvfShuffleReader that can be used to read the shuffled partitions.
    async fn shuffle(
        mut self,
        data: Box<dyn RecordBatchStream + Unpin + 'static>,
    ) -> Result<Box<dyn IvfShuffleReader>>;
}

#[async_trait::async_trait]
/// A specification to build a IvfShuffler.
pub trait IvfShufflerSpec {
    async fn build(&self) -> Box<dyn IvfShuffler>;
}
