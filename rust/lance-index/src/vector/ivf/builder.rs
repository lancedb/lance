// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Build IVF model

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, FixedSizeListArray, UInt32Array, UInt64Array};
use futures::TryStreamExt;
use object_store::path::Path;
use snafu::{location, Location};

use lance_core::error::{Error, Result};
use lance_io::stream::RecordBatchStream;

/// Parameters to build IVF partitions
#[derive(Debug, Clone)]
pub struct IvfBuildParams {
    /// Number of partitions to build.
    pub num_partitions: usize,

    // ---- kmeans parameters
    /// Max number of iterations to train kmeans.
    pub max_iters: usize,

    /// Use provided IVF centroids.
    pub centroids: Option<Arc<FixedSizeListArray>>,

    pub sample_rate: usize,

    /// Precomputed partitions file (row_id -> partition_id)
    /// mutually exclusive with `precomputed_shuffle_buffers`
    pub precomputed_partitons_file: Option<String>,

    /// Precomputed shuffle buffers (row_id -> partition_id, pq_code)
    /// mutually exclusive with `precomputed_partitons_file`
    /// requires `centroids` to be set
    ///
    /// The input is expected to be (/dir/to/buffers, [buffer1.lance, buffer2.lance, ...])
    pub precomputed_shuffle_buffers: Option<(Path, Vec<String>)>,

    pub shuffle_partition_batches: usize,

    pub shuffle_partition_concurrency: usize,

    /// Use residual vectors to build sub-vector.
    pub use_residual: bool,

    /// Storage options used to load precomputed partitions.
    pub storage_options: Option<HashMap<String, String>>,
}

impl Default for IvfBuildParams {
    fn default() -> Self {
        Self {
            num_partitions: 32,
            max_iters: 50,
            centroids: None,
            sample_rate: 256, // See faiss
            precomputed_partitons_file: None,
            precomputed_shuffle_buffers: None,
            shuffle_partition_batches: 1024 * 10,
            shuffle_partition_concurrency: 2,
            use_residual: true,
            storage_options: None,
        }
    }
}

impl IvfBuildParams {
    /// Create a new instance of `IvfBuildParams`.
    pub fn new(num_partitions: usize) -> Self {
        Self {
            num_partitions,
            ..Default::default()
        }
    }

    /// Create a new instance of [`IvfBuildParams`] with centroids.
    pub fn try_with_centroids(
        num_partitions: usize,
        centroids: Arc<FixedSizeListArray>,
    ) -> Result<Self> {
        if num_partitions != centroids.len() {
            return Err(Error::Index {
                message: format!(
                    "IvfBuildParams::try_with_centroids: num_partitions {} != centroids.len() {}",
                    num_partitions,
                    centroids.len()
                ),
                location: location!(),
            });
        }
        Ok(Self {
            num_partitions,
            centroids: Some(centroids),
            ..Default::default()
        })
    }
}

/// Load precomputed partitions from disk.
///
/// Currently, because `Dataset` is not cleanly refactored from `lance` to `lance-core`,
/// we have to use `RecordBatchStream` as parameter.
pub async fn load_precomputed_partitions(
    stream: impl RecordBatchStream + Unpin + 'static,
    size_hint: usize,
) -> Result<HashMap<u64, u32>> {
    let partition_lookup = stream
        .try_fold(HashMap::with_capacity(size_hint), |mut lookup, batch| {
            let row_ids: &UInt64Array = batch
                .column_by_name("row_id")
                .expect("malformed partition file: missing row_id column")
                .as_primitive();
            let partitions: &UInt32Array = batch
                .column_by_name("partition")
                .expect("malformed partition file: missing partition column")
                .as_primitive();
            row_ids
                .values()
                .iter()
                .zip(partitions.values().iter())
                .for_each(|(row_id, partition)| {
                    lookup.insert(*row_id, *partition);
                });
            async move { Ok(lookup) }
        })
        .await?;

    Ok(partition_lookup)
}
