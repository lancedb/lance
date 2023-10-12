// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node
///
/// Modified from diskann paper. The vector store is backed by the `lance` dataset.
mod builder;
pub(crate) mod row_vertex;
mod search;

use super::{
    graph::{Vertex, VertexSerDe},
    MetricType,
};
pub(crate) use builder::build_diskann_index;
use lance_index::vector::pq::PQBuildParams;
pub(crate) use row_vertex::RowVertex;
pub(crate) use search::DiskANNIndex;

#[derive(Clone, Debug)]
pub struct DiskANNParams {
    /// out-degree bound (R)
    pub r: usize,

    /// Distance threshold
    pub alpha: f32,

    /// Search list size
    pub l: usize,

    /// Parameters to build PQ index.
    pub pq_params: PQBuildParams,

    /// Metric type.
    pub metric_type: MetricType,
}

// Default values from DiskANN paper.
impl Default for DiskANNParams {
    fn default() -> Self {
        Self {
            r: 50,
            alpha: 1.2,
            l: 70,
            pq_params: PQBuildParams::default(),
            metric_type: MetricType::L2,
        }
    }
}

impl DiskANNParams {
    pub fn new(r: usize, alpha: f32, l: usize) -> Self {
        Self {
            r,
            alpha,
            l,
            pq_params: PQBuildParams::default(),
            metric_type: MetricType::L2,
        }
    }

    pub fn r(&mut self, r: usize) -> &mut Self {
        self.r = r;
        self
    }

    pub fn alpha(&mut self, alpha: f32) -> &mut Self {
        self.alpha = alpha;
        self
    }

    pub fn l(&mut self, l: usize) -> &mut Self {
        self.l = l;
        self
    }

    /// Set the number of bits for PQ.
    pub fn pq_num_bits(&mut self, nbits: usize) -> &mut Self {
        self.pq_params.num_bits = nbits;
        self
    }

    pub fn pq_num_sub_vectors(&mut self, m: usize) -> &mut Self {
        self.pq_params.num_sub_vectors = m;
        self
    }

    pub fn use_opq(&mut self, use_opq: bool) -> &mut Self {
        self.pq_params.use_opq = use_opq;
        self
    }

    pub fn metric_type(&mut self, metric_type: MetricType) -> &mut Self {
        self.metric_type = metric_type;
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_testing::datagen::generate_random_array;
    use tempfile::tempdir;

    use super::*;
    use crate::{
        arrow::*,
        dataset::Dataset,
        index::{
            DatasetIndexExt,
            {vector::VectorIndexParams, IndexType},
        },
    };

    #[tokio::test]
    async fn test_create_index() {
        let test_dir = tempdir().unwrap();

        let dimension = 16;
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "embeddings",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimension,
            ),
            false,
        )]));

        let float_arr = generate_random_array(512 * dimension as usize);
        let vectors =
            Arc::new(FixedSizeListArray::try_new_from_values(float_arr, dimension).unwrap());
        let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors.clone()]).unwrap()];

        let test_uri = test_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Make sure valid arguments should create index successfully
        let params =
            VectorIndexParams::with_diskann_params(MetricType::L2, DiskANNParams::default());
        dataset
            .create_index(&["embeddings"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        // Check the version is set correctly
        let indices = dataset.load_indices().await.unwrap();
        let actual = indices.first().unwrap().dataset_version;
        let expected = dataset.manifest.version - 1;
        assert_eq!(actual, expected);
    }
}
