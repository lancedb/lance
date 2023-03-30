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

//! Vamana Graph, described in DiskANN (NeurIPS' 19) and its following papers.

use std::sync::Arc;

use arrow::array::as_primitive_array;
use arrow::datatypes::UInt64Type;
use arrow_array::UInt64Array;
use futures::{stream, StreamExt, TryStreamExt};

use super::graph::Vertex;
use crate::arrow::*;
use crate::dataset::{Dataset, ROW_ID};
use crate::{Error, Result};

struct VemanaData {
    row_id: u64,
}

pub struct VamanaBuilder {
    dataset: Arc<Dataset>,

    vertices: Vec<Vertex<VemanaData>>,
}

impl VamanaBuilder {
    /// Randomly initialize the graph.
    ///
    /// Parameters
    /// ----------
    ///  - dataset: the dataset to index
    ///  - r: the number of neighbors to connect to
    ///
    async fn try_init(dataset: Arc<Dataset>, r: usize) -> Result<Self> {
        let total = dataset.count_rows().await?;
        let scanner = dataset
            .scan()
            .with_row_id()
            .try_into_stream()
            .await
            .unwrap();

        let batches = scanner.try_collect::<Vec<_>>().await?;
        let mut vertices = Vec::new();
        let mut vectex_id = 0;
        for batch in batches {
            let row_id = as_primitive_array::<UInt64Type>(
                batch
                    .column_by_qualified_name(ROW_ID)
                    .ok_or(Error::Index("row_id not found".to_string()))?,
            );
            for i in 0..row_id.len() {
                vertices.push(Vertex {
                    id: vectex_id,
                    neighbors: Vec::new(),
                    auxilary: VemanaData {
                        row_id: row_id.value(i),
                    },
                });
                vectex_id += 1;
            }
        }

        Ok(Self {
            dataset,
            vertices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchReader};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use tempfile;

    use crate::arrow::*;
    use crate::dataset::WriteParams;
    use crate::utils::testing::generate_random_array;

    async fn create_dataset(uri: &str, n: usize, dim: usize) -> Arc<Dataset> {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Box::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            true,
        )]));
        let data = generate_random_array(n * dim);
        let batches = RecordBatchBuffer::new(vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(
                FixedSizeListArray::try_new(&data, dim as i32).unwrap(),
            )],
        )
        .unwrap()]);

        let mut write_params = WriteParams::default();
        write_params.max_rows_per_file = 40;
        write_params.max_rows_per_group = 10;
        let mut batches: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut batches, uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(uri).await.unwrap();
        Arc::new(dataset)
    }

    #[tokio::test]
    async fn test_init() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();
        let dataset = create_dataset(uri, 200, 64).await;

        let inited_graph = VamanaBuilder::try_init(dataset, 10).await.unwrap();
    }
}
