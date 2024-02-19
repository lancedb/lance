// Copyright 2024 Lance Developers.
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

//! HNSW graph implementation.
//!
//! Hierarchical Navigable Small World (HNSW).
//!

use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::{
    builder::{ListBuilder, UInt32Builder},
    cast::AsArray,
    ListArray, RecordBatch, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use lance_core::{Error, Result};
use lance_file::{reader::FileReader, writer::FileWriter};
use lance_linalg::distance::MetricType;
use lance_table::io::manifest::ManifestDescribing;
use serde::{Deserialize, Serialize};
use serde_json::json;
use snafu::{location, Location};

use super::graph::{
    builder::GraphBuilder, storage::VectorStorage, Graph, OrderedFloat, NEIGHBORS_COL,
    NEIGHBORS_FIELD,
};
use crate::vector::graph::beam_search;
pub mod builder;
pub use builder::HNSWBuilder;
mod storage;

const HNSW_TYPE: &str = "HNSW";
const VECTOR_ID_COL: &str = "__vector_id";
const POINTER_COL: &str = "__pointer";

lazy_static::lazy_static! {
    /// POINTER field.
    ///
    pub static ref POINTER_FIELD: Field = Field::new(POINTER_COL, DataType::UInt32, true);

    /// Id of the vector in the [VectorStorage].
    pub static ref VECTOR_ID_FIELD: Field = Field::new(VECTOR_ID_COL, DataType::UInt32, true);
}

/// One level of the HNSW graph.
///
struct HnswLevel {
    /// All the nodes in this level.
    // TODO: we just load the whole level into memory without pagation.
    nodes: RecordBatch,

    /// A reference to the neighbors of each node.
    ///
    /// Keep a reference of this array so that `neighbors()` can return reference
    /// without lifetime issue.
    neighbors: Arc<ListArray>,

    /// The values of the neighbors array.
    neighbors_values: Arc<UInt32Array>,

    /// Vector storage of the graph.
    vectors: Arc<dyn VectorStorage>,
}

impl HnswLevel {
    /// Load one Hnsw level from the file.
    async fn load(
        reader: &FileReader,
        row_range: Range<usize>,
        vectors: Arc<dyn VectorStorage>,
    ) -> Result<Self> {
        let nodes = reader.read_range(row_range, reader.schema(), None).await?;
        Ok(Self::new(nodes, vectors))
    }

    /// Create a new Hnsw Level from the nodes and the vector storage object.
    fn new(nodes: RecordBatch, vectors: Arc<dyn VectorStorage>) -> Self {
        let neighbors: Arc<ListArray> = nodes
            .column_by_name(NEIGHBORS_COL)
            .unwrap()
            .as_list()
            .clone()
            .into();
        let values: Arc<UInt32Array> = neighbors.values().as_primitive().clone().into();
        let vector_ids: Arc<UInt32Array> = nodes
            .column_by_name(VECTOR_ID_COL)
            .unwrap()
            .as_primitive()
            .clone()
            .into();
        let vectors = Arc::new(storage::HnswRemappingStorage::new(vectors, vector_ids));
        Self {
            nodes,
            neighbors,
            neighbors_values: values,
            vectors,
        }
    }

    fn from_builder(builder: &GraphBuilder, vectors: Arc<dyn VectorStorage>) -> Result<Self> {
        let mut neighbours_builder = ListBuilder::new(UInt32Builder::new());
        let mut pointers_builder = UInt32Builder::new();
        let mut vector_id_builder = UInt32Builder::new();

        for (_, node) in builder.nodes.iter() {
            neighbours_builder.append_value(node.neighbors.values().map(|&n| Some(n)));
            pointers_builder.append_value(node.pointer);
            vector_id_builder.append_value(node.id);
        }

        let schema = Schema::new(vec![
            NEIGHBORS_FIELD.clone(),
            VECTOR_ID_FIELD.clone(),
            POINTER_FIELD.clone(),
        ]);
        let batch = RecordBatch::try_new(
            schema.into(),
            vec![
                Arc::new(neighbours_builder.finish()),
                Arc::new(vector_id_builder.finish()),
                Arc::new(pointers_builder.finish()),
            ],
        )?;

        Ok(Self::new(batch, vectors))
    }

    fn schema(&self) -> SchemaRef {
        self.nodes.schema()
    }

    /// Range of neighbors for the given node, specified by its index.
    fn neighbors_range(&self, idx: u32) -> Range<usize> {
        let start = self.neighbors.value_offsets()[idx as usize] as usize;
        let end = start + self.neighbors.value_length(idx as usize) as usize;
        start..end
    }
}

impl Graph for HnswLevel {
    fn len(&self) -> usize {
        self.nodes.num_rows()
    }

    fn neighbors(&self, key: u32) -> Option<Box<dyn Iterator<Item = &u32> + '_>> {
        let range = self.neighbors_range(key);
        Some(Box::new(self.neighbors_values.values()[range].iter()))
    }

    fn storage(&self) -> Arc<dyn VectorStorage> {
        self.vectors.clone()
    }
}

/// HNSW graph.
///
pub struct HNSW {
    levels: Vec<HnswLevel>,
    metric_type: MetricType,
    /// Entry point of the graph.
    entry_point: u32,
}

impl Debug for HNSW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HNSW(max_layers: {}, metric={})",
            self.levels.len(),
            self.metric_type
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SchemaIndexMetadata {
    #[serde(rename = "type")]
    type_: String,

    metric_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct HnswMetadata {
    entry_point: u32,
    level_offsets: Vec<usize>,
}

impl HNSW {
    /// Load the HNSW graph from a [FileReader].
    ///
    /// Parameters
    /// ----------
    /// reader : &FileReader
    /// vector_storage : Arc<dyn VectorStorage>
    pub async fn load(reader: &FileReader, vector_storage: Arc<dyn VectorStorage>) -> Result<Self> {
        let schema = reader.schema();
        let mt = if let Some(index_metadata) = schema.metadata.get("lance:index") {
            let index_metadata: SchemaIndexMetadata = serde_json::from_str(index_metadata)?;
            if index_metadata.type_ != "HNSW" {
                return Err(Error::Index {
                    message: "index type is not HNSW".to_string(),
                    location: location!(),
                });
            }
            MetricType::try_from(index_metadata.metric_type.as_str())?
        } else {
            return Err(Error::Index {
                message: "index metadata not found in the schema".to_string(),
                location: location!(),
            });
        };
        let hnsw_metadata: HnswMetadata =
            serde_json::from_str(schema.metadata.get("lance:hnsw").ok_or_else(|| {
                Error::Index {
                    message: "hnsw metadata not found in the schema".to_string(),
                    location: location!(),
                }
            })?)?;

        let mut levels = vec![];
        for i in 0..hnsw_metadata.level_offsets.len() - 1 {
            let start = hnsw_metadata.level_offsets[i];
            let end = hnsw_metadata.level_offsets[i + 1];
            levels.push(HnswLevel::load(reader, start..end, vector_storage.clone()).await?);
        }
        Ok(Self {
            levels,
            metric_type: mt,
            entry_point: hnsw_metadata.entry_point,
        })
    }

    fn from_builder(levels: Vec<HnswLevel>, entry_point: u32, metric_type: MetricType) -> Self {
        Self {
            levels,
            metric_type,
            entry_point,
        }
    }

    pub fn schema(&self) -> SchemaRef {
        self.levels[0].schema()
    }

    /// Search for the nearest neighbors of the query vector.
    ///
    /// Parameters
    /// ----------
    /// query : &[f32]
    ///     The query vector.
    /// k : usize
    ///    The number of nearest neighbors to search for.
    /// ef : usize
    ///    The size of dynamic candidate list
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(u32, f32)>> {
        let mut ep = vec![self.entry_point];
        let num_layers = self.levels.len();
        for layer in self.levels.iter().rev().take(num_layers - 1) {
            let candidates = beam_search(layer, &ep, query, 1)?;
            ep = select_neighbors(&candidates, 1).map(|(_, id)| id).collect();
        }
        let candidates = beam_search(&self.levels[0], &ep, query, ef)?;
        Ok(select_neighbors(&candidates, k)
            .map(|(d, u)| (u, d.into()))
            .collect())
    }

    /// Write the HNSW graph to a Lance file.
    pub async fn write(&self, writer: &mut FileWriter<ManifestDescribing>) -> Result<()> {
        let mut level_offsets = vec![0];
        for level in self.levels.iter() {
            level_offsets.push(level_offsets.last().unwrap() + level.len());
            // TODO: add chunking to each batch.
            writer.write(&[level.nodes.clone()]).await?;
        }
        level_offsets.pop();

        let index_metadata = json!({
            "type": HNSW_TYPE,
            "metric_type": self.metric_type.to_string(),
        });
        let hnsw_metadata = HnswMetadata {
            entry_point: self.entry_point,
            level_offsets,
        };

        let mut metadata = HashMap::<String, String>::new();
        metadata.insert("lance:index".to_string(), index_metadata.to_string());
        metadata.insert(
            "lance:hnsw".to_string(),
            serde_json::to_string(&hnsw_metadata)?,
        );
        writer.finish_with_metadata(&metadata).await?;
        Ok(())
    }
}

/// Select neighbors from the ordered candidate list.
///
/// Algorithm 3 in the HNSW paper.
fn select_neighbors(
    orderd_candidates: &BTreeMap<OrderedFloat, u32>,
    k: usize,
) -> impl Iterator<Item = (OrderedFloat, u32)> + '_ {
    orderd_candidates.iter().take(k).map(|(&d, &u)| (d, u))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;

    use super::super::graph::OrderedFloat;
    use crate::vector::graph::memory::InMemoryVectorStorage;
    use arrow_array::types::Float32Type;
    use lance_linalg::matrix::MatrixView;
    use lance_testing::datagen::generate_random_array;

    #[test]
    fn test_select_neighbors() {
        let candidates: BTreeMap<OrderedFloat, u32> =
            (1..6).map(|i| (OrderedFloat(i as f32), i)).collect();

        let result = select_neighbors(&candidates, 3).collect::<Vec<_>>();
        assert_eq!(
            result,
            vec![
                (OrderedFloat(1.0), 1),
                (OrderedFloat(2.0), 2),
                (OrderedFloat(3.0), 3)
            ]
        );

        assert_eq!(select_neighbors(&candidates, 0).collect::<Vec<_>>(), vec![]);

        assert_eq!(
            select_neighbors(&candidates, 8).collect::<Vec<_>>(),
            vec![
                (OrderedFloat(1.0), 1),
                (OrderedFloat(2.0), 2),
                (OrderedFloat(3.0), 3),
                (OrderedFloat(4.0), 4),
                (OrderedFloat(5.0), 5),
            ]
        );
    }

    #[test]
    fn test_build_hnsw() {
        const DIM: usize = 32;
        const TOTAL: usize = 2048;
        const MAX_EDGES: usize = 32;
        let data = generate_random_array(TOTAL * DIM);
        let mat = Arc::new(MatrixView::<Float32Type>::new(data.into(), DIM));
        let store = Arc::new(InMemoryVectorStorage::new(mat.clone(), MetricType::L2));
        let hnsw = HNSWBuilder::new(store.clone())
            .max_num_edges(MAX_EDGES)
            .ef_construction(50)
            .build()
            .unwrap();
        assert!(hnsw.levels.len() > 1);
        assert_eq!(hnsw.levels[0].len(), TOTAL);

        hnsw.levels.windows(2).for_each(|w| {
            let (prev, next) = (&w[0], &w[1]);
            assert!(prev.len() >= next.len());
        });

        hnsw.levels.iter().for_each(|layer| {
            for i in 0..layer.len() {
                // If the node exist on this layer, check its out-degree.
                if let Some(neighbors) = layer.neighbors(i as u32) {
                    assert!(neighbors.count() <= MAX_EDGES);
                }
            }
        });
    }

    fn ground_truth(mat: &MatrixView<Float32Type>, query: &[f32], k: usize) -> HashSet<u32> {
        let mut dists = vec![];
        for i in 0..mat.num_rows() {
            let dist = lance_linalg::distance::l2_distance(query, mat.row(i).unwrap());
            dists.push((dist, i as u32));
        }
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.truncate(k);
        dists.into_iter().map(|(_, i)| i).collect()
    }

    #[test]
    fn test_search() {
        const DIM: usize = 32;
        const TOTAL: usize = 2048;
        const MAX_EDGES: usize = 32;
        const K: usize = 10;

        let data = generate_random_array(TOTAL * DIM);
        let mat = Arc::new(MatrixView::<Float32Type>::new(data.into(), DIM));
        let vectors = Arc::new(InMemoryVectorStorage::new(mat.clone(), MetricType::L2));
        let q = mat.row(0).unwrap();

        let hnsw = HNSWBuilder::new(vectors.clone())
            .max_num_edges(MAX_EDGES)
            .ef_construction(100)
            .build()
            .unwrap();

        let results: HashSet<u32> = hnsw
            .search(q, 10, 150)
            .unwrap()
            .iter()
            .map(|(i, _)| *i)
            .collect();
        let gt = ground_truth(&mat, q, K);
        let recall = results.intersection(&gt).count() as f32 / K as f32;
        assert!(recall >= 0.7, "Recall: {}", recall);
    }
}
