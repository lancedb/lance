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

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use arrow::datatypes::UInt32Type;
use arrow_array::{
    builder::{ListBuilder, UInt32Builder},
    cast::AsArray,
    ListArray, RecordBatch, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::{StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::{Error, Result};
use lance_file::{reader::FileReader, writer::FileWriter};
use lance_io::object_store::ObjectStore;
use lance_linalg::distance::{DistanceType, MetricType};
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use serde_json::json;
use snafu::{location, Location};

use self::builder::HNSW_METADATA_KEY;

use super::graph::{
    builder::GraphBuilder, greedy_search, storage::VectorStorage, Graph, OrderedFloat, OrderedNode,
    NEIGHBORS_COL, NEIGHBORS_FIELD,
};
use super::ivf::storage::IvfData;
use crate::vector::graph::beam_search;

pub mod builder;

pub use builder::HNSWBuilder;

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
#[derive(Clone)]
struct HnswLevel {
    /// Vector ID to the node index in `nodes`.
    /// The node on different layer share the same Vector ID, which is the index
    /// in the [VectorStorage].
    id_to_node: HashMap<u32, usize>,

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
        let id_to_node = nodes
            .column_by_name(VECTOR_ID_COL)
            .unwrap()
            .as_primitive::<UInt32Type>()
            .values()
            .iter()
            .enumerate()
            .map(|(idx, &vec_id)| (vec_id, idx))
            .collect::<HashMap<u32, usize>>();
        Self {
            nodes,
            neighbors,
            neighbors_values: values,
            id_to_node,
            vectors,
        }
    }

    fn from_builder(builder: &GraphBuilder, vectors: Arc<dyn VectorStorage>) -> Result<Self> {
        let mut vector_id_builder = UInt32Builder::with_capacity(builder.len());
        let mut neighbours_builder =
            ListBuilder::with_capacity(UInt32Builder::new(), builder.len());

        for &id in builder.nodes.keys().sorted() {
            let node = builder.nodes.get(&id).unwrap();
            assert_eq!(node.id, id);
            vector_id_builder.append_value(node.id);
            neighbours_builder.append_value(node.neighbors.clone().iter().map(|n| Some(n.id)));
        }

        let schema = Schema::new(vec![VECTOR_ID_FIELD.clone(), NEIGHBORS_FIELD.clone()]);
        let batch = RecordBatch::try_new(
            schema.into(),
            vec![
                Arc::new(vector_id_builder.finish()),
                Arc::new(neighbours_builder.finish()),
            ],
        )?;

        Ok(Self::new(batch, vectors))
    }

    fn schema(&self) -> SchemaRef {
        self.nodes.schema()
    }

    /// Range of neighbors for the given node, specified by its index.
    fn neighbors_range(&self, id: u32) -> Range<usize> {
        let idx = self.id_to_node[&id];
        let start = self.neighbors.value_offsets()[idx] as usize;
        let end = start + self.neighbors.value_length(idx) as usize;
        start..end
    }
}

impl Graph for HnswLevel {
    fn len(&self) -> usize {
        self.nodes.num_rows()
    }

    fn neighbors(&self, key: u32) -> Option<Box<dyn Iterator<Item = u32> + '_>> {
        let range = self.neighbors_range(key);
        Some(Box::new(
            self.neighbors_values.values()[range].iter().copied(),
        ))
    }

    fn storage(&self) -> Arc<dyn VectorStorage> {
        self.vectors.clone()
    }
}

/// HNSW graph.
///
#[derive(Clone)]
pub struct HNSW {
    levels: Vec<HnswLevel>,
    distance_type: MetricType,
    /// Entry point of the graph.
    entry_point: u32,

    #[allow(dead_code)]
    /// Whether to use the heuristic to select neighbors (Algorithm 4 or 3 in the paper).
    use_select_heuristic: bool,
}

impl Debug for HNSW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HNSW(max_layers: {}, metric={})",
            self.levels.len(),
            self.distance_type
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SchemaIndexMetadata {
    #[serde(rename = "type")]
    type_: String,

    metric_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswMetadata {
    entry_point: u32,
    level_offsets: Vec<usize>,
}

impl HNSW {
    pub fn empty() -> Self {
        Self {
            levels: vec![],
            distance_type: MetricType::L2,
            entry_point: 0,
            use_select_heuristic: true,
        }
    }

    /// The number of nodes in the level 0 of the graph.
    pub fn len(&self) -> usize {
        self.levels[0].len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Metric type of the graph.
    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    /// Load the HNSW graph from a [FileReader].
    ///
    /// Parameters
    /// ----------
    /// - *reader*: the file reader to read the graph from.
    /// - *vector_storage*: A preloaded [VectorStorage] storage.
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
            serde_json::from_str(schema.metadata.get(HNSW_METADATA_KEY).ok_or_else(|| {
                Error::Index {
                    message: "hnsw metadata not found in the schema".to_string(),
                    location: location!(),
                }
            })?)?;

        let levels = futures::stream::iter(0..hnsw_metadata.level_offsets.len() - 1)
            .map(|i| {
                let start = hnsw_metadata.level_offsets[i];
                let end = hnsw_metadata.level_offsets[i + 1];
                HnswLevel::load(reader, start..end, vector_storage.clone())
            })
            .buffered(num_cpus::get())
            .try_collect()
            .await?;
        Ok(Self {
            levels,
            distance_type: mt,
            entry_point: hnsw_metadata.entry_point,
            use_select_heuristic: true,
        })
    }

    /// Load a partition of HNSW
    ///
    /// Parameters
    /// ----------
    /// - *reader*: the file reader to read the graph from.
    /// - *range*: the row range of the partition.
    /// - *metric_type*: the metric type of the index.
    /// - *vector_storage*: A preloaded [VectorStorage] storage.
    /// - *metadata*: the metadata of the HNSW.
    pub async fn load_partition(
        reader: &FileReader,
        range: std::ops::Range<usize>,
        metric_type: MetricType,
        vector_storage: Arc<dyn VectorStorage>,
        metadata: HnswMetadata,
    ) -> Result<Self> {
        let mut levels = Vec::with_capacity(metadata.level_offsets.len() - 1);
        // TODO: load levels in parallel.
        for i in 0..metadata.level_offsets.len() - 1 {
            let start = metadata.level_offsets[i] + range.start;
            let end = metadata.level_offsets[i + 1] + range.end;
            levels.push(HnswLevel::load(reader, start..end, vector_storage.clone()).await?);
        }
        Ok(Self {
            levels,
            distance_type: metric_type,
            entry_point: metadata.entry_point,
            use_select_heuristic: true,
        })
    }

    fn from_builder(
        levels: Vec<HnswLevel>,
        entry_point: u32,
        metric_type: MetricType,
        use_select_heuristic: bool,
    ) -> Self {
        Self {
            levels,
            distance_type: metric_type,
            entry_point,
            use_select_heuristic,
        }
    }

    /// The Arrow schema of the graph.
    pub fn schema(&self) -> SchemaRef {
        self.levels[0].schema()
    }

    /// Search for the nearest neighbors of the query vector.
    ///
    /// Parameters
    /// ----------
    /// - *query* : the query vector.
    /// - *k* : the number of nearest neighbors to search for.
    /// - *ef* : the size of dynamic candidate list.
    ///
    /// Returns
    /// -------
    /// A list of `(id_in_graph, distance)` pairs. Or Error if the search failed.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        bitset: Option<RoaringBitmap>,
    ) -> Result<Vec<(u32, f32)>> {
        let mut ep = self.entry_point;
        let num_layers = self.levels.len();

        for level in self.levels.iter().rev().take(num_layers - 1) {
            ep = greedy_search(level, ep, query)?.1;
        }

        let candidates = beam_search(&self.levels[0], &[ep], query, ef, bitset.as_ref())?;
        Ok(select_neighbors(&candidates, k)
            .map(|(d, u)| (u, d.into()))
            .collect())
    }

    /// Returns the metadata of this [`HNSW`].
    pub fn metadata(&self) -> HnswMetadata {
        let mut level_offsets = Vec::with_capacity(self.levels.len() + 1);
        let mut offset = 0;
        level_offsets.push(offset);
        for level in self.levels.iter() {
            offset += level.len();
            level_offsets.push(offset);
        }

        HnswMetadata {
            entry_point: self.entry_point,
            level_offsets,
        }
    }

    /// Write the HNSW graph to a Lance file.
    pub async fn write(&self, writer: &mut FileWriter<ManifestDescribing>) -> Result<()> {
        self.write_levels(writer).await?;

        let index_metadata = json!({
            "type": HNSW_TYPE,
            "metric_type": self.distance_type.to_string(),
        });
        let hnsw_metadata = self.metadata();

        let mut metadata = HashMap::<String, String>::new();
        metadata.insert("lance:index".to_string(), index_metadata.to_string());
        metadata.insert(
            "lance:hnsw".to_string(),
            serde_json::to_string(&hnsw_metadata)?,
        );
        writer.finish_with_metadata(&metadata).await?;
        Ok(())
    }

    /// Write partitioned HNSWs to the file.
    ///
    /// Parameters
    /// ----------
    /// - *object_store*: the object store to write the file to.
    /// - *path*: the path to write the file to.
    /// - *partitions*: the partitions of the HNSW graph.
    pub async fn write_parted_hnsw(
        object_store: &ObjectStore,
        path: &Path,
        partitions: Box<dyn Iterator<Item = Self>>,
    ) -> Result<()> {
        let mut peek = partitions.peekable();
        let first = peek.peek().ok_or(Error::Index {
            message: "No partitions to write".to_string(),
            location: location!(),
        })?;
        let schema = first.schema();
        let lance_schema = lance_core::datatypes::Schema::try_from(schema.as_ref())?;
        let mut writer = FileWriter::<ManifestDescribing>::try_new(
            object_store,
            path,
            lance_schema,
            &Default::default(), // TODO: support writer options.
        )
        .await?;

        let mut ivf_data = IvfData::empty();
        for hnsw in peek {
            let num_rows = hnsw.write_levels(&mut writer).await?;
            ivf_data.add_partition(num_rows as u32);
        }
        ivf_data.write(&mut writer).await?;

        Ok(())
    }

    /// Write levels' nodes, and returns the offset of each level.
    pub async fn write_levels(&self, writer: &mut FileWriter<ManifestDescribing>) -> Result<usize> {
        let mut num_rows = 0;
        for level in self.levels.iter() {
            writer.write(&[level.nodes.clone()]).await?;
            num_rows += level.nodes.num_rows();
        }
        Ok(num_rows)
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

/// Algorithm 4 in the HNSW paper.
///
///
/// Modifies to the original algorithm:
/// 1. Do not use keepPrunedConnections, we use a heap to capture nearest neighbors.
fn select_neighbors_heuristic(
    graph: &dyn Graph,
    query: &[f32],
    orderd_candidates: &BTreeMap<OrderedFloat, u32>,
    k: usize,
    extend_candidates: bool,
) -> impl Iterator<Item = (OrderedFloat, u32)> {
    let mut heap: BinaryHeap<OrderedNode> = BinaryHeap::from_iter(
        orderd_candidates
            .iter()
            .map(|(&d, &u)| OrderedNode { id: u, dist: d }),
    );

    if extend_candidates {
        let dist_calc = graph.storage().dist_calculator(query);
        let mut visited = HashSet::with_capacity(orderd_candidates.len() * 64);
        visited.extend(orderd_candidates.values());
        orderd_candidates.iter().for_each(|(_, &u)| {
            if let Some(neighbors) = graph.neighbors(u) {
                neighbors.for_each(|n| {
                    if !visited.contains(&n) {
                        let d: OrderedFloat = dist_calc.distance(&[n])[0].into();
                        heap.push((d, n).into());
                    }
                    visited.insert(n);
                });
            }
        });
    }

    heap.into_sorted_vec().into_iter().take(k).map(|n| n.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::vector::graph::memory::InMemoryVectorStorage;
    use arrow_array::types::Float32Type;
    use lance_linalg::matrix::MatrixView;
    use lance_testing::datagen::generate_random_array;
    use tests::builder::HnswBuildParams;

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
                (OrderedFloat(3.0), 3),
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
        let hnsw = HNSWBuilder::with_params(
            HnswBuildParams::default()
                .max_num_edges(MAX_EDGES)
                .ef_construction(50),
            store.clone(),
        )
        .build()
        .unwrap();
        assert!(hnsw.levels.len() > 1);
        assert_eq!(hnsw.levels[0].len(), TOTAL);

        hnsw.levels.windows(2).for_each(|w| {
            let (prev, next) = (&w[0], &w[1]);
            assert!(prev.len() >= next.len());
        });

        hnsw.levels.iter().for_each(|layer| {
            for &i in layer.id_to_node.keys() {
                // If the node exist on this layer, check its out-degree.
                if let Some(neighbors) = layer.neighbors(i) {
                    let cnt = neighbors.count();
                    assert!(cnt <= MAX_EDGES, "actual {}, max_edges: {}", cnt, MAX_EDGES);
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
        const TOTAL: usize = 10_000;
        const MAX_EDGES: usize = 30;
        const K: usize = 100;

        let data = generate_random_array(TOTAL * DIM);
        let mat = Arc::new(MatrixView::<Float32Type>::new(data.into(), DIM));
        let vectors = Arc::new(InMemoryVectorStorage::new(mat.clone(), MetricType::L2));
        let q = mat.row(0).unwrap();

        let hnsw = HNSWBuilder::with_params(
            HnswBuildParams::default()
                .max_num_edges(MAX_EDGES)
                .ef_construction(100)
                .max_level(4),
            vectors.clone(),
        )
        .build()
        .unwrap();

        let results: HashSet<u32> = hnsw
            .search(q, K, 128, None)
            .unwrap()
            .iter()
            .map(|(i, _)| *i)
            .collect();
        let gt = ground_truth(&mat, q, K);
        let recall = results.intersection(&gt).count() as f32 / K as f32;
        // TODO: improve the recall.
        assert!(recall >= 0.9, "Recall: {}", recall);
    }
}
