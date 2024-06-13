// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Builder of Hnsw Graph.

use arrow::array::{AsArray, ListBuilder, UInt32Builder};
use arrow::compute::concat_batches;
use arrow::datatypes::{Float32Type, UInt32Type};
use arrow_array::{ArrayRef, Float32Array, RecordBatch, UInt64Array};
use crossbeam_queue::ArrayQueue;
use deepsize::DeepSizeOf;
use itertools::Itertools;

use lance_linalg::distance::DistanceType;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use roaring::RoaringBitmap;
use serde_json::json;
use snafu::{location, Location};
use std::cmp::min;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::RwLock;

use lance_core::{Error, Result};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

use super::super::graph::beam_search;
use super::{select_neighbors_heuristic, HnswMetadata, HNSW_TYPE, VECTOR_ID_COL, VECTOR_ID_FIELD};
use crate::prefilter::PreFilter;
use crate::vector::flat::storage::FlatStorage;
use crate::vector::graph::builder::GraphBuilderNode;
use crate::vector::graph::greedy_search;
use crate::vector::graph::{
    Graph, OrderedFloat, OrderedNode, VisitedGenerator, DISTS_FIELD, NEIGHBORS_COL, NEIGHBORS_FIELD,
};
use crate::vector::storage::{DistCalculator, VectorStore};
use crate::vector::v3::subindex::IvfSubIndex;
use crate::vector::{Query, DIST_COL, VECTOR_RESULT_SCHEMA};
use crate::{IndexMetadata, INDEX_METADATA_SCHEMA_KEY};

pub const HNSW_METADATA_KEY: &str = "lance:hnsw";

/// Parameters of building HNSW index
#[derive(Debug, Clone, Serialize, Deserialize, DeepSizeOf)]
pub struct HnswBuildParams {
    /// max level ofm
    pub max_level: u16,

    /// number of connections to establish while inserting new element
    pub m: usize,

    /// size of the dynamic list for the candidates
    pub ef_construction: usize,

    /// the max number of threads to use for building the graph
    pub parallel_limit: Option<usize>,

    /// number of vectors ahead to prefetch while building the graph
    pub prefetch_distance: Option<usize>,
}

impl Default for HnswBuildParams {
    fn default() -> Self {
        Self {
            max_level: 7,
            m: 20,
            ef_construction: 150,
            parallel_limit: None,
            prefetch_distance: Some(2),
        }
    }
}

impl HnswBuildParams {
    /// The maximum level of the graph.
    /// The default value is `8`.
    pub fn max_level(mut self, max_level: u16) -> Self {
        self.max_level = max_level;
        self
    }

    /// The number of connections to establish while inserting new element
    /// The default value is `30`.
    pub fn num_edges(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Number of candidates to be considered when searching for the nearest neighbors
    /// during the construction of the graph.
    ///
    /// The default value is `100`.
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// The max number of threads to use for building the graph.
    pub fn parallel_limit(mut self, limit: usize) -> Self {
        self.parallel_limit = Some(limit);
        self
    }

    pub async fn build(self, data: ArrayRef) -> Result<HNSW> {
        // We have normalized the vectors if the metric type is cosine, so we can use the L2 distance
        let vec_store = Arc::new(FlatStorage::new(
            data.as_fixed_size_list().clone(),
            DistanceType::L2,
        ));
        HNSW::index_vectors(vec_store.as_ref(), self)
    }
}

/// Build a HNSW graph.
///
/// Currently, the HNSW graph is fully built in memory.
///
/// During the build, the graph is built layer by layer.
///
/// Each node in the graph has a global ID which is the index on the base layer.
#[derive(Clone, DeepSizeOf)]
pub struct HNSW {
    inner: Arc<HnswBuilder>,
}

impl Debug for HNSW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HNSW(max_layers: {}, metric={})",
            self.inner.max_level() as usize,
            self.inner.distance_type,
        )
    }
}

impl HNSW {
    pub fn len(&self) -> usize {
        self.inner.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn max_level(&self) -> u16 {
        self.inner.max_level()
    }

    pub fn num_nodes(&self, level: usize) -> usize {
        self.inner.num_nodes(level)
    }

    pub fn nodes(&self) -> Arc<Vec<RwLock<GraphBuilderNode>>> {
        self.inner.nodes()
    }

    pub fn distance_type(&self) -> DistanceType {
        self.inner.distance_type
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search_inner(
        &self,
        query: ArrayRef,
        k: usize,
        ef: usize,
        bitset: Option<RoaringBitmap>,
        visited_generator: &mut VisitedGenerator,
        storage: &impl VectorStore,
        prefetch_distance: Option<usize>,
    ) -> Result<Vec<OrderedNode>> {
        let dist_calc = storage.dist_calculator(query);
        let mut ep = OrderedNode::new(0, dist_calc.distance(0).into());
        let nodes = &self.nodes();
        for level in (0..self.max_level()).rev() {
            let cur_level = HnswLevelView::new(level, nodes);
            ep = greedy_search(&cur_level, ep, &dist_calc);
        }

        let bottom_level = HnswBottomView::new(nodes);
        Ok(beam_search(
            &bottom_level,
            &ep,
            ef,
            &dist_calc,
            bitset.as_ref(),
            prefetch_distance,
            visited_generator,
        )
        .into_iter()
        .take(k)
        .collect())
    }

    pub fn search_basic(
        &self,
        query: ArrayRef,
        k: usize,
        ef: usize,
        bitset: Option<RoaringBitmap>,
        storage: &impl VectorStore,
    ) -> Result<Vec<OrderedNode>> {
        let mut visited_generator = self
            .inner
            .visited_generator_queue
            .pop()
            .unwrap_or_else(|| VisitedGenerator::new(storage.len()));
        let result = self.search_inner(
            query,
            k,
            ef,
            bitset,
            &mut visited_generator,
            storage,
            Some(2),
        );

        match self.inner.visited_generator_queue.push(visited_generator) {
            Ok(_) => {}
            Err(_) => {
                println!("visited_generator_queue is full");
            }
        }

        result
    }

    /// Returns the metadata of this [`HNSW`].
    pub fn metadata(&self) -> HnswMetadata {
        // calculate the offsets of each level,
        // start from 0
        let level_offsets = self
            .inner
            .level_count
            .iter()
            .chain(iter::once(&AtomicUsize::new(0)))
            .scan(0, |state, x| {
                let start = *state;
                *state += x.load(Ordering::Relaxed);
                Some(start)
            })
            .collect();

        HnswMetadata {
            entry_point: self.inner.entry_point,
            params: self.inner.params.clone(),
            level_offsets,
        }
    }

    pub fn schema() -> arrow_schema::Schema {
        arrow_schema::Schema::new(vec![
            VECTOR_ID_FIELD.clone(),
            NEIGHBORS_FIELD.clone(),
            DISTS_FIELD.clone(),
        ])
    }
}

struct HnswBuilder {
    distance_type: DistanceType,
    params: HnswBuildParams,

    nodes: Arc<Vec<RwLock<GraphBuilderNode>>>,
    level_count: Vec<AtomicUsize>,

    entry_point: u32,

    visited_generator_queue: Arc<ArrayQueue<VisitedGenerator>>,
}

impl DeepSizeOf for HnswBuilder {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.distance_type.deep_size_of_children(context)
            + self.params.deep_size_of_children(context)
            + self.nodes.deep_size_of_children(context)
            + self.level_count.deep_size_of_children(context)
        // Skipping the visited_generator_queue
    }
}

impl HnswBuilder {
    fn max_level(&self) -> u16 {
        self.params.max_level
    }

    fn num_nodes(&self, level: usize) -> usize {
        self.level_count[level].load(Ordering::Relaxed)
    }

    fn nodes(&self) -> Arc<Vec<RwLock<GraphBuilderNode>>> {
        self.nodes.clone()
    }

    /// Create a new [`HNSWBuilder`] with prepared params and in memory vector storage.
    pub fn with_params(
        distance_type: DistanceType,
        params: HnswBuildParams,
        storage: &impl VectorStore,
    ) -> Self {
        let len = storage.len();
        let max_level = params.max_level;

        let level_count = (0..max_level)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>();

        let visited_generator_queue = Arc::new(ArrayQueue::new(num_cpus::get() * 2));
        for _ in 0..(num_cpus::get() * 2) {
            visited_generator_queue
                .push(VisitedGenerator::new(0))
                .unwrap();
        }
        let mut builder = Self {
            distance_type,
            params,
            nodes: Arc::new(Vec::with_capacity(0)),
            level_count,
            entry_point: 0,
            visited_generator_queue,
        };

        if storage.is_empty() {
            return builder;
        }

        let mut nodes = Vec::with_capacity(len);
        {
            nodes.push(RwLock::new(GraphBuilderNode::new(0, max_level as usize)));
            for i in 1..len {
                nodes.push(RwLock::new(GraphBuilderNode::new(
                    i as u32,
                    builder.random_level() as usize + 1,
                )));
            }
        }
        builder.nodes = Arc::new(nodes);

        builder
    }

    /// New node's level
    ///
    /// See paper `Algorithm 1`
    fn random_level(&self) -> u16 {
        let mut rng = thread_rng();
        let ml = 1.0 / (self.params.m as f32).ln();
        min(
            (-rng.gen::<f32>().ln() * ml) as u16,
            self.params.max_level - 1,
        )
    }

    /// Insert one node.
    fn insert(
        &self,
        node: u32,
        visited_generator: &mut VisitedGenerator,
        storage: &impl VectorStore,
    ) {
        let nodes = &self.nodes;
        let target_level = nodes[node as usize].read().unwrap().level_neighbors.len() as u16 - 1;
        let mut ep = OrderedNode::new(
            self.entry_point,
            storage.distance_between(node, self.entry_point).into(),
        );

        //
        // Search for entry point in paper.
        // ```
        //   for l_c in (L..l+1) {
        //     W = Search-Layer(q, ep, ef=1, l_c)
        //    ep = Select-Neighbors(W, 1)
        //  }
        // ```
        let dist_calc = storage.dist_calculator_from_id(node);
        for level in (target_level + 1..self.params.max_level).rev() {
            let cur_level = HnswLevelView::new(level, nodes);
            ep = greedy_search(&cur_level, ep, &dist_calc);
        }

        let mut pruned_neighbors_per_level: Vec<Vec<_>> =
            vec![Vec::new(); (target_level + 1) as usize];
        {
            let mut current_node = nodes[node as usize].write().unwrap();
            for level in (0..=target_level).rev() {
                self.level_count[level as usize].fetch_add(1, Ordering::Relaxed);

                let neighbors = self.search_level(&ep, level, &dist_calc, nodes, visited_generator);
                for neighbor in &neighbors {
                    current_node.add_neighbor(neighbor.id, neighbor.dist, level);
                }
                self.prune(storage, &mut current_node, level);
                pruned_neighbors_per_level[level as usize]
                    .clone_from(&current_node.level_neighbors_ranked[level as usize]);

                ep = neighbors[0].clone();
            }
        }
        for (level, pruned_neighbors) in pruned_neighbors_per_level.iter().enumerate() {
            let _: Vec<_> = pruned_neighbors
                .iter()
                .map(|unpruned_edge| {
                    let level = level as u16;
                    let m_max = match level {
                        0 => self.params.m * 2,
                        _ => self.params.m,
                    };
                    if unpruned_edge.dist
                        < nodes[unpruned_edge.id as usize]
                            .read()
                            .unwrap()
                            .cutoff(level, m_max)
                    {
                        let mut chosen_node = nodes[unpruned_edge.id as usize].write().unwrap();
                        chosen_node.add_neighbor(node, unpruned_edge.dist, level);
                        self.prune(storage, &mut chosen_node, level);
                    }
                })
                .collect();
        }
    }

    fn search_level(
        &self,
        ep: &OrderedNode,
        level: u16,
        dist_calc: &impl DistCalculator,
        nodes: &Vec<RwLock<GraphBuilderNode>>,
        visited_generator: &mut VisitedGenerator,
    ) -> Vec<OrderedNode> {
        let cur_level = HnswLevelView::new(level, nodes);
        beam_search(
            &cur_level,
            ep,
            self.params.ef_construction,
            dist_calc,
            None,
            self.params.prefetch_distance,
            visited_generator,
        )
    }

    fn prune(&self, storage: &impl VectorStore, builder_node: &mut GraphBuilderNode, level: u16) {
        let m_max = match level {
            0 => self.params.m * 2,
            _ => self.params.m,
        };

        let neighbors_ranked = &mut builder_node.level_neighbors_ranked[level as usize];
        let level_neighbors = neighbors_ranked.clone();
        if level_neighbors.len() <= m_max {
            builder_node.update_from_ranked_neighbors(level);
            return;
            //return level_neighbors;
        }

        *neighbors_ranked = select_neighbors_heuristic(storage, &level_neighbors, m_max);
        builder_node.update_from_ranked_neighbors(level);
    }
}

// View of a level in HNSW graph.
// This is used to iterate over neighbors in a specific level.
pub(crate) struct HnswLevelView<'a> {
    level: u16,
    nodes: &'a Vec<RwLock<GraphBuilderNode>>,
}

impl<'a> HnswLevelView<'a> {
    pub fn new(level: u16, nodes: &'a Vec<RwLock<GraphBuilderNode>>) -> Self {
        Self { level, nodes }
    }
}

impl<'a> Graph for HnswLevelView<'a> {
    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn neighbors(&self, key: u32) -> Arc<Vec<u32>> {
        let node = &self.nodes[key as usize];
        node.read().unwrap().level_neighbors[self.level as usize].clone()
    }
}

pub(crate) struct HnswBottomView<'a> {
    nodes: &'a Vec<RwLock<GraphBuilderNode>>,
}

impl<'a> HnswBottomView<'a> {
    pub fn new(nodes: &'a Vec<RwLock<GraphBuilderNode>>) -> Self {
        Self { nodes }
    }
}

impl<'a> Graph for HnswBottomView<'a> {
    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn neighbors(&self, key: u32) -> Arc<Vec<u32>> {
        let node = &self.nodes[key as usize];
        node.read().unwrap().bottom_neighbors.clone()
    }
}

pub struct HnswQueryParams {
    pub ef: usize,
}

impl<'a> From<&'a Query> for HnswQueryParams {
    fn from(query: &Query) -> Self {
        let k = query.k * query.refine_factor.unwrap_or(1) as usize;
        Self {
            ef: query.ef.unwrap_or(k + k / 2),
        }
    }
}

impl IvfSubIndex for HNSW {
    type BuildParams = HnswBuildParams;
    type QueryParams = HnswQueryParams;

    fn load(data: RecordBatch) -> Result<Self>
    where
        Self: Sized,
    {
        let index_metadata = data
            .schema_ref()
            .metadata()
            .get(INDEX_METADATA_SCHEMA_KEY)
            .ok_or(Error::Index {
                message: format!("{} not found", INDEX_METADATA_SCHEMA_KEY),
                location: location!(),
            })?;
        let index_metadata: IndexMetadata = serde_json::from_str(index_metadata)?;
        let distance_type = DistanceType::try_from(index_metadata.distance_type.as_str())?;

        let hnsw_metadata =
            data.schema_ref()
                .metadata()
                .get(HNSW_METADATA_KEY)
                .ok_or(Error::Index {
                    message: format!("{} not found", HNSW_METADATA_KEY),
                    location: location!(),
                })?;
        let hnsw_metadata: HnswMetadata = serde_json::from_str(hnsw_metadata)?;

        let levels: Vec<_> = hnsw_metadata
            .level_offsets
            .iter()
            .tuple_windows()
            .map(|(start, end)| data.slice(*start, end - start))
            .collect();

        let level_count = levels.iter().map(|b| b.num_rows()).collect::<Vec<_>>();

        let bottom_level_len = levels[0].num_rows();
        let mut nodes = Vec::with_capacity(bottom_level_len);
        for i in 0..bottom_level_len {
            nodes.push(GraphBuilderNode::new(i as u32, levels.len()));
        }
        for (level, batch) in levels.into_iter().enumerate() {
            let ids = batch
                .column_by_name(VECTOR_ID_COL)
                .ok_or(Error::Index {
                    message: format!("{} column not found in HNSW file", VECTOR_ID_COL),
                    location: location!(),
                })?
                .as_primitive::<UInt32Type>();
            let neighbors = batch
                .column_by_name(NEIGHBORS_COL)
                .ok_or(Error::Index {
                    message: format!("{} column not found in HNSW file", NEIGHBORS_COL),
                    location: location!(),
                })?
                .as_list::<i32>();
            let distances = batch
                .column_by_name(DIST_COL)
                .ok_or(Error::Index {
                    message: format!("{} column not found in HNSW file", DIST_COL),
                    location: location!(),
                })?
                .as_list::<i32>();

            for ((node, neighbors), distances) in
                ids.iter().zip(neighbors.iter()).zip(distances.iter())
            {
                let node = node.unwrap();
                let neighbors = neighbors.as_ref().unwrap().as_primitive::<UInt32Type>();
                let distances = distances.as_ref().unwrap().as_primitive::<Float32Type>();

                nodes[node as usize].level_neighbors_ranked[level] = neighbors
                    .iter()
                    .zip(distances.iter())
                    .map(|(n, dist)| OrderedNode::new(n.unwrap(), OrderedFloat(dist.unwrap())))
                    .collect();
                nodes[node as usize].update_from_ranked_neighbors(level as u16);
            }
        }

        let visited_generator_queue = Arc::new(ArrayQueue::new(num_cpus::get() * 2));
        for _ in 0..(num_cpus::get() * 2) {
            visited_generator_queue
                .push(VisitedGenerator::new(0))
                .unwrap();
        }
        let inner = HnswBuilder {
            distance_type,
            params: hnsw_metadata.params,
            nodes: Arc::new(nodes.into_iter().map(RwLock::new).collect()),
            level_count: level_count.into_iter().map(AtomicUsize::new).collect(),
            entry_point: hnsw_metadata.entry_point,
            visited_generator_queue,
        };

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn use_residual() -> bool {
        false
    }

    fn name(&self) -> &str {
        HNSW_TYPE
    }

    fn search(
        &self,
        query: ArrayRef,
        k: usize,
        params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
    ) -> Result<RecordBatch> {
        let schema = VECTOR_RESULT_SCHEMA.clone();

        if self.is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }

        let bitmap = if prefilter.is_empty() {
            None
        } else {
            let indices = prefilter.filter_row_ids(Box::new(storage.row_ids()));
            Some(
                RoaringBitmap::from_sorted_iter(indices.into_iter().map(|i| i as u32)).map_err(
                    |e| Error::Index {
                        message: format!("Error creating RoaringBitmap: {}", e),
                        location: location!(),
                    },
                )?,
            )
        };

        if params.ef < k {
            return Err(Error::Index {
                message: "ef must be greater than or equal to k".to_string(),
                location: location!(),
            });
        }

        let results = self.search_basic(query.clone(), k, params.ef, bitmap, storage)?;

        let row_ids = UInt64Array::from_iter_values(results.iter().map(|x| storage.row_id(x.id)));
        let distances = Arc::new(Float32Array::from_iter_values(
            results.iter().map(|x| x.dist.0),
        ));

        Ok(RecordBatch::try_new(
            schema,
            vec![distances, Arc::new(row_ids)],
        )?)
    }

    /// Given a vector storage, containing all the data for the IVF partition, build the sub index.
    fn index_vectors(storage: &impl VectorStore, params: Self::BuildParams) -> Result<Self>
    where
        Self: Sized,
    {
        let inner = HnswBuilder::with_params(storage.distance_type(), params, storage);
        let hnsw = Self {
            inner: Arc::new(inner),
        };

        log::info!(
            "Building HNSW graph: num={}, metric_type={}, max_levels={}, m={}, ef_construction={}",
            storage.len(),
            hnsw.inner.distance_type,
            hnsw.inner.params.max_level,
            hnsw.inner.params.m,
            hnsw.inner.params.ef_construction
        );
        if storage.len() <= 1 {
            return Ok(hnsw);
        }

        let len = storage.len();
        let parallel_limit = hnsw
            .inner
            .params
            .parallel_limit
            .unwrap_or_else(num_cpus::get)
            .max(1);
        log::info!("Building HNSW graph with parallel_limit={}", parallel_limit);
        let chunk_size = (storage.len() - 1).div_ceil(parallel_limit);
        hnsw.inner.level_count[0].fetch_add(1, Ordering::Relaxed);
        (1..storage.len())
            .into_par_iter()
            .chunks(chunk_size)
            .for_each(|chunk| {
                let mut visited_generator = VisitedGenerator::new(len);
                for node in chunk.into_iter() {
                    hnsw.inner
                        .insert(node as u32, &mut visited_generator, storage);
                }
            });

        assert_eq!(hnsw.inner.level_count[0].load(Ordering::Relaxed), len);
        Ok(hnsw)
    }

    /// Return the schema of the sub index
    fn schema(&self) -> arrow_schema::SchemaRef {
        Self::schema().into()
    }

    /// Encode the sub index into a record batch
    fn to_batch(&self) -> Result<RecordBatch> {
        let mut vector_id_builder = UInt32Builder::with_capacity(self.len());
        let mut neighbors_builder = ListBuilder::with_capacity(UInt32Builder::new(), self.len());
        let mut distances_builder =
            ListBuilder::with_capacity(arrow_array::builder::Float32Builder::new(), self.len());
        let mut batches = Vec::with_capacity(self.max_level() as usize);
        for level in 0..self.max_level() {
            let level = level as usize;
            for (id, node) in self.inner.nodes.iter().enumerate() {
                let node = node.read().unwrap();
                if level >= node.level_neighbors.len() || node.level_neighbors[level].is_empty() {
                    continue;
                }
                let neighbors = node.level_neighbors[level].iter().map(|n| Some(*n));
                let distances = node.level_neighbors_ranked[level]
                    .iter()
                    .map(|n| Some(n.dist.0));
                vector_id_builder.append_value(id as u32);
                neighbors_builder.append_value(neighbors);
                distances_builder.append_value(distances);
            }

            let batch = RecordBatch::try_new(
                self.schema(),
                vec![
                    Arc::new(vector_id_builder.finish()),
                    Arc::new(neighbors_builder.finish()),
                    Arc::new(distances_builder.finish()),
                ],
            )?;
            batches.push(batch);
        }

        let index_metadata = json!(IndexMetadata {
            index_type: HNSW_TYPE.to_string(),
            distance_type: self.inner.distance_type.to_string(),
        });

        let metadata = self.metadata();
        let metadata = serde_json::to_string(&metadata)?;
        let schema = self
            .schema()
            .as_ref()
            .clone()
            .with_metadata(HashMap::from_iter(vec![
                (
                    INDEX_METADATA_SCHEMA_KEY.to_string(),
                    index_metadata.to_string(),
                ),
                (HNSW_METADATA_KEY.to_string(), metadata),
            ]));
        let batch = concat_batches(&self.schema(), batches.iter())?;
        let batch = batch.with_schema(Arc::new(schema))?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::FixedSizeListArray;
    use arrow_schema::Schema;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_file::{
        reader::FileReader,
        writer::{FileWriter, FileWriterOptions},
    };
    use lance_io::object_store::ObjectStore;
    use lance_linalg::distance::DistanceType;
    use lance_table::format::SelfDescribingFileReader;
    use lance_table::io::manifest::ManifestDescribing;
    use lance_testing::datagen::generate_random_array;
    use object_store::path::Path;

    use crate::scalar::IndexWriter;
    use crate::vector::v3::subindex::IvfSubIndex;
    use crate::vector::{
        flat::storage::FlatStorage,
        graph::{DISTS_FIELD, NEIGHBORS_FIELD},
        hnsw::{builder::HnswBuildParams, HNSW, VECTOR_ID_FIELD},
    };

    #[tokio::test]
    async fn test_builder_write_load() {
        const DIM: usize = 32;
        const TOTAL: usize = 2048;
        const NUM_EDGES: usize = 20;
        let data = generate_random_array(TOTAL * DIM);
        let fsl = FixedSizeListArray::try_new_from_values(data, DIM as i32).unwrap();
        let store = Arc::new(FlatStorage::new(fsl.clone(), DistanceType::L2));
        let builder = HNSW::index_vectors(
            store.as_ref(),
            HnswBuildParams::default()
                .num_edges(NUM_EDGES)
                .ef_construction(50),
        )
        .unwrap();

        let object_store = ObjectStore::memory();
        let path = Path::from("test_builder_write_load");
        let writer = object_store.create(&path).await.unwrap();
        let schema = Schema::new(vec![
            VECTOR_ID_FIELD.clone(),
            NEIGHBORS_FIELD.clone(),
            DISTS_FIELD.clone(),
        ]);
        let schema = lance_core::datatypes::Schema::try_from(&schema).unwrap();
        let mut writer = FileWriter::<ManifestDescribing>::with_object_writer(
            writer,
            schema,
            &FileWriterOptions::default(),
        )
        .unwrap();
        let batch = builder.to_batch().unwrap();
        let metadata = batch.schema_ref().metadata().clone();
        writer.write_record_batch(batch).await.unwrap();
        writer.finish_with_metadata(&metadata).await.unwrap();

        let reader = FileReader::try_new_self_described(&object_store, &path, None)
            .await
            .unwrap();
        let batch = reader
            .read_range(0..reader.len(), reader.schema())
            .await
            .unwrap();
        let loaded_hnsw = HNSW::load(batch).unwrap();

        let query = fsl.value(0);
        let k = 10;
        let ef = 50;
        let builder_results = builder
            .search_basic(query.clone(), k, ef, None, store.as_ref())
            .unwrap();
        let loaded_results = loaded_hnsw
            .search_basic(query, k, ef, None, store.as_ref())
            .unwrap();
        assert_eq!(builder_results, loaded_results);
    }
}
