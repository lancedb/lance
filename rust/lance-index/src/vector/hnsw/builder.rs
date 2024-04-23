// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Builder of Hnsw Graph.

use std::cmp::min;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use itertools::Itertools;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::Result;
use rand::{thread_rng, Rng};

use super::super::graph::beam_search;
use super::{select_neighbors, select_neighbors_heuristic, HNSW};
use crate::vector::graph::builder::GraphBuilderNode;
use crate::vector::graph::storage::DistCalculator;
use crate::vector::graph::{greedy_search, storage::VectorStorage};
use crate::vector::graph::{Graph, OrderedFloat, OrderedNode};

pub const HNSW_METADATA_KEY: &str = "lance:hnsw";

/// Parameters of building HNSW index
#[derive(Debug, Clone)]
pub struct HnswBuildParams {
    /// max level ofm
    pub max_level: u16,

    /// number of connections to establish while inserting new element
    pub m: usize,

    /// max number of connections for each element per layers.
    pub m_max: usize,

    /// size of the dynamic list for the candidates
    pub ef_construction: usize,

    /// whether select neighbors heuristic
    pub use_select_heuristic: bool,

    /// the max number of threads to use for building the graph
    pub parallel_limit: Option<usize>,
}

impl Default for HnswBuildParams {
    fn default() -> Self {
        Self {
            max_level: 7,
            m: 20,
            m_max: 40,
            ef_construction: 100,
            use_select_heuristic: true,
            parallel_limit: None,
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

    /// The maximum number of connections for each node per layer.
    /// The default value is `64`.
    pub fn max_num_edges(mut self, m_max: usize) -> Self {
        self.m_max = m_max;
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

    /// Use select heuristic when searching for the nearest neighbors.
    ///
    /// See algorithm 4 in HNSW paper.
    pub fn use_select_heuristic(mut self, flag: bool) -> Self {
        self.use_select_heuristic = flag;
        self
    }

    /// The max number of threads to use for building the graph.
    pub fn parallel_limit(mut self, limit: usize) -> Self {
        self.parallel_limit = Some(limit);
        self
    }
}

/// Build a HNSW graph.
///
/// Currently, the HNSW graph is fully built in memory.
///
/// During the build, the graph is built layer by layer.
///
/// Each node in the graph has a global ID which is the index on the base layer.
pub struct HNSWBuilder {
    inner: Arc<HNSWBuilderInner>,
}

impl HNSWBuilder {
    /// Create a new [`HNSWBuilder`] with in memory vector storage.
    pub fn new(vectors: Arc<dyn VectorStorage>) -> Self {
        Self::with_params(HnswBuildParams::default(), vectors)
    }

    pub fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }

    pub fn num_nodes(&self, level: usize) -> usize {
        self.inner.num_nodes(level)
    }

    pub fn nodes(&self) -> Arc<RwLock<Vec<GraphBuilderNode>>> {
        self.inner.nodes()
    }

    pub fn storage(&self) -> Arc<dyn VectorStorage> {
        self.inner.storage()
    }

    /// Create a new [`HNSWBuilder`] with prepared params and in memory vector storage.
    pub fn with_params(params: HnswBuildParams, vectors: Arc<dyn VectorStorage>) -> Self {
        let inner = Arc::new(HNSWBuilderInner::with_params(params, vectors));
        Self { inner }
    }

    /// Build the graph, with the already provided `VectorStorage` as backing storage for HNSW graph.
    pub async fn build(&mut self) -> Result<HNSW> {
        log::info!(
            "Building HNSW graph: num={}, metric_type={}, max_levels={}, m_max={}, ef_construction={}",
            self.inner.vectors.len(),
            self.inner.vectors.metric_type(),
            self.inner.params.max_level,
            self.inner.params.m_max,
            self.inner.params.ef_construction
        );

        if self.inner.vectors.len() == 0 {
            return Ok(HNSW::from_builder(
                self,
                self.inner.entry_point,
                self.inner.vectors.metric_type(),
                self.inner.params.use_select_heuristic,
            ));
        }

        let parallel_limit = self
            .inner
            .params
            .parallel_limit
            .unwrap_or_else(num_cpus::get);
        let mut tasks = Vec::with_capacity(parallel_limit);
        let chunk_size = (self.inner.vectors.len() - 1).div_ceil(parallel_limit);
        for chunk in &(1..self.inner.vectors.len()).chunks(chunk_size) {
            let chunk = chunk.collect_vec();
            let inner = self.inner.clone();
            tasks.push(spawn_cpu(move || {
                for node in chunk {
                    inner.insert(node as u32)?;
                }
                Result::Ok(())
            }));
        }

        futures::future::try_join_all(tasks).await?;

        Ok(HNSW::from_builder(
            self,
            self.inner.entry_point,
            self.inner.vectors.metric_type(),
            self.inner.params.use_select_heuristic,
        ))
    }
}

struct HNSWBuilderInner {
    params: HnswBuildParams,

    /// Vector storage for the graph.
    vectors: Arc<dyn VectorStorage>,

    nodes: Arc<RwLock<Vec<GraphBuilderNode>>>,
    level_count: Vec<AtomicUsize>,

    entry_point: u32,
}

impl HNSWBuilderInner {
    pub fn num_levels(&self) -> usize {
        self.params.max_level as usize
    }

    pub fn num_nodes(&self, level: usize) -> usize {
        self.level_count[level].load(Ordering::Relaxed)
    }

    pub fn nodes(&self) -> Arc<RwLock<Vec<GraphBuilderNode>>> {
        self.nodes.clone()
    }

    pub fn storage(&self) -> Arc<dyn VectorStorage> {
        self.vectors.clone()
    }

    /// Create a new [`HNSWBuilder`] with prepared params and in memory vector storage.
    pub fn with_params(params: HnswBuildParams, vectors: Arc<dyn VectorStorage>) -> Self {
        let len = vectors.len();
        let max_level = params.max_level;

        let mut level_count = Vec::with_capacity(max_level as usize);
        for _ in 0..max_level {
            level_count.push(AtomicUsize::new(0));
        }

        let builder = Self {
            params,
            vectors,
            nodes: Arc::new(RwLock::new(Vec::with_capacity(len))),
            level_count,
            entry_point: 0,
        };

        {
            let mut nodes = builder.nodes.write().unwrap();
            nodes.push(GraphBuilderNode::new(0, max_level as usize));
            for i in 1..len {
                nodes.push(GraphBuilderNode::new(
                    i as u32,
                    builder.random_level() as usize + 1,
                ));
            }
        }

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
    fn insert(&self, node: u32) -> Result<()> {
        let target_level = self.nodes.read().unwrap()[node as usize]
            .level_neighbors
            .len() as u16
            - 1;
        let mut ep = OrderedNode::new(
            self.entry_point,
            self.vectors.distance_between(node, self.entry_point).into(),
        );

        //
        // Search for entry point in paper.
        // ```
        //   for l_c in (L..l+1) {
        //     W = Search-Layer(q, ep, ef=1, l_c)
        //    ep = Select-Neighbors(W, 1)
        //  }
        // ```
        let dist_calc = self.vectors.dist_calculator_from_id(node);
        for level in (target_level + 1..self.params.max_level).rev() {
            let cur_level = HnswLevelView::new(level, self);
            ep = greedy_search(&cur_level, ep, dist_calc.as_ref())?;
        }

        for level in (0..=target_level).rev() {
            self.level_count[level as usize].fetch_add(1, Ordering::Relaxed);

            let (candidates, neighbors) = self.search_level(&ep, level, dist_calc.as_ref())?;
            for neighbor in neighbors {
                self.connect(node, neighbor.id, neighbor.dist, level);
                self.prune(neighbor.id, level);
            }

            ep = candidates[0].clone();
        }

        Ok(())
    }

    fn search_level(
        &self,
        ep: &OrderedNode,
        level: u16,
        dist_calc: &dyn DistCalculator,
    ) -> Result<(Vec<OrderedNode>, Vec<OrderedNode>)> {
        let cur_level = HnswLevelView::new(level, self);
        let candidates = beam_search(&cur_level, ep, self.params.ef_construction, dist_calc, None)?;

        let neighbors = if self.params.use_select_heuristic {
            select_neighbors_heuristic(&cur_level, &candidates, self.params.m).collect()
        } else {
            select_neighbors(&candidates, self.params.m)
                .cloned()
                .collect()
        };

        Ok((candidates, neighbors))
    }

    fn connect(&self, u: u32, v: u32, dist: OrderedFloat, level: u16) {
        let nodes = self.nodes.read().unwrap();
        nodes[u as usize].add_neighbor(v, dist, level);
        nodes[v as usize].add_neighbor(u, dist, level);
    }

    fn prune(&self, id: u32, level: u16) {
        let node = &self.nodes.read().unwrap()[id as usize];

        let mut level_neighbors = node.level_neighbors[level as usize].write().unwrap();
        if level_neighbors.len() <= self.params.m_max {
            return;
        }

        let neighbors = level_neighbors.iter().cloned().collect_vec();

        let level_view = HnswLevelView::new(level, self);
        let new_neighbors =
            select_neighbors_heuristic(&level_view, &neighbors, self.params.m_max).collect();

        *level_neighbors = new_neighbors;
    }
}

// View of a level in HNSW graph.
// This is used to iterate over neighbors in a specific level.
pub(crate) struct HnswLevelView<'a> {
    level: u16,
    builder: &'a HNSWBuilderInner,
}

impl<'a> HnswLevelView<'a> {
    fn new(level: u16, builder: &'a HNSWBuilderInner) -> Self {
        Self { level, builder }
    }
}

impl<'a> Graph for HnswLevelView<'a> {
    fn len(&self) -> usize {
        self.builder.level_count[self.level as usize].load(Ordering::Relaxed)
    }

    fn neighbors(&self, key: u32) -> Option<Box<dyn Iterator<Item = u32> + '_>> {
        let node = &self.builder.nodes.read().unwrap()[key as usize];

        Some(
            node.level_neighbors
                .get(self.level as usize)
                .map(|neighbors| {
                    let iter: Box<dyn Iterator<Item = u32>> = Box::new(
                        neighbors
                            .read()
                            .unwrap()
                            .clone()
                            .into_sorted_vec()
                            .into_iter()
                            .map(|n| n.id),
                    );
                    iter
                })
                .unwrap_or_else(|| {
                    let iter: Box<dyn Iterator<Item = u32>> = Box::new(std::iter::empty());
                    iter
                }),
        )
    }

    fn storage(&self) -> Arc<dyn VectorStorage> {
        self.builder.vectors.clone()
    }
}
