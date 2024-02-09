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

use lance_linalg::distance::MetricType;
use rand::{thread_rng, Rng};

use super::super::graph::beam_search;
use super::{select_neighbors, HNSW};
use crate::vector::graph::{builder::GraphBuilder, storage::VectorStorage};
use lance_core::Result;

/// Build a HNSW graph.
///
/// Currently, the HNSW graph is fully built in memory.
pub struct HNSWBuilder<V: VectorStorage<f32>> {
    /// max level of
    max_level: u16,

    /// max number of connections ifor each element per layers.
    m_max: usize,

    /// Size of the dynamic list for the candidates
    ef_construction: usize,

    metric_type: MetricType,

    vectors: V,

    levels: Vec<GraphBuilder<V>>,

    entry_point: u32,
}

impl<V: VectorStorage<f32> + 'static> HNSWBuilder<V> {
    pub fn new(vectors: V) -> Self {
        Self {
            max_level: 8,
            m_max: 16,
            ef_construction: 100,
            metric_type: MetricType::L2,
            vectors,
            levels: vec![],
            entry_point: 0,
        }
    }

    /// Metric type
    pub fn metric_type(mut self, metric_type: MetricType) -> Self {
        self.metric_type = metric_type;
        self
    }

    /// The maximum level of the graph.
    pub fn max_level(mut self, max_level: u16) -> Self {
        self.max_level = max_level;
        self
    }

    /// The maximum number of connections for each node per layer.
    pub fn max_num_edges(mut self, m_max: usize) -> Self {
        self.m_max = m_max;
        self
    }

    /// Number of candidates to be considered when searching for the nearest neighbors
    /// during the construction of the graph.
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    fn random_level(&self) -> u16 {
        let mut rng = thread_rng();
        (rng.gen::<f32>().ln() * self.max_level as f32).floor() as u16
    }

    /// Insert one node.
    fn insert(&mut self, node: u32) -> Result<()> {
        let vector = self.vectors.get(node as usize);
        let level = self.random_level();

        let levels_to_search = self.levels.len() - level as usize;
        let mut ep = vec![self.entry_point];

        //
        // Search for entry point in paper.
        // ```
        //   for l_c in (L..l+1) {
        //     W = Search-Layer(q, ep, ef=1, l_c)
        //    ep = Select-Neighbors(W, 1)
        //  }
        // ```
        for cur_level in self.levels.iter().rev().take(levels_to_search) {
            let candidates = beam_search(cur_level, &ep, vector, 1)?;
            let neighbours = select_neighbors(&candidates, 1);
            assert!(neighbours.len() == 1);
            ep = vec![neighbours[0].1];
        }

        for cur_level in self.levels.iter_mut().rev().take(levels_to_search) {
            let candidates = beam_search(cur_level, &ep, vector, self.ef_construction)?;
            let neighbours = select_neighbors(&candidates, self.m_max);
            for (dist, nb) in neighbours.iter() {
                cur_level.bi_connect(node, *nb)?;
            }
            for (_, nb) in neighbours {
                cur_level.prune(nb, self.m_max)?;
            }
            ep = candidates.values().copied().collect::<Vec<_>>();
        }

        while level > self.levels.len() as u16 {
            let mut level =
                GraphBuilder::<V>::new(self.vectors.clone()).metric_type(self.metric_type);
            level.insert(node);
            self.levels.push(level);
        }

        Ok(())
    }

    pub fn build(&mut self) -> HNSW {
        let mut levels = Vec::with_capacity(self.max_level as usize);
        let level = GraphBuilder::<V>::new(self.vectors.clone()).metric_type(self.metric_type);
        levels.push(level);
        let mut entry_point = 0;
        let mut max_level = 0;
        levels.get_mut(0).unwrap().insert(0);

        for i in 1..self.vectors.len() {
            self.insert(i as u32);
        }
        unimplemented!()
    }
}
