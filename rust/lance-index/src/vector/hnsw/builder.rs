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

use std::cmp::min;
use std::collections::HashMap;
use std::sync::Arc;

use lance_core::Result;
use lance_linalg::distance::MetricType;
use rand::{thread_rng, Rng};

use super::{select_neighbors, HNSW};
use crate::vector::graph::{beam_search, builder::GraphBuilder, storage::VectorStorage, Graph};

/// Build a HNSW graph.
///
/// Currently, the HNSW graph is fully built in memory.
pub struct HNSWBuilder<V: VectorStorage<f32>> {
    /// max level of
    max_level: u16,

    /// M_l parameter in the paper.
    m_level_decay: f32,

    /// max number of connections ifor each element per layers.
    m_max: usize,

    /// Size of the dynamic list for the candidates
    ef_construction: usize,

    /// Metric type to compute the distance.
    metric_type: MetricType,

    /// Vector storage for the graph.
    vectors: V,

    levels: Vec<GraphBuilder<V>>,

    entry_point: u64,
}

impl<V: VectorStorage<f32> + 'static> HNSWBuilder<V> {
    pub fn new(vectors: V) -> Self {
        Self {
            max_level: 8,
            m_max: 32,
            ef_construction: 100,
            metric_type: MetricType::L2,
            vectors,
            levels: vec![],
            entry_point: 0,
            m_level_decay: 1.0 / 8_f32.ln(),
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
        self.m_level_decay = 1.0 / (max_level as f32).ln();
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

    /// new node's level
    ///
    /// See paper `Algorithm 1`
    fn random_level(&self) -> u16 {
        let mut rng = thread_rng();
        let r = rng.gen::<f32>();
        min(
            (-r.ln() * self.m_level_decay).floor() as u16,
            self.max_level,
        )
    }

    /// Insert one node.
    fn insert(&mut self, node: u64) -> Result<()> {
        let vector = self.vectors.get(node as usize);
        let level = self.random_level();

        let levels_to_search = if self.levels.len() > level as usize {
            self.levels.len() - level as usize - 1
        } else {
            0
        };
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
            let candidates = beam_search(cur_level, &ep, vector, self.ef_construction)?;
            let neighbours = select_neighbors(&candidates, 1);
            ep = neighbours.map(|(_, id)| id).collect();
        }
        for cur_level in self.levels.iter_mut().rev().skip(levels_to_search) {
            cur_level.insert(node);
            let candidates = beam_search(cur_level, &ep, vector, self.ef_construction)?;
            let neighbours = select_neighbors(&candidates, self.m_max).collect::<Vec<_>>();
            for (_, nb) in neighbours.iter() {
                cur_level.connect(node, *nb)?;
            }
            for (_, nb) in neighbours {
                cur_level.prune(nb, self.m_max)?;
            }
            ep = candidates.values().copied().collect::<Vec<_>>();
        }

        if level > self.levels.len() as u16 {
            self.entry_point = node;
        }

        Ok(())
    }

    /// Build a sealed HNSW graph.
    pub fn build(&mut self) -> Result<HNSW> {
        log::info!(
            "Building HNSW graph: metric_type={}, max_levels={}, m_max={}, ef_construction={}",
            self.metric_type,
            self.max_level,
            self.m_max,
            self.ef_construction
        );

        for _ in 0..self.max_level {
            let mut level =
                GraphBuilder::<V>::new(self.vectors.clone()).metric_type(self.metric_type);
            level.insert(0);
            self.levels.push(level);
        }

        for i in 1..self.vectors.len() {
            self.insert(i as u64)?;
        }

        remapping_levels(&mut self.levels);

        let graphs = self
            .levels
            .iter()
            .map(|l| l.build().into())
            .collect::<Vec<Arc<dyn Graph>>>();
        Ok(HNSW::from_builder(
            graphs,
            self.entry_point,
            self.metric_type,
        ))
    }
}

/// Because each level is stored as a separate continous RecordBatch. We need to remap the pointers
/// to the nodes in the previous level to the index in the current RecordBatch.
fn remapping_levels<V: VectorStorage<f32> + 'static>(levels: &mut [GraphBuilder<V>]) {
    for i in 1..levels.len() {
        let prev_level = &levels[i - 1];
        let mapping = prev_level
            .nodes
            .keys()
            .enumerate()
            .map(|(i, &id)| (id, i as u64))
            .collect::<HashMap<_, _>>();
        let cur_level = &mut levels[i];
        let current_mapping = cur_level
            .nodes
            .keys()
            .enumerate()
            .map(|(idx, &id)| (id, idx as u64))
            .collect::<HashMap<_, _>>();
        for node in cur_level.nodes.values_mut() {
            node.set_pointer(*mapping.get(&node.id).expect("Expect the pointer exists"));

            // Remapping the neighbors within this level of graph.
            node.neighbors = node
                .neighbors
                .iter()
                .map(|(d, n)| {
                    (
                        *d,
                        *current_mapping.get(n).expect("Expect the pointer exists"),
                    )
                })
                .collect();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::types::Float32Type;
    use lance_linalg::matrix::MatrixView;
    use lance_testing::datagen::generate_random_array;

    #[test]
    fn test_remapping_levels() {
        let data = generate_random_array(8 * 100);
        let mat = MatrixView::<Float32Type>::new(Arc::new(data), 8);
        let mut level0 = GraphBuilder::new(mat.clone());
        for i in 0..100 {
            level0.insert(i as u64);
        }
        let mut level1 = GraphBuilder::new(mat.clone());
        for i in [0, 5, 10, 15, 20, 30, 40, 50] {
            level1.insert(i as u64);
        }
        let mut level2 = GraphBuilder::new(mat.clone());
        for i in [0, 10, 20, 50] {
            level2.insert(i as u64);
        }
        let mut levels = [level0, level1, level2];
        remapping_levels(&mut levels);
        assert_eq!(
            levels[1]
                .nodes
                .values()
                .map(|n| n.pointer)
                .collect::<Vec<_>>(),
            vec![0, 5, 10, 15, 20, 30, 40, 50]
        );
        assert_eq!(
            levels[2]
                .nodes
                .values()
                .map(|n| n.pointer)
                .collect::<Vec<_>>(),
            vec![0, 2, 4, 7]
        );
    }
}
