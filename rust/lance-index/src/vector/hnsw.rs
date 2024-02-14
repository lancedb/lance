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

use std::sync::Arc;
use std::{collections::BTreeMap, fmt::Debug};

use lance_core::Result;
use lance_linalg::distance::MetricType;

use super::graph::{Graph, OrderedFloat};
use crate::vector::graph::beam_search;

pub mod builder;

pub use builder::HNSWBuilder;

/// HNSW graph.
///
pub struct HNSW {
    layers: Vec<Arc<dyn Graph>>,
    metric_type: MetricType,
    /// Entry point of the graph.
    entry_point: u64,
}

impl Debug for HNSW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HNSW(max_layers: {}, metric={})",
            self.layers.len(),
            self.metric_type
        )
    }
}

impl HNSW {
    fn from_builder(
        layers: Vec<Arc<dyn Graph>>,
        entry_point: u64,
        metric_type: MetricType,
    ) -> Self {
        Self {
            layers,
            metric_type,
            entry_point,
        }
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
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(u64, f32)>> {
        let mut ep = vec![self.entry_point];
        let num_layers = self.layers.len();
        for layer in self.layers.iter().rev().take(num_layers - 1) {
            let candidates = beam_search(layer.as_ref(), &ep, query, 1)?;
            ep = select_neighbors(&candidates, 1).map(|(_, id)| id).collect();
        }
        let candidates = beam_search(self.layers[0].as_ref(), &ep, query, ef)?;
        Ok(select_neighbors(&candidates, k)
            .map(|(d, u)| (u, d.into()))
            .collect())
    }
}

/// Select neighbors from the ordered candidate list.
///
/// Algorithm 3 in the HNSW paper.
fn select_neighbors(
    orderd_candidates: &BTreeMap<OrderedFloat, u64>,
    k: usize,
) -> impl Iterator<Item = (OrderedFloat, u64)> + '_ {
    orderd_candidates.iter().take(k).map(|(&d, &u)| (d, u))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;

    use super::super::graph::OrderedFloat;
    use arrow_array::types::Float32Type;
    use lance_linalg::matrix::MatrixView;
    use lance_testing::datagen::generate_random_array;

    #[test]
    fn test_select_neighbors() {
        let candidates: BTreeMap<OrderedFloat, u64> =
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
        let mat = MatrixView::<Float32Type>::new(data.into(), DIM);
        let hnsw = HNSWBuilder::new(mat)
            .max_num_edges(MAX_EDGES)
            .ef_construction(50)
            .build()
            .unwrap();
        assert!(hnsw.layers.len() > 1);
        assert_eq!(hnsw.layers[0].len(), TOTAL);

        hnsw.layers.windows(2).for_each(|w| {
            let (prev, next) = (w[0].as_ref(), w[1].as_ref());
            assert!(prev.len() >= next.len());
        });

        hnsw.layers.iter().for_each(|layer| {
            for i in 0..TOTAL {
                // If the node exist on this layer, check its out-degree.
                if let Some(neighbors) = layer.neighbors(i as u64) {
                    assert!(neighbors.len() <= MAX_EDGES);
                }
            }
        });
    }

    fn ground_truth(mat: &MatrixView<Float32Type>, query: &[f32], k: usize) -> HashSet<u64> {
        let mut dists = vec![];
        for i in 0..mat.num_rows() {
            let dist = lance_linalg::distance::l2_distance(query, mat.row(i).unwrap());
            dists.push((dist, i as u64));
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
        let mat = MatrixView::<Float32Type>::new(data.into(), DIM);
        let q = mat.row(0).unwrap();

        let hnsw = HNSWBuilder::new(mat.clone())
            .max_num_edges(MAX_EDGES)
            .ef_construction(100)
            .build()
            .unwrap();

        let results: HashSet<_> = hnsw
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
